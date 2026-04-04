from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9:]+", _norm_text(s))


def _char_ngrams(s: str, n: int = 3) -> list[str]:
    x = f" {_norm_text(s)} "
    if len(x) < n:
        return [x]
    return [x[i : i + n] for i in range(len(x) - n + 1)]


def _stable_hash_idx(text: str, dim: int) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % dim


def vectorize(text: str, dim: int) -> torch.Tensor:
    v = torch.zeros(dim, dtype=torch.float32)
    for t in _tok(text):
        v[_stable_hash_idx(f"tok:{t}", dim)] += 1.0
    for g in _char_ngrams(text, n=3):
        v[_stable_hash_idx(f"c3:{g}", dim)] += 0.5
    return v


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", text)]


def _normalize_number_words(text: str) -> str:
    m = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
    }
    out = text
    for w, d in m.items():
        out = re.sub(rf"\b{w}\b", d, out, flags=re.I)
    return out


def _duration_minutes_from_text(text: str) -> int | None:
    m = re.search(
        r"(\d{1,2}):(\d{2})\s*(am|pm)?\b.*?(\d{1,2}):(\d{2})\s*(am|pm)?\b",
        text,
        flags=re.I,
    )
    if not m:
        return None
    h1, mm1, ampm1, h2, mm2, ampm2 = m.groups()
    h1_i, mm1_i, h2_i, mm2_i = int(h1), int(mm1), int(h2), int(mm2)

    def _to_24h(h: int, ampm: str | None) -> int:
        if ampm is None:
            return h
        a = ampm.lower()
        if a == "am":
            return 0 if h == 12 else h
        return 12 if h == 12 else h + 12

    t1 = _to_24h(h1_i, ampm1) * 60 + mm1_i
    t2 = _to_24h(h2_i, ampm2) * 60 + mm2_i
    if t2 < t1:
        t2 += 24 * 60
    return t2 - t1


def _weekday_shift(text: str) -> str | None:
    low = _normalize_number_words(_norm_text(text))
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    m = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", low)
    if not m:
        return None
    day = m.group(1)
    idx = days.index(day)
    ints = _parse_ints(low)
    off = ints[0] if ints else 1
    if "before" in low:
        off = -off
    tgt = days[(idx + off) % 7]
    return tgt.capitalize()


def _compute_answer_by_type(q: str, pred_type: str) -> str | None:
    low = _normalize_number_words(_norm_text(q))
    ints = _parse_ints(low)

    if pred_type in {"comparison", "comparison_max"} and len(ints) >= 2:
        return str(max(ints[0], ints[1]))
    if pred_type == "comparison_min" and len(ints) >= 2:
        return str(min(ints[0], ints[1]))
    if pred_type in {"abs_diff"} and len(ints) >= 2:
        return str(abs(ints[0] - ints[1]))
    if pred_type in {"time_delta", "time_delta_ampm"}:
        d = _duration_minutes_from_text(low)
        return str(d) if d is not None else None
    if pred_type == "weekday_offset":
        return _weekday_shift(low)
    if pred_type == "half_plus" and len(ints) >= 2:
        if "half of" in low:
            return str((ints[0] // 2) + ints[1])
        if "added to half of" in low:
            return str(ints[0] + (ints[1] // 2))
        return str(ints[0] + (ints[1] // 2))
    if pred_type == "add_phrase" and len(ints) >= 2:
        return str(ints[0] + ints[1])
    if pred_type == "division" and len(ints) >= 2 and ints[1] != 0:
        a, b = ints[0], ints[1]
        return str(a // b if a % b == 0 else a / b)
    if pred_type == "arith_bin" and len(ints) >= 2:
        a, b = ints[0], ints[1]
        if any(x in low for x in ["*", "multiply", "times"]):
            return str(a * b)
        if any(x in low for x in ["/", "divide", "divided", "over"]) and b != 0:
            return str(a // b if a % b == 0 else a / b)
        if any(x in low for x in ["-", "minus", "subtract"]):
            return str(a - b)
        return str(a + b)
    if pred_type == "multi_step":
        # Generic two-op fallback using first 3 ints and operation order in text.
        # Explicit short templates first.
        if "double" in low and ("subtract" in low or "take away" in low) and len(ints) >= 2:
            x, y = ints[0], ints[1]
            return str((x * 2) - y)
        if "triple" in low and ("subtract" in low or "take away" in low) and len(ints) >= 2:
            x, y = ints[0], ints[1]
            return str((x * 3) - y)
        if "double" in low and "add" in low and len(ints) >= 2:
            x, y = ints[0], ints[1]
            return str((x * 2) + y)
        if "triple" in low and "add" in low and len(ints) >= 2:
            x, y = ints[0], ints[1]
            return str((x * 3) + y)

        # 2-int special: "multiply a by b" should still work.
        if len(ints) == 2 and ("multiply" in low or "times" in low):
            return str(ints[0] * ints[1])

        if len(ints) < 3:
            return None
        x, y, z = ints[0], ints[1], ints[2]
        if "add" in low and ("multiply" in low or "times" in low) and low.find("add") < max(low.find("multiply"), low.find("times")):
            return str((x + y) * z)
        if ("multiply" in low or "times" in low) and "add" in low and min(
            [i for i in [low.find("multiply"), low.find("times")] if i >= 0]
        ) < low.find("add"):
            return str((x * y) + z)
        if ("subtract" in low or "take away" in low) and ("multiply" in low or "times" in low):
            return str((x - y) * z)
        if ("double" in low or "triple" in low) and ("subtract" in low or "add" in low):
            scale = 3 if "triple" in low else 2
            v = x * scale
            return str(v - y if "subtract" in low else v + y)
    return None


@dataclass
class LearnedSolverConfig:
    input_dim: int = 2048
    epochs: int = 120
    lr: float = 3e-2
    weight_decay: float = 1e-4


class LinearTypeHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_type_predictor(
    questions: list[str],
    labels: list[str],
    cfg: LearnedSolverConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    cfg = cfg or LearnedSolverConfig()
    uniq = sorted(set(labels))
    l2i = {l: i for i, l in enumerate(uniq)}
    y = torch.tensor([l2i[l] for l in labels], dtype=torch.long)
    x = torch.stack([vectorize(q, cfg.input_dim) for q in questions], dim=0)

    torch.manual_seed(seed)
    model = LinearTypeHead(cfg.input_dim, len(uniq))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for _ in range(cfg.epochs):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)
        train_acc = float((pred == y).float().mean().item())

    return {
        "state_dict": model.state_dict(),
        "labels": uniq,
        "input_dim": cfg.input_dim,
        "train_acc": train_acc,
    }


class LearnedTypeSolver:
    def __init__(self, ckpt_path: str):
        blob = torch.load(Path(ckpt_path), map_location="cpu")
        self.labels: list[str] = list(blob["labels"])
        self.input_dim = int(blob["input_dim"])
        self.model = LinearTypeHead(self.input_dim, len(self.labels))
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

    def solve(self, question: str) -> tuple[str, dict[str, Any]]:
        x = vectorize(question, self.input_dim).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)[0]
            probs = torch.softmax(logits, dim=0)
            idx = int(torch.argmax(probs).item())
        pred_type = self.labels[idx]
        ans = _compute_answer_by_type(question, pred_type)
        if ans is None:
            ans = "Unknown"
        return ans, {
            "mode": "learned_program",
            "pred_type": pred_type,
            "pred_type_conf": float(probs[idx].item()),
            "solver": "learned_type_predictor+typed_executor",
        }
