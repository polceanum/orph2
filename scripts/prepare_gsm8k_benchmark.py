#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _extract_final_answer(ans: str) -> str:
    text = str(ans).strip()
    # GSM8K answers typically end with "#### <final_answer>"
    m = re.search(r"####\s*([^\n]+)\s*$", text)
    if m:
        text = m.group(1).strip()
    # Normalize common numeric formatting.
    text = text.replace(",", "").strip()
    return text


def _difficulty_bucket(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["percent", "%", "ratio", "proportion", "probability"]):
        return "ratio_percent"
    if any(k in q for k in ["hour", "minute", "day", "week", "month", "year", "time"]):
        return "time_rate"
    if any(k in q for k in ["each", "per", "every", "times"]):
        return "multiplicative"
    return "arith_general"


def _token_len(question: str) -> int:
    return len(re.findall(r"\w+", question.lower()))


def _n_numbers(question: str) -> int:
    return len(re.findall(r"-?\d+(?:\.\d+)?", question))


def _ood_split(question: str, mode: str, length_threshold: int | None = None) -> str:
    if mode == "none":
        return "test"
    # Deterministic local OOD probing splits.
    bucket = _difficulty_bucket(question)
    if mode == "heuristic":
        return "ood" if bucket in {"ratio_percent", "time_rate"} else "iid"
    if mode == "type_holdout_strict":
        n_num = _n_numbers(question)
        return "ood" if (bucket in {"ratio_percent", "time_rate"} or n_num >= 4) else "iid"
    if mode == "length_holdout":
        if length_threshold is None:
            raise ValueError("length_threshold required for length_holdout mode")
        return "ood" if _token_len(question) > length_threshold else "iid"
    raise ValueError(f"Unsupported ood split mode: {mode}")


def _load_from_hf(split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets package is required for --source hf. Install with: pip install datasets"
        ) from exc
    ds = load_dataset("gsm8k", "main", split=split)
    return [dict(x) for x in ds]


def _load_from_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ln in Path(path).read_text().splitlines():
        if ln.strip():
            rows.append(json.loads(ln))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "jsonl"], default="hf")
    ap.add_argument("--input-jsonl", default=None, help="Required when --source=jsonl")
    ap.add_argument("--hf-split", default="test", help="HuggingFace split, default test")
    ap.add_argument("--max-samples", type=int, default=0, help="0 means all")
    ap.add_argument(
        "--ood-split-mode",
        choices=["none", "heuristic", "type_holdout_strict", "length_holdout"],
        default="heuristic",
        help="Local OOD probing split strategy; not a canonical benchmark split.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.source == "jsonl":
        if not args.input_jsonl:
            raise ValueError("--input-jsonl is required when --source=jsonl")
        rows = _load_from_jsonl(args.input_jsonl)
    else:
        rows = _load_from_hf(args.hf_split)

    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    length_threshold: int | None = None
    if args.ood_split_mode == "length_holdout":
        lengths = sorted(_token_len(str(r.get("question", ""))) for r in rows if str(r.get("question", "")).strip())
        if lengths:
            idx = int(0.65 * (len(lengths) - 1))
            length_threshold = lengths[idx]

    out_rows: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        q = str(r.get("question", "")).strip()
        a = _extract_final_answer(str(r.get("answer", "")))
        if not q or not a:
            continue
        split = _ood_split(q, mode=args.ood_split_mode, length_threshold=length_threshold)
        out_rows.append(
            {
                "id": f"gsm8k-{i:06d}",
                "question": q,
                "answer": a,
                "split": split,
                "metadata": {
                    "source": "gsm8k_main",
                    "source_split": args.hf_split if args.source == "hf" else "jsonl",
                    "split_mode": args.ood_split_mode,
                    "type": _difficulty_bucket(q),
                    "token_len": _token_len(q),
                    "n_numbers": _n_numbers(q),
                },
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    summary = {
        "out": str(out_path),
        "n_rows": len(out_rows),
        "source": args.source,
        "hf_split": args.hf_split,
        "ood_split_mode": args.ood_split_mode,
        "length_threshold": length_threshold,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
