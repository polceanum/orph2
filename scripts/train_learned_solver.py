#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_agent.benchmarks import load_jsonl_benchmark
from llm_agent.eval import exact_match
from llm_agent.learned_solver import (
    LearnedSolverConfig,
    _EXECUTOR_TYPES,
    _compute_answer_by_type,
    train_type_predictor,
)


def _label_from_task(task) -> str:
    t = str(task.metadata.get("type", "")).strip()
    if not t:
        raise ValueError(f"Task {task.task_id} missing metadata.type for learned training")
    return t


def _executor_pseudo_label(task) -> str | None:
    matches: list[str] = []
    for t in _EXECUTOR_TYPES:
        ans = _compute_answer_by_type(task.question, t)
        if ans is not None and exact_match(ans, task.answer):
            matches.append(t)
    if not matches:
        return None
    # Prefer stable deterministic choice if multiple match.
    return sorted(set(matches))[0]


def _executor_mapped_label(task) -> str | None:
    t = str(task.metadata.get("type", "")).strip().lower()
    mapping = {
        "arith_general": "arith_bin",
        "multiplicative": "multi_step",
        "ratio_percent": "arith_bin",
        "time_rate": "multi_step",
    }
    return mapping.get(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--train-split", default="iid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--label-mode",
        choices=["metadata", "executor_pseudo", "executor_hybrid"],
        default="metadata",
    )
    ap.add_argument("--input-dim", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()

    benchmark_paths = [x.strip() for x in str(args.benchmark).split(",") if x.strip()]
    tasks = []
    for bp in benchmark_paths:
        tasks.extend(load_jsonl_benchmark(bp, split=args.train_split))
    if not tasks:
        raise ValueError("No tasks found for selected train split")
    if args.label_mode == "metadata":
        questions = [t.question for t in tasks]
        labels = [_label_from_task(t) for t in tasks]
    elif args.label_mode == "executor_pseudo":
        pairs: list[tuple[str, str]] = []
        for t in tasks:
            lab = _executor_pseudo_label(t)
            if lab is not None:
                pairs.append((t.question, lab))
        if not pairs:
            raise ValueError("No executor pseudo-labels found; cannot train learned solver")
        questions = [q for q, _ in pairs]
        labels = [l for _, l in pairs]
    else:
        # Prefer exact pseudo labels when available; backfill with mapped labels.
        pairs: list[tuple[str, str]] = []
        for t in tasks:
            lab = _executor_pseudo_label(t)
            if lab is None:
                lab = _executor_mapped_label(t)
            if lab is not None:
                pairs.append((t.question, lab))
        if not pairs:
            raise ValueError("No executor labels found for hybrid mode")
        questions = [q for q, _ in pairs]
        labels = [l for _, l in pairs]

    cfg = LearnedSolverConfig(
        input_dim=args.input_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    blob = train_type_predictor(questions=questions, labels=labels, cfg=cfg, seed=args.seed)
    blob["train_split"] = args.train_split
    blob["benchmark"] = args.benchmark
    blob["benchmark_paths"] = benchmark_paths
    blob["seed"] = args.seed
    blob["n_train"] = len(tasks)
    blob["label_mode"] = args.label_mode

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out_path)
    report = {
        "out": str(out_path),
        "benchmark": args.benchmark,
        "benchmark_paths": benchmark_paths,
        "train_split": args.train_split,
        "seed": args.seed,
        "n_train": len(questions),
        "label_mode": args.label_mode,
        "labels": blob["labels"],
        "train_acc": blob["train_acc"],
    }
    print(json.dumps(report, indent=2))
    print(f"Wrote learned solver checkpoint to {out_path}")


if __name__ == "__main__":
    main()
