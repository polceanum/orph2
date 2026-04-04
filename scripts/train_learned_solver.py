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
from llm_agent.learned_solver import LearnedSolverConfig, train_type_predictor


def _label_from_task(task) -> str:
    t = str(task.metadata.get("type", "")).strip()
    if not t:
        raise ValueError(f"Task {task.task_id} missing metadata.type for learned training")
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--train-split", default="iid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input-dim", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()

    tasks = load_jsonl_benchmark(args.benchmark, split=args.train_split)
    if not tasks:
        raise ValueError("No tasks found for selected train split")
    questions = [t.question for t in tasks]
    labels = [_label_from_task(t) for t in tasks]

    cfg = LearnedSolverConfig(
        input_dim=args.input_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    blob = train_type_predictor(questions=questions, labels=labels, cfg=cfg, seed=args.seed)
    blob["train_split"] = args.train_split
    blob["benchmark"] = args.benchmark
    blob["seed"] = args.seed
    blob["n_train"] = len(tasks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out_path)
    report = {
        "out": str(out_path),
        "benchmark": args.benchmark,
        "train_split": args.train_split,
        "seed": args.seed,
        "n_train": len(tasks),
        "labels": blob["labels"],
        "train_acc": blob["train_acc"],
    }
    print(json.dumps(report, indent=2))
    print(f"Wrote learned solver checkpoint to {out_path}")


if __name__ == "__main__":
    main()
