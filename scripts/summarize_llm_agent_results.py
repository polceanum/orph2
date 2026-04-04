#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def agg(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"mean": None, "std": None, "ci95_halfwidth": None}
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    std = math.sqrt(v)
    ci95 = 1.96 * std / math.sqrt(len(xs))
    return {"mean": m, "std": std, "ci95_halfwidth": ci95}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument(
        "--oracle",
        default=None,
        help="Optional oracle/reference run JSON produced by run_llm_agent_eval.py",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    first_blob = None
    for p in args.inputs:
        blob = json.loads(Path(p).read_text())
        if first_blob is None:
            first_blob = blob
        rows.append(
            {
                "file": p,
                "seed": blob.get("seed"),
                "config": blob.get("config"),
                "n_tasks": blob["summary"]["n_tasks"],
                "accuracy": blob["summary"]["accuracy"],
                "per_split_accuracy": blob["summary"].get("per_split_accuracy"),
                "model_provider": blob["settings"].get("model_provider"),
                "model_name": blob["settings"].get("model_name"),
                "agent_mode": blob["settings"].get("agent_mode"),
            }
        )

    # Random-chance context: uniform guessing over unique gold answers in eval set.
    # This is a conservative, task-set-level reference (not a learned baseline).
    random_chance_accuracy = None
    if first_blob is not None:
        golds = [str(p.get("gold_answer", "")).strip() for p in first_blob.get("predictions", [])]
        unique_golds = {g for g in golds if g}
        if unique_golds:
            random_chance_accuracy = 1.0 / float(len(unique_golds))

    oracle_row = None
    if args.oracle:
        oracle_blob = json.loads(Path(args.oracle).read_text())
        oracle_row = {
            "file": args.oracle,
            "seed": oracle_blob.get("seed"),
            "config": oracle_blob.get("config"),
            "n_tasks": oracle_blob["summary"]["n_tasks"],
            "accuracy": oracle_blob["summary"]["accuracy"],
            "per_split_accuracy": oracle_blob["summary"].get("per_split_accuracy"),
            "model_provider": oracle_blob["settings"].get("model_provider"),
            "model_name": oracle_blob["settings"].get("model_name"),
            "agent_mode": oracle_blob["settings"].get("agent_mode"),
        }

    out = {
        "n": len(rows),
        "runs": rows,
        "aggregate": {
            "accuracy": agg([float(r["accuracy"]) for r in rows]),
            "n_tasks": agg([float(r["n_tasks"]) for r in rows]),
        },
        "references": {
            "random_chance_uniform_over_unique_gold": random_chance_accuracy,
            "oracle": oracle_row,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
