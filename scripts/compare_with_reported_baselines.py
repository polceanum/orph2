#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _normalize_to_percent(v: float, unit: str) -> float:
    u = unit.lower().strip()
    if u in {"percent", "%"}:
        return float(v)
    if u in {"fraction", "acc_0_1"}:
        return float(v) * 100.0
    raise ValueError(f"Unknown unit: {unit}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="docs/LITERATURE_BASELINE_REGISTRY.json")
    ap.add_argument("--benchmark-key", required=True)
    ap.add_argument("--run-json", required=True, help="Output JSON from run_llm_agent_eval.py")
    ap.add_argument("--metric", choices=["accuracy", "iid", "ood"], default="accuracy")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    reg = json.loads(Path(args.registry).read_text())
    bench = reg.get("benchmarks", {}).get(args.benchmark_key, {})
    refs = bench.get("reported_results", [])
    run = json.loads(Path(args.run_json).read_text())

    if args.metric == "accuracy":
        ours = float(run["summary"]["accuracy"])
    else:
        ours = float(run["summary"]["per_split_accuracy"][args.metric])
    ours_pct = ours * 100.0

    comparisons = []
    for r in refs:
        if "value" not in r:
            continue
        unit = str(r.get("unit", "percent"))
        ref_pct = _normalize_to_percent(float(r["value"]), unit)
        comparisons.append(
            {
                "name": r.get("name"),
                "reference_percent": ref_pct,
                "ours_percent": ours_pct,
                "delta_percent_points": ours_pct - ref_pct,
                "source_url": r.get("source_url"),
            }
        )

    out = {
        "benchmark_key": args.benchmark_key,
        "metric": args.metric,
        "run_json": args.run_json,
        "ours_percent": ours_pct,
        "n_reported_references": len(comparisons),
        "comparisons": comparisons,
        "notes": bench.get("notes"),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
