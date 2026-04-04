#!/usr/bin/env python3
"""Summarize one or more Minigrid benchmark JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def aggregate(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "std": None, "ci95_halfwidth": None}
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    std = math.sqrt(v)
    ci95 = 1.96 * std / math.sqrt(len(xs)) if len(xs) > 0 else None
    return {"mean": m, "std": std, "ci95_halfwidth": ci95}


def extract_ood(path: str) -> float:
    blob = json.loads(Path(path).read_text())
    return float(blob["summary"]["ood_success_rate_mean"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", default=None, help="Optional random baseline JSON for normalized OOD score.")
    ap.add_argument("--oracle", default=None, help="Optional oracle JSON for normalized OOD score.")
    args = ap.parse_args()

    rows = []
    for p in args.inputs:
        blob = json.loads(Path(p).read_text())
        rows.append(
            {
                "file": p,
                "seed": blob.get("seed"),
                "config": blob.get("config"),
                "iid_success_rate": blob["summary"]["iid_success_rate_mean"],
                "ood_success_rate": blob["summary"]["ood_success_rate_mean"],
                "ood_minus_iid": blob["summary"]["ood_minus_iid_success_rate"],
            }
        )

    out: dict[str, object] = {
        "n": len(rows),
        "runs": rows,
        "aggregate": {
            "iid_success_rate": aggregate([r["iid_success_rate"] for r in rows]),
            "ood_success_rate": aggregate([r["ood_success_rate"] for r in rows]),
            "ood_minus_iid": aggregate([r["ood_minus_iid"] for r in rows]),
        },
    }
    if args.random and args.oracle:
        random_ood = extract_ood(args.random)
        oracle_ood = extract_ood(args.oracle)
        denom = oracle_ood - random_ood
        ood_mean = out["aggregate"]["ood_success_rate"]["mean"]  # type: ignore[index]
        norm = None
        if ood_mean is not None and abs(denom) > 1e-12:
            norm = float((float(ood_mean) - random_ood) / denom)
        out["normalization"] = {
            "random_ood_success_rate": random_ood,
            "oracle_ood_success_rate": oracle_ood,
            "denominator_oracle_minus_random": denom,
            "normalized_ood_score": norm,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
