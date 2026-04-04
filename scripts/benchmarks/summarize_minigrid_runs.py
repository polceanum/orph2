#!/usr/bin/env python3
"""Summarize one or more Minigrid benchmark JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def aggregate(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "std": None}
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return {"mean": m, "std": math.sqrt(v)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
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

    out = {
        "n": len(rows),
        "runs": rows,
        "aggregate": {
            "iid_success_rate": aggregate([r["iid_success_rate"] for r in rows]),
            "ood_success_rate": aggregate([r["ood_success_rate"] for r in rows]),
            "ood_minus_iid": aggregate([r["ood_minus_iid"] for r in rows]),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
