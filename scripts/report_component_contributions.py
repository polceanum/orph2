#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def std_pop(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def ci95_half(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return 1.96 * std_pop(xs) / math.sqrt(len(xs))


def load_run(bench: str, method_token: str, seed: int) -> dict | None:
    p = Path(f"artifacts/llm_agent/local_reasoning_ood_{bench}_mock_{method_token}_s{seed}.json")
    if not p.exists():
        p = Path(f"artifacts/llm_agent/local_reasoning_ood_{bench}_{method_token}_s{seed}.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


def get_metric(run: dict, metric: str) -> float:
    s = run["summary"]
    if metric == "accuracy":
        return float(s["accuracy"])
    return float(s["per_split_accuracy"][metric])


def aggregate_delta(
    bench: str, seeds: list[int], left_token: str, right_token: str, metric: str
) -> dict[str, float] | None:
    vals: list[float] = []
    for sd in seeds:
        l = load_run(bench, left_token, sd)
        r = load_run(bench, right_token, sd)
        if l is None or r is None:
            return None
        vals.append(get_metric(l, metric) - get_metric(r, metric))
    return {"mean": mean(vals), "std": std_pop(vals), "ci95_half": ci95_half(vals)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", required=True, help="e.g. v2 v3 v4 v5 v6")
    ap.add_argument("--seeds", required=True, help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    benches = [b if b.startswith("v") else f"v{b}" for b in args.benchmarks]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    # Method tokens map to run filename tokens.
    methods = {
        "direct": "direct",
        "sota_sc_verifier": "sota",
        "adaptive_router": "adaptive",
        "adaptive_router_tools": "adaptive_tools",
        "symbolic_only": "symbolic_only",
        "learned_program": "learned_program",
    }
    # Contribution deltas are intentionally directional and modular.
    contributions = [
        ("rewrite_sc_verifier_over_direct", "sota_sc_verifier", "direct"),
        ("adaptive_over_sota", "adaptive_router", "sota_sc_verifier"),
        ("tools_over_adaptive", "adaptive_router_tools", "adaptive_router"),
        ("symbolic_over_tools", "symbolic_only", "adaptive_router_tools"),
        ("learned_over_sota", "learned_program", "sota_sc_verifier"),
        ("learned_over_symbolic", "learned_program", "symbolic_only"),
    ]
    metrics = ["accuracy", "iid", "ood"]

    out: dict[str, object] = {"benchmarks": {}, "global": {}}
    md: list[str] = ["# Component Contribution Report", ""]
    global_accum: dict[str, dict[str, list[float]]] = {
        c[0]: {m: [] for m in metrics} for c in contributions
    }

    for b in benches:
        b_blob: dict[str, object] = {"contributions": {}}
        md.append(f"## {b}")
        md.append("")
        md.append("| Contribution | Acc delta mean±CI | IID delta mean±CI | OOD delta mean±CI |")
        md.append("|---|---:|---:|---:|")

        for label, left, right in contributions:
            row: dict[str, dict[str, float] | None] = {}
            ok = True
            for m in metrics:
                agg = aggregate_delta(b, seeds, methods[left], methods[right], m)
                row[m] = agg
                if agg is None:
                    ok = False
                else:
                    global_accum[label][m].append(float(agg["mean"]))
            b_blob["contributions"][label] = {
                "left": left,
                "right": right,
                "metrics": row,
            }
            if ok:
                md.append(
                    f"| {label} | "
                    f"{row['accuracy']['mean']:.3f}±{row['accuracy']['ci95_half']:.3f} | "
                    f"{row['iid']['mean']:.3f}±{row['iid']['ci95_half']:.3f} | "
                    f"{row['ood']['mean']:.3f}±{row['ood']['ci95_half']:.3f} |"
                )
            else:
                md.append(f"| {label} | missing | missing | missing |")
        md.append("")
        out["benchmarks"][b] = b_blob

    g: dict[str, object] = {}
    md.append("## Cross-Benchmark Means")
    md.append("")
    md.append("| Contribution | Acc mean±CI | IID mean±CI | OOD mean±CI |")
    md.append("|---|---:|---:|---:|")
    for label, _, _ in contributions:
        row = {}
        for m in metrics:
            vals = global_accum[label][m]
            if vals:
                row[m] = {"mean": mean(vals), "std": std_pop(vals), "ci95_half": ci95_half(vals)}
            else:
                row[m] = None
        g[label] = row
        if row["accuracy"] is None:
            md.append(f"| {label} | missing | missing | missing |")
        else:
            md.append(
                f"| {label} | "
                f"{row['accuracy']['mean']:.3f}±{row['accuracy']['ci95_half']:.3f} | "
                f"{row['iid']['mean']:.3f}±{row['iid']['ci95_half']:.3f} | "
                f"{row['ood']['mean']:.3f}±{row['ood']['ci95_half']:.3f} |"
            )
    md.append("")
    md.append("Note: CI values are directional; use 5 seeds for stronger claims.")

    out["global"] = g
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n")
    print(json.dumps(out, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
