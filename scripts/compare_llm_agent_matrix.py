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


def random_chance_from_benchmark(path: Path) -> float | None:
    uniq: set[str] = set()
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        ans = str(rec.get("answer", "")).strip()
        if ans:
            uniq.add(ans)
    if not uniq:
        return None
    return 1.0 / float(len(uniq))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", required=True, help="e.g. v2 v3 v4")
    ap.add_argument("--seeds", required=True, help="comma-separated seeds, e.g. 0,1,2")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--ref-method", default="sota")
    ap.add_argument(
        "--methods",
        default="direct,sota,adaptive,adaptive_tools",
        help="comma-separated method keys to include; valid: direct,sota,adaptive,adaptive_tools,symbolic_only,learned_program",
    )
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    all_methods = {
        "direct": "direct",
        "sota": "sota",
        "adaptive": "adaptive",
        "adaptive_tools": "adaptive_tools",
        "symbolic_only": "symbolic_only",
        "learned_program": "learned_program",
    }
    method_keys = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = {k: all_methods[k] for k in method_keys}

    out: dict[str, object] = {"benchmarks": {}, "global": {}}
    md_lines: list[str] = ["# LLM-Agent Matrix Comparison", ""]

    global_rows: list[dict[str, float]] = []
    for b in args.benchmarks:
        b_key = f"v{b}" if not str(b).startswith("v") else str(b)
        bnum = b_key[1:] if b_key.startswith("v") else b_key
        bpath = Path(f"benchmarks/local_reasoning_ood_{b_key}.jsonl")
        chance = random_chance_from_benchmark(bpath) if bpath.exists() else None

        bench_blob: dict[str, object] = {
            "benchmark": b_key,
            "benchmark_path": str(bpath),
            "random_chance": chance,
            "methods": {},
            "deltas_vs_ref": {},
        }
        by_method_seed: dict[str, dict[int, dict[str, float]]] = {}
        for m_name, m_token in methods.items():
            per_seed: dict[int, dict[str, float]] = {}
            for s in seeds:
                p = Path(f"artifacts/llm_agent/local_reasoning_ood_{b_key}_mock_{m_token}_s{s}.json")
                if not p.exists():
                    p = Path(f"artifacts/llm_agent/local_reasoning_ood_{b_key}_{m_token}_s{s}.json")
                blob = json.loads(p.read_text())
                per_seed[s] = {
                    "accuracy": float(blob["summary"]["accuracy"]),
                    "iid": float(blob["summary"]["per_split_accuracy"]["iid"]),
                    "ood": float(blob["summary"]["per_split_accuracy"]["ood"]),
                }
            by_method_seed[m_name] = per_seed
            acc = [per_seed[s]["accuracy"] for s in seeds]
            iid = [per_seed[s]["iid"] for s in seeds]
            ood = [per_seed[s]["ood"] for s in seeds]
            bench_blob["methods"][m_name] = {
                "accuracy": {"mean": mean(acc), "std": std_pop(acc), "ci95_half": ci95_half(acc)},
                "iid": {"mean": mean(iid), "std": std_pop(iid), "ci95_half": ci95_half(iid)},
                "ood": {"mean": mean(ood), "std": std_pop(ood), "ci95_half": ci95_half(ood)},
            }

        ref = args.ref_method
        for m_name in methods:
            if m_name == ref:
                continue
            d_acc = [by_method_seed[m_name][s]["accuracy"] - by_method_seed[ref][s]["accuracy"] for s in seeds]
            d_iid = [by_method_seed[m_name][s]["iid"] - by_method_seed[ref][s]["iid"] for s in seeds]
            d_ood = [by_method_seed[m_name][s]["ood"] - by_method_seed[ref][s]["ood"] for s in seeds]
            bench_blob["deltas_vs_ref"][m_name] = {
                "accuracy": {"mean": mean(d_acc), "std": std_pop(d_acc), "ci95_half": ci95_half(d_acc)},
                "iid": {"mean": mean(d_iid), "std": std_pop(d_iid), "ci95_half": ci95_half(d_iid)},
                "ood": {"mean": mean(d_ood), "std": std_pop(d_ood), "ci95_half": ci95_half(d_ood)},
            }
            global_rows.append(
                {
                    "benchmark": float(bnum),
                    "method": float(list(methods.keys()).index(m_name)),
                    "delta_ood_vs_ref": mean(d_ood),
                }
            )

        out["benchmarks"][b_key] = bench_blob
        md_lines.append(f"## {b_key}")
        md_lines.append("")
        md_lines.append(f"- Random-chance reference: `{chance}`")
        md_lines.append(f"- Reference method: `{ref}`")
        md_lines.append("")
        md_lines.append("| Method | Acc mean±CI | IID mean±CI | OOD mean±CI |")
        md_lines.append("|---|---:|---:|---:|")
        for m_name in methods:
            ms = bench_blob["methods"][m_name]
            md_lines.append(
                f"| {m_name} | "
                f"{ms['accuracy']['mean']:.3f}±{ms['accuracy']['ci95_half']:.3f} | "
                f"{ms['iid']['mean']:.3f}±{ms['iid']['ci95_half']:.3f} | "
                f"{ms['ood']['mean']:.3f}±{ms['ood']['ci95_half']:.3f} |"
            )
        md_lines.append("")
        md_lines.append(f"| Delta vs {ref} | Acc mean±CI | IID mean±CI | OOD mean±CI |")
        md_lines.append("|---|---:|---:|---:|")
        for m_name in methods:
            if m_name == ref:
                continue
            ds = bench_blob["deltas_vs_ref"][m_name]
            md_lines.append(
                f"| {m_name} - {ref} | "
                f"{ds['accuracy']['mean']:.3f}±{ds['accuracy']['ci95_half']:.3f} | "
                f"{ds['iid']['mean']:.3f}±{ds['iid']['ci95_half']:.3f} | "
                f"{ds['ood']['mean']:.3f}±{ds['ood']['ci95_half']:.3f} |"
            )
        md_lines.append("")

    # Simple cross-benchmark headline for first non-ref method on OOD
    headline_method = next((m for m in methods if m != ref), None)
    ood_lifts = []
    for b in out["benchmarks"].values():
        d = b["deltas_vs_ref"].get(headline_method) if headline_method else None
        if d:
            ood_lifts.append(float(d["ood"]["mean"]))
    out["global"] = {
        "headline_method": headline_method,
        "headline_vs_ref_ood_mean_across_benchmarks": mean(ood_lifts) if ood_lifts else None,
        "headline_vs_ref_ood_ci95_half_across_benchmarks": ci95_half(ood_lifts) if ood_lifts else None,
    }
    md_lines.append("## Headline")
    md_lines.append("")
    if headline_method is None or not ood_lifts:
        md_lines.append("- No non-reference methods provided; no headline delta computed.")
    else:
        md_lines.append(
            f"- {headline_method} OOD lift vs reference across listed benchmarks: "
            f"`{out['global']['headline_vs_ref_ood_mean_across_benchmarks']:.3f}`"
            f" ± `{out['global']['headline_vs_ref_ood_ci95_half_across_benchmarks']:.3f}`"
        )
    md_lines.append("")
    md_lines.append("Note: 3-seed CIs are directional only; treat as preliminary.")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")
    print(json.dumps(out, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
