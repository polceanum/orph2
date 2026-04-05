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
    ap.add_argument("--benchmark-path", required=True)
    ap.add_argument("--benchmark-key", required=True, help="e.g. gsm8k_main_mock")
    ap.add_argument("--seeds", required=True, help="comma-separated, e.g. 0,1,2")
    ap.add_argument(
        "--methods",
        default="direct,sota,adaptive,adaptive_tools,symbolic_only",
        help="comma-separated method keys",
    )
    ap.add_argument("--ref-method", default="sota")
    ap.add_argument(
        "--artifact-template",
        default="artifacts/llm_agent/{benchmark_key}_{method}_s{seed}.json",
        help="Path template with {benchmark_key}, {method}, {seed}",
    )
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    chance = random_chance_from_benchmark(Path(args.benchmark_path))

    by_method_seed: dict[str, dict[int, dict[str, float]]] = {}
    for method in methods:
        by_method_seed[method] = {}
        for seed in seeds:
            p = Path(
                args.artifact_template.format(
                    benchmark_key=args.benchmark_key,
                    method=method,
                    seed=seed,
                )
            )
            if not p.exists():
                raise FileNotFoundError(f"Missing artifact: {p}")
            blob = json.loads(p.read_text())
            by_method_seed[method][seed] = {
                "accuracy": float(blob["summary"]["accuracy"]),
                "iid": float(blob["summary"]["per_split_accuracy"].get("iid", float("nan"))),
                "ood": float(blob["summary"]["per_split_accuracy"].get("ood", float("nan"))),
            }

    out: dict[str, object] = {
        "benchmark_key": args.benchmark_key,
        "benchmark_path": args.benchmark_path,
        "random_chance": chance,
        "methods": {},
        "deltas_vs_ref": {},
        "ref_method": args.ref_method,
        "seeds": seeds,
    }
    md = [
        f"# Benchmark Comparison: {args.benchmark_key}",
        "",
        f"- Benchmark path: `{args.benchmark_path}`",
        f"- Random-chance reference: `{chance}`",
        f"- Reference method: `{args.ref_method}`",
        "",
        "| Method | Acc mean±CI | IID mean±CI | OOD mean±CI |",
        "|---|---:|---:|---:|",
    ]

    for method in methods:
        acc = [by_method_seed[method][s]["accuracy"] for s in seeds]
        iid = [by_method_seed[method][s]["iid"] for s in seeds]
        ood = [by_method_seed[method][s]["ood"] for s in seeds]
        out["methods"][method] = {
            "accuracy": {"mean": mean(acc), "std": std_pop(acc), "ci95_half": ci95_half(acc)},
            "iid": {"mean": mean(iid), "std": std_pop(iid), "ci95_half": ci95_half(iid)},
            "ood": {"mean": mean(ood), "std": std_pop(ood), "ci95_half": ci95_half(ood)},
        }
        ms = out["methods"][method]
        md.append(
            f"| {method} | "
            f"{ms['accuracy']['mean']:.3f}±{ms['accuracy']['ci95_half']:.3f} | "
            f"{ms['iid']['mean']:.3f}±{ms['iid']['ci95_half']:.3f} | "
            f"{ms['ood']['mean']:.3f}±{ms['ood']['ci95_half']:.3f} |"
        )

    md.extend(
        [
            "",
            f"| Delta vs {args.ref_method} | Acc mean±CI | IID mean±CI | OOD mean±CI |",
            "|---|---:|---:|---:|",
        ]
    )
    ref = args.ref_method
    for method in methods:
        if method == ref:
            continue
        d_acc = [by_method_seed[method][s]["accuracy"] - by_method_seed[ref][s]["accuracy"] for s in seeds]
        d_iid = [by_method_seed[method][s]["iid"] - by_method_seed[ref][s]["iid"] for s in seeds]
        d_ood = [by_method_seed[method][s]["ood"] - by_method_seed[ref][s]["ood"] for s in seeds]
        out["deltas_vs_ref"][method] = {
            "accuracy": {"mean": mean(d_acc), "std": std_pop(d_acc), "ci95_half": ci95_half(d_acc)},
            "iid": {"mean": mean(d_iid), "std": std_pop(d_iid), "ci95_half": ci95_half(d_iid)},
            "ood": {"mean": mean(d_ood), "std": std_pop(d_ood), "ci95_half": ci95_half(d_ood)},
        }
        ds = out["deltas_vs_ref"][method]
        md.append(
            f"| {method} - {ref} | "
            f"{ds['accuracy']['mean']:.3f}±{ds['accuracy']['ci95_half']:.3f} | "
            f"{ds['iid']['mean']:.3f}±{ds['iid']['ci95_half']:.3f} | "
            f"{ds['ood']['mean']:.3f}±{ds['ood']['ci95_half']:.3f} |"
        )

    md.append("")
    md.append("Note: 1-seed results are exploratory; CI values are directional only.")

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
