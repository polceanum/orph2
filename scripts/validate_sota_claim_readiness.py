#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(bench: str, method_token: str, seed: int) -> dict | None:
    p = Path(f"artifacts/llm_agent/local_reasoning_ood_{bench}_mock_{method_token}_s{seed}.json")
    if not p.exists():
        p = Path(f"artifacts/llm_agent/local_reasoning_ood_{bench}_{method_token}_s{seed}.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-key", required=True, help="e.g. mmlu, gsm8k, bbh")
    ap.add_argument("--local-benchmark", required=True, help="e.g. v6 for local_reasoning_ood_v6")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default="sota,learned_program,symbolic_only")
    ap.add_argument("--registry", default="docs/LITERATURE_BASELINE_REGISTRY.json")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    registry = json.loads(Path(args.registry).read_text())
    bench_meta = registry.get("benchmarks", {}).get(args.benchmark_key, {})
    claim_rules = registry.get("claim_rules", {})
    min_refs = int(claim_rules.get("min_independent_reported_baselines_per_benchmark", 2))
    require_non_mock = bool(claim_rules.get("require_non_mock_provider", True))

    failures: list[str] = []
    checks: dict[str, object] = {}

    refs = bench_meta.get("reported_results", [])
    n_refs = sum(1 for r in refs if r.get("value") is not None and r.get("source_url"))
    checks["reported_baselines_count"] = n_refs
    checks["required_reported_baselines_count"] = min_refs
    if n_refs < min_refs:
        failures.append(
            f"Literature references incomplete: have {n_refs} reported baselines with values, require {min_refs}."
        )

    provider_rows: list[dict[str, object]] = []
    for m in methods:
        for sd in seeds:
            run = load_run(args.local_benchmark, m, sd)
            if run is None:
                failures.append(f"Missing run artifact for method={m} seed={sd}.")
                continue
            provider = run.get("settings", {}).get("model_provider")
            provider_rows.append({"method": m, "seed": sd, "provider": provider})
            if require_non_mock and str(provider).lower() == "mock":
                failures.append(f"Mock provider detected for method={m} seed={sd}; non-mock required for SOTA claims.")
    checks["providers"] = provider_rows

    if str(args.local_benchmark).startswith("v"):
        failures.append(
            "Current benchmark is local_reasoning_ood synthetic fixture; not acceptable as standalone SOTA evidence."
        )

    status = "pass" if not failures else "fail"
    out = {
        "status": status,
        "benchmark_key": args.benchmark_key,
        "local_benchmark": args.local_benchmark,
        "methods": methods,
        "seeds": seeds,
        "checks": checks,
        "failures": failures,
        "next_actions": [
            "Run same methods on external benchmark adapter with non-mock backend.",
            "Add >=2 paper-reported comparator baselines with protocol-matched settings.",
            "Re-run this validator before SOTA wording."
        ],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    raise SystemExit(0 if status == "pass" else 2)


if __name__ == "__main__":
    main()
