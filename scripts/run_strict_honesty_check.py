#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BENCHMARKS = {
    "main": "benchmarks/external/gsm8k_main_test_oodheuristic_v0.jsonl",
    "typeholdout": "benchmarks/external/gsm8k_main_test_ood_typeholdout_v1.jsonl",
    "lengthholdout": "benchmarks/external/gsm8k_main_test_ood_lengthholdout_v1.jsonl",
}

METHODS = [
    "direct",
    "sota",
    "adaptive",
    "adaptive_tools",
    "symbolic_only",
    "adaptive_tools_strict",
    "symbolic_only_strict",
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--benchmarks", default="main,typeholdout,lengthholdout")
    ap.add_argument("--skip-iid-wall-validation", action="store_true")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    benchmarks = [x.strip() for x in args.benchmarks.split(",") if x.strip()]

    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    if not args.skip_iid_wall_validation:
        run([
            py,
            str(repo_root / "scripts" / "validate_strict_iid_rule_registry.py"),
            "--agent-path",
            str(repo_root / "llm_agent" / "agent.py"),
            "--registry-path",
            str(repo_root / "configs" / "llm_agent" / "iid_rule_registry_gsm8k_main.txt"),
            "--benchmark-path",
            str(repo_root / "benchmarks" / "external" / "gsm8k_main_test_oodheuristic_v0.jsonl"),
        ])

    for bench in benchmarks:
        if bench not in BENCHMARKS:
            raise ValueError(f"Unsupported benchmark key: {bench}")
        for seed in seeds:
            for method in METHODS:
                cfg = repo_root / "configs" / "llm_agent" / f"gsm8k_{bench}_mock_{method}.yaml"
                out = repo_root / "artifacts" / "llm_agent" / f"gsm8k_{bench}_mock_{method}_s{seed}_{args.tag}.json"
                run([
                    py,
                    str(repo_root / "scripts" / "run_llm_agent_eval.py"),
                    "--config",
                    str(cfg),
                    "--seed",
                    str(seed),
                    "--out",
                    str(out),
                ])

    artifact_template = f"artifacts/llm_agent/{{benchmark_key}}_{{method}}_s{{seed}}_{args.tag}.json"
    for bench in benchmarks:
        bench_path = BENCHMARKS[bench]
        run([
            py,
            str(repo_root / "scripts" / "compare_llm_agent_benchmark.py"),
            "--benchmark-path",
            bench_path,
            "--benchmark-key",
            f"gsm8k_{bench}_mock",
            "--seeds",
            args.seeds,
            "--methods",
            ",".join(METHODS),
            "--ref-method",
            "sota",
            "--artifact-template",
            artifact_template,
            "--out-json",
            str(repo_root / "artifacts" / "llm_agent" / f"gsm8k_{bench}_mock_matrix_s{args.seeds.replace(',', '')}_{args.tag}.json"),
            "--out-md",
            str(repo_root / "artifacts" / "llm_agent" / f"gsm8k_{bench}_mock_matrix_s{args.seeds.replace(',', '')}_{args.tag}.md"),
        ])


if __name__ == "__main__":
    main()