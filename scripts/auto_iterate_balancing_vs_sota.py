#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def run_capture_json(cmd: list[str]) -> dict[str, Any]:
    out = subprocess.check_output(cmd, text=True)
    # run_llm_agent_eval prints JSON summary then trailing line, parse first JSON object.
    start = out.find("{")
    end = out.rfind("}")
    if start >= 0 and end > start:
        return json.loads(out[start : end + 1])
    return {}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def save_yaml(path: Path, blob: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(blob, sort_keys=False))


def module_diversity_from_artifact(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    counts: dict[str, int] = {}
    for p in data.get("predictions", []):
        m = str(p.get("trace", {}).get("selected_module", "<none>"))
        counts[m] = counts.get(m, 0) + 1
    total = max(1, sum(counts.values()))
    fracs = {k: v / total for k, v in counts.items()}
    non_symbolic = 1.0 - fracs.get("symbolic", 0.0)
    fracs["non_symbolic_fraction"] = non_symbolic
    return fracs


def matrix_metrics(path: Path, method: str) -> dict[str, float]:
    d = json.loads(path.read_text())
    m = d["methods"][method]
    return {
        "acc": float(m["accuracy"]["mean"]),
        "iid": float(m["iid"]["mean"]),
        "ood": float(m["ood"]["mean"]),
    }


def choose_best_candidate(py: str, tag: str, base_cfgs: dict[str, Path]) -> dict[str, float]:
    # Conservative base set + nearby perturbations for length-focused tuning.
    base_candidates = [
        {
            "routing_conf_threshold": 0.60,
            "routing_fast_k": 3,
            "routing_agreement_weight": 0.20,
            "source_bias_fast": 0.15,
            "source_bias_sota": 0.20,
            "source_bias_learned": 0.25,
            "source_bias_symbolic": 0.18,
            "learned_min_confidence": 0.75,
            "long_question_token_threshold": 45,
            "long_question_learned_boost": 0.06,
            "long_question_sota_boost": 0.04,
            "long_question_symbolic_penalty": 0.03,
        },
        {
            "routing_conf_threshold": 0.55,
            "routing_fast_k": 5,
            "routing_agreement_weight": 0.25,
            "source_bias_fast": 0.10,
            "source_bias_sota": 0.22,
            "source_bias_learned": 0.30,
            "source_bias_symbolic": 0.16,
            "learned_min_confidence": 0.80,
            "long_question_token_threshold": 42,
            "long_question_learned_boost": 0.08,
            "long_question_sota_boost": 0.05,
            "long_question_symbolic_penalty": 0.03,
        },
        {
            "routing_conf_threshold": 0.67,
            "routing_fast_k": 3,
            "routing_agreement_weight": 0.30,
            "source_bias_fast": 0.12,
            "source_bias_sota": 0.18,
            "source_bias_learned": 0.35,
            "source_bias_symbolic": 0.14,
            "learned_min_confidence": 0.85,
            "long_question_token_threshold": 40,
            "long_question_learned_boost": 0.10,
            "long_question_sota_boost": 0.06,
            "long_question_symbolic_penalty": 0.04,
        },
        {
            "routing_conf_threshold": 0.60,
            "routing_fast_k": 3,
            "routing_agreement_weight": 0.20,
            "source_bias_fast": 0.15,
            "source_bias_sota": 0.20,
            "source_bias_learned": 0.10,
            "source_bias_symbolic": 0.28,
            "learned_min_confidence": 0.90,
            "long_question_token_threshold": 45,
            "long_question_learned_boost": 0.08,
            "long_question_sota_boost": 0.05,
            "long_question_symbolic_penalty": 0.04,
        },
        {
            "routing_conf_threshold": 0.67,
            "routing_fast_k": 3,
            "routing_agreement_weight": 0.20,
            "source_bias_fast": 0.15,
            "source_bias_sota": 0.20,
            "source_bias_learned": 0.05,
            "source_bias_symbolic": 0.30,
            "learned_min_confidence": 0.95,
            "long_question_token_threshold": 42,
            "long_question_learned_boost": 0.06,
            "long_question_sota_boost": 0.04,
            "long_question_symbolic_penalty": 0.05,
        },
    ]

    candidates: list[dict[str, float]] = list(base_candidates)
    deltas = [
        {"routing_conf_threshold": 0.55, "routing_fast_k": 5, "source_bias_learned": 0.12, "source_bias_symbolic": 0.24, "learned_min_confidence": 0.85, "long_question_token_threshold": 40, "long_question_learned_boost": 0.10, "long_question_sota_boost": 0.06, "long_question_symbolic_penalty": 0.04},
        {"routing_conf_threshold": 0.60, "routing_fast_k": 5, "source_bias_learned": 0.08, "source_bias_symbolic": 0.30, "learned_min_confidence": 0.90, "long_question_token_threshold": 42, "long_question_learned_boost": 0.06, "long_question_sota_boost": 0.04, "long_question_symbolic_penalty": 0.05},
        {"routing_conf_threshold": 0.67, "routing_fast_k": 3, "source_bias_learned": 0.15, "source_bias_symbolic": 0.24, "learned_min_confidence": 0.85, "long_question_token_threshold": 45, "long_question_learned_boost": 0.08, "long_question_sota_boost": 0.05, "long_question_symbolic_penalty": 0.03},
        {"routing_conf_threshold": 0.55, "routing_fast_k": 3, "source_bias_learned": 0.05, "source_bias_symbolic": 0.32, "learned_min_confidence": 0.95, "long_question_token_threshold": 40, "long_question_learned_boost": 0.12, "long_question_sota_boost": 0.06, "long_question_symbolic_penalty": 0.06},
    ]
    for b in base_candidates:
        for d in deltas:
            c = dict(b)
            c.update(d)
            if c not in candidates:
                candidates.append(c)

    tmp_dir = ROOT / "artifacts" / "llm_agent" / "auto_iter"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    best = None
    for i, cand in enumerate(candidates):
        oods: dict[str, float] = {}
        non_symbolic_fracs: list[float] = []
        for bench, cfg_path in base_cfgs.items():
            cfg = load_yaml(cfg_path)
            cfg.setdefault("agent", {}).update(cand)
            tmp_cfg = tmp_dir / f"cand_{tag}_{i}_{bench}.yaml"
            save_yaml(tmp_cfg, cfg)
            out = ROOT / "artifacts" / "llm_agent" / f"gsm8k_{bench}_mock_adaptive_tools_strict_s0_auto_cand_{tag}_{i}.json"
            run(
                [
                    py,
                    str(ROOT / "scripts" / "run_llm_agent_eval.py"),
                    "--config",
                    str(tmp_cfg),
                    "--seed",
                    "0",
                    "--out",
                    str(out),
                ]
            )
            blob = json.loads(out.read_text())
            oods[bench] = float(blob["summary"]["per_split_accuracy"].get("ood", 0.0))
            div = module_diversity_from_artifact(out)
            non_symbolic_fracs.append(div.get("non_symbolic_fraction", 0.0))

        avg_ood = sum(oods.values()) / max(1, len(oods))
        length_ood = oods.get("lengthholdout", 0.0)
        avg_non_symbolic = sum(non_symbolic_fracs) / max(1, len(non_symbolic_fracs))
        score = avg_ood + 0.35 * length_ood + 0.05 * avg_non_symbolic
        rec = {
            "cand": cand,
            "oods": oods,
            "avg_ood": avg_ood,
            "score": score,
            "avg_non_symbolic": avg_non_symbolic,
        }
        if best is None or rec["score"] > best["score"]:
            best = rec

    assert best is not None
    return best["cand"]


def apply_candidate_to_strict_configs(cand: dict[str, float]) -> None:
    for bench in ["main", "typeholdout", "lengthholdout"]:
        p = ROOT / "configs" / "llm_agent" / f"gsm8k_{bench}_mock_adaptive_tools_strict.yaml"
        cfg = load_yaml(p)
        cfg.setdefault("agent", {}).update(cand)
        save_yaml(p, cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag-prefix", default="auto_balance")
    ap.add_argument("--target-main-ood", type=float, default=0.15)
    ap.add_argument("--target-type-ood", type=float, default=0.12)
    ap.add_argument("--target-length-ood", type=float, default=0.05)
    ap.add_argument("--train-class-balance", action="store_true")
    ap.add_argument("--train-augment-shape", action="store_true")
    args = ap.parse_args()

    py = sys.executable

    # Ensure learned checkpoint exists (IID-only training).
    learned_ckpt = ROOT / "artifacts" / "llm_agent" / "learned" / "gsm8k_main_iid_typehead_s0.pt"
    train_cmd = [
        py,
        str(ROOT / "scripts" / "train_learned_solver.py"),
        "--benchmark",
        ",".join(
            [
                str(ROOT / "benchmarks" / "external" / "gsm8k_main_test_oodheuristic_v0.jsonl"),
                str(ROOT / "benchmarks" / "external" / "gsm8k_main_test_ood_typeholdout_v1.jsonl"),
                str(ROOT / "benchmarks" / "external" / "gsm8k_main_test_ood_lengthholdout_v1.jsonl"),
            ]
        ),
        "--train-split",
        "iid",
        "--label-mode",
        "executor_hybrid",
        "--seed",
        "0",
        "--out",
        str(learned_ckpt),
    ]
    if args.train_class_balance:
        train_cmd.append("--class-balance")
    if args.train_augment_shape:
        train_cmd.append("--augment-shape")
    run(train_cmd)

    history: list[dict[str, Any]] = []

    base_cfgs = {
        "main": ROOT / "configs" / "llm_agent" / "gsm8k_main_mock_adaptive_tools_strict.yaml",
        "typeholdout": ROOT / "configs" / "llm_agent" / "gsm8k_typeholdout_mock_adaptive_tools_strict.yaml",
        "lengthholdout": ROOT / "configs" / "llm_agent" / "gsm8k_lengthholdout_mock_adaptive_tools_strict.yaml",
    }

    for r in range(1, args.rounds + 1):
        tag = f"{args.tag_prefix}_r{r}_20260406"

        cand = choose_best_candidate(py=py, tag=tag, base_cfgs=base_cfgs)
        apply_candidate_to_strict_configs(cand)

        # Run strict matrix with regular SOTA comparison included.
        run(
            [
                py,
                str(ROOT / "scripts" / "run_strict_honesty_check.py"),
                "--tag",
                tag,
                "--seeds",
                args.seeds,
            ]
        )

        main_m = matrix_metrics(
            ROOT / "artifacts" / "llm_agent" / f"gsm8k_main_mock_matrix_s{args.seeds.replace(',', '')}_{tag}.json",
            "adaptive_tools_strict",
        )
        type_m = matrix_metrics(
            ROOT / "artifacts" / "llm_agent" / f"gsm8k_typeholdout_mock_matrix_s{args.seeds.replace(',', '')}_{tag}.json",
            "adaptive_tools_strict",
        )
        len_m = matrix_metrics(
            ROOT / "artifacts" / "llm_agent" / f"gsm8k_lengthholdout_mock_matrix_s{args.seeds.replace(',', '')}_{tag}.json",
            "adaptive_tools_strict",
        )
        sota_main = matrix_metrics(
            ROOT / "artifacts" / "llm_agent" / f"gsm8k_main_mock_matrix_s{args.seeds.replace(',', '')}_{tag}.json",
            "sota",
        )

        rec = {
            "round": r,
            "tag": tag,
            "candidate": cand,
            "main": main_m,
            "typeholdout": type_m,
            "lengthholdout": len_m,
            "sota_main": sota_main,
            "delta_vs_sota_main_ood": main_m["ood"] - sota_main["ood"],
        }
        history.append(rec)

        if (
            main_m["ood"] >= args.target_main_ood
            and type_m["ood"] >= args.target_type_ood
            and len_m["ood"] >= args.target_length_ood
        ):
            break

    out = {
        "config": {
            "rounds": args.rounds,
            "seeds": args.seeds,
            "tag_prefix": args.tag_prefix,
            "targets": {
                "main_ood": args.target_main_ood,
                "type_ood": args.target_type_ood,
                "length_ood": args.target_length_ood,
            },
        },
        "history": history,
    }
    out_path = ROOT / "artifacts" / "llm_agent" / f"{args.tag_prefix}_history_20260406.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote auto-iteration history to {out_path}")


if __name__ == "__main__":
    main()
