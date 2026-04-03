import argparse
import json
from pathlib import Path

import torch

from eval_baselines import (
    build_no_rules_model,
    build_random_probe_model,
    build_recurrent_baseline,
    build_structured_model,
    cfg_get,
    load_checkpoint,
    load_config,
    run_eval,
    sample_eval_episodes,
    set_seed,
)
from ood_solver.envs.config import build_env_from_cfg


def _metric_gap(iid: float | None, ood: float | None):
    if iid is None or ood is None:
        return None
    return ood - iid


def _metric_ratio(numer: float | None, denom: float | None, eps: float = 1e-8):
    if numer is None or denom is None:
        return None
    if abs(denom) < eps:
        return None
    return numer / denom


def _summarize_gap(iid_result: dict, ood_result: dict) -> dict:
    iid = iid_result["overall"]
    ood = ood_result["overall"]
    return {
        "seq_acc_gap": _metric_gap(iid.get("seq_acc"), ood.get("seq_acc")),
        "seq_acc_retention": _metric_ratio(ood.get("seq_acc"), iid.get("seq_acc")),
        "loss_gap": _metric_gap(iid.get("loss"), ood.get("loss")),
        "loss_ratio": _metric_ratio(ood.get("loss"), iid.get("loss")),
        "diagnostic_probe_acc_gap": _metric_gap(iid.get("diagnostic_probe_acc"), ood.get("diagnostic_probe_acc")),
        "diagnostic_probe_acc_retention": _metric_ratio(
            ood.get("diagnostic_probe_acc"), iid.get("diagnostic_probe_acc")
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--structured-ckpt", type=str, required=True)
    parser.add_argument("--random-probe-ckpt", type=str, default=None)
    parser.add_argument("--no-rules-ckpt", type=str, default=None)
    parser.add_argument("--recurrent-ckpt", type=str, default=None)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--out", type=str, default="artifacts/eval_ood_baselines.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(cfg_get(cfg, "seed", 0))
    set_seed(seed)

    iid_env = build_env_from_cfg(
        cfg,
        section="eval.id" if cfg_get(cfg, "eval.id", None) is not None else "env",
        seed=seed,
        seed_offset=int(cfg_get(cfg, "eval.seed_offset", 123)),
    )
    ood_section = "eval.ood"
    if cfg_get(cfg, ood_section, None) is None:
        raise ValueError(
            "Missing eval.ood section in config. Define OOD split, e.g. eval.ood.param_ranges or eval.ood.mechanisms."
        )
    ood_env = build_env_from_cfg(
        cfg,
        section=ood_section,
        seed=seed,
        seed_offset=int(cfg_get(cfg, "eval.ood.seed_offset", 999)),
    )

    batch_size = int(cfg_get(cfg, "train.batch_size", 32))
    iid_episodes = sample_eval_episodes(iid_env, batch_size=batch_size, num_batches=args.num_batches)
    ood_episodes = sample_eval_episodes(ood_env, batch_size=batch_size, num_batches=args.num_batches)

    results = {"iid": {}, "ood": {}, "gap": {}}

    structured_model = build_structured_model(cfg, device)
    load_checkpoint(structured_model, args.structured_ckpt)
    results["iid"]["structured"] = run_eval(structured_model, iid_env, iid_episodes, device, structured=True)
    results["ood"]["structured"] = run_eval(structured_model, ood_env, ood_episodes, device, structured=True)
    results["gap"]["structured"] = _summarize_gap(results["iid"]["structured"], results["ood"]["structured"])

    if args.random_probe_ckpt is not None:
        rp_model = build_random_probe_model(cfg, device)
        load_checkpoint(rp_model, args.random_probe_ckpt)
        results["iid"]["random_probe"] = run_eval(rp_model, iid_env, iid_episodes, device, structured=True)
        results["ood"]["random_probe"] = run_eval(rp_model, ood_env, ood_episodes, device, structured=True)
        results["gap"]["random_probe"] = _summarize_gap(results["iid"]["random_probe"], results["ood"]["random_probe"])

    if args.no_rules_ckpt is not None:
        nr_model = build_no_rules_model(cfg, device)
        load_checkpoint(nr_model, args.no_rules_ckpt)
        results["iid"]["no_rules"] = run_eval(nr_model, iid_env, iid_episodes, device, structured=True)
        results["ood"]["no_rules"] = run_eval(nr_model, ood_env, ood_episodes, device, structured=True)
        results["gap"]["no_rules"] = _summarize_gap(results["iid"]["no_rules"], results["ood"]["no_rules"])

    if args.recurrent_ckpt is not None:
        rec_model = build_recurrent_baseline(cfg, device)
        load_checkpoint(rec_model, args.recurrent_ckpt)
        results["iid"]["recurrent"] = run_eval(rec_model, iid_env, iid_episodes, device, structured=False)
        results["ood"]["recurrent"] = run_eval(rec_model, ood_env, ood_episodes, device, structured=False)
        results["gap"]["recurrent"] = _summarize_gap(results["iid"]["recurrent"], results["ood"]["recurrent"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote OOD eval to {out_path}")


if __name__ == "__main__":
    main()
