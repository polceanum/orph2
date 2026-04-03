import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path as _Path
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_seeds(seeds_text: str) -> list[int]:
    seeds = []
    for part in seeds_text.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds provided.")
    return seeds


def cfg_set(cfg: dict, key: str, value):
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def run_cmd(cmd: list[str], cwd: Path):
    # Force unbuffered Python output so long runs stream progress immediately.
    if cmd and _Path(cmd[0]).name.startswith("python") and "-u" not in cmd[1:3]:
        cmd = [cmd[0], "-u", *cmd[1:]]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def aggregate_seed_results(seed_results: dict[str, dict]) -> dict:
    models = {}
    for seed_data in seed_results.values():
        for model_name, model_result in seed_data.items():
            models.setdefault(model_name, []).append(model_result["overall"])

    summary = {}
    for model_name, rows in models.items():
        keys = sorted(rows[0].keys())
        metrics = {}
        for key in keys:
            vals = [r[key] for r in rows if r[key] is not None]
            if not vals:
                metrics[key] = {"mean": None, "std": None}
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            metrics[key] = {"mean": mean, "std": math.sqrt(var)}
        summary[model_name] = metrics
    return summary


def aggregate_ood_seed_results(seed_results: dict[str, dict], split: str) -> dict:
    models = {}
    for seed_data in seed_results.values():
        split_data = seed_data.get(split, {})
        for model_name, model_result in split_data.items():
            models.setdefault(model_name, []).append(model_result["overall"])

    summary = {}
    for model_name, rows in models.items():
        keys = sorted(rows[0].keys())
        metrics = {}
        for key in keys:
            vals = [r[key] for r in rows if r[key] is not None]
            if not vals:
                metrics[key] = {"mean": None, "std": None}
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            metrics[key] = {"mean": mean, "std": math.sqrt(var)}
        summary[model_name] = metrics
    return summary


def _mean_std(vals: list[float]) -> dict:
    if not vals:
        return {"mean": None, "std": None}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {"mean": mean, "std": math.sqrt(var)}


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _bootstrap_mean_summary(
    values: list[float],
    seed: int,
    num_bootstrap: int = 4000,
    alpha: float = 0.05,
) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "p_gt_zero": None,
            "p_lt_zero": None,
        }
    rng = random.Random(seed)
    n = len(values)
    obs_mean = sum(values) / n
    means = []
    for _ in range(num_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    ci_low = _percentile(means, lower_q)
    ci_high = _percentile(means, upper_q)
    p_gt_zero = sum(1 for m in means if m > 0.0) / len(means)
    p_lt_zero = sum(1 for m in means if m < 0.0) / len(means)
    return {
        "n": n,
        "mean": obs_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_gt_zero": p_gt_zero,
        "p_lt_zero": p_lt_zero,
    }


def _safe_ratio(numer: float | None, denom: float | None, eps: float = 1e-8):
    if numer is None or denom is None:
        return None
    if abs(denom) < eps:
        return None
    return numer / denom


def _get_batch_metric(model_blob: dict, metric: str) -> list[float]:
    batch = model_blob.get("batch_overall", {})
    vals = batch.get(metric, [])
    if vals:
        return [float(v) for v in vals]
    overall = model_blob.get("overall", {})
    v = overall.get(metric)
    return [float(v)] if v is not None else []


def summarize_structured_vs_baselines(
    per_seed: dict[str, dict],
    per_seed_ood: dict[str, dict] | None = None,
    bootstrap_seed: int = 20260402,
) -> dict:
    baselines = sorted(
        {
            model_name
            for seed_blob in per_seed.values()
            for model_name in seed_blob.keys()
            if model_name != "structured"
        }
    )
    out: dict[str, dict] = {}
    for baseline in baselines:
        iid_seed_deltas = []
        iid_batch_deltas = []
        ood_seed_deltas = []
        ood_batch_deltas = []
        gap_seed_deltas = []
        gap_batch_deltas = []
        retention_seed_deltas = []

        for seed_key in sorted(per_seed.keys(), key=int):
            seed_blob = per_seed[seed_key]
            if "structured" not in seed_blob or baseline not in seed_blob:
                continue
            s_iid = seed_blob["structured"]["overall"]["seq_acc"]
            b_iid = seed_blob[baseline]["overall"]["seq_acc"]
            iid_seed_deltas.append(float(s_iid - b_iid))

            s_iid_batches = _get_batch_metric(seed_blob["structured"], "seq_acc")
            b_iid_batches = _get_batch_metric(seed_blob[baseline], "seq_acc")
            n_iid = min(len(s_iid_batches), len(b_iid_batches))
            iid_batch_deltas.extend([s_iid_batches[i] - b_iid_batches[i] for i in range(n_iid)])

            if per_seed_ood is None or seed_key not in per_seed_ood:
                continue
            ood_blob = per_seed_ood[seed_key]
            if baseline not in ood_blob.get("iid", {}) or baseline not in ood_blob.get("ood", {}):
                continue
            s_ood = ood_blob["ood"]["structured"]["overall"]["seq_acc"]
            b_ood = ood_blob["ood"][baseline]["overall"]["seq_acc"]
            ood_seed_deltas.append(float(s_ood - b_ood))

            s_ood_batches = _get_batch_metric(ood_blob["ood"]["structured"], "seq_acc")
            b_ood_batches = _get_batch_metric(ood_blob["ood"][baseline], "seq_acc")
            n_ood = min(len(s_ood_batches), len(b_ood_batches))
            ood_batch_deltas.extend([s_ood_batches[i] - b_ood_batches[i] for i in range(n_ood)])

            s_gap = ood_blob["gap"]["structured"]["seq_acc_gap"]
            b_gap = ood_blob["gap"][baseline]["seq_acc_gap"]
            if s_gap is not None and b_gap is not None:
                gap_seed_deltas.append(float(s_gap - b_gap))

            n_gap = min(n_iid, n_ood)
            for i in range(n_gap):
                s_gap_b = s_ood_batches[i] - s_iid_batches[i]
                b_gap_b = b_ood_batches[i] - b_iid_batches[i]
                gap_batch_deltas.append(s_gap_b - b_gap_b)

            s_ret = _safe_ratio(s_ood, s_iid)
            b_ret = _safe_ratio(b_ood, b_iid)
            if s_ret is not None and b_ret is not None:
                retention_seed_deltas.append(float(s_ret - b_ret))

        out[baseline] = {
            "iid_seq_acc_delta": {
                "seed_level": _mean_std(iid_seed_deltas),
                "batch_bootstrap_95ci": _bootstrap_mean_summary(
                    iid_batch_deltas, seed=bootstrap_seed + 1
                ),
            },
            "ood_seq_acc_delta": {
                "seed_level": _mean_std(ood_seed_deltas),
                "batch_bootstrap_95ci": _bootstrap_mean_summary(
                    ood_batch_deltas, seed=bootstrap_seed + 2
                ),
            },
            "ood_gap_seq_acc_delta": {
                "seed_level": _mean_std(gap_seed_deltas),
                "batch_bootstrap_95ci": _bootstrap_mean_summary(
                    gap_batch_deltas, seed=bootstrap_seed + 3
                ),
            },
            "ood_retention_delta": {
                "seed_level": _mean_std(retention_seed_deltas),
            },
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/small_tuned.yaml")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--tag", type=str, default="small_tuned_suite")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text())
    seeds = parse_seeds(args.seeds)

    root_dir = PROJECT_ROOT / "artifacts" / "sweeps" / args.tag
    cfg_dir = root_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    per_seed_ood = {}
    has_ood = isinstance(base_cfg.get("eval", {}), dict) and isinstance(base_cfg.get("eval", {}).get("ood"), dict)
    for seed in seeds:
        cfg = json.loads(json.dumps(base_cfg))
        cfg_set(cfg, "seed", seed)
        seed_dir = root_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_cfg_path = cfg_dir / f"seed_{seed}.yaml"
        seed_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        structured_ckpt = seed_dir / "structured.pt"
        random_probe_ckpt = seed_dir / "random_probe.pt"
        no_rules_ckpt = seed_dir / "no_rules.pt"
        recurrent_ckpt = seed_dir / "recurrent.pt"
        eval_out = seed_dir / "eval.json"
        eval_ood_out = seed_dir / "eval_ood.json"

        run_cmd(
            [
                args.python,
                "scripts/train_structured.py",
                "--config",
                str(seed_cfg_path),
                "--save",
                str(structured_ckpt),
            ],
            PROJECT_ROOT,
        )
        run_cmd(
            [
                args.python,
                "scripts/train_random_probe.py",
                "--config",
                str(seed_cfg_path),
                "--save",
                str(random_probe_ckpt),
            ],
            PROJECT_ROOT,
        )
        run_cmd(
            [
                args.python,
                "scripts/train_no_rules.py",
                "--config",
                str(seed_cfg_path),
                "--save",
                str(no_rules_ckpt),
            ],
            PROJECT_ROOT,
        )
        run_cmd(
            [
                args.python,
                "scripts/train_recurrent.py",
                "--config",
                str(seed_cfg_path),
                "--save",
                str(recurrent_ckpt),
            ],
            PROJECT_ROOT,
        )
        run_cmd(
            [
                args.python,
                "scripts/eval_baselines.py",
                "--config",
                str(seed_cfg_path),
                "--structured-ckpt",
                str(structured_ckpt),
                "--random-probe-ckpt",
                str(random_probe_ckpt),
                "--no-rules-ckpt",
                str(no_rules_ckpt),
                "--recurrent-ckpt",
                str(recurrent_ckpt),
                "--num-batches",
                str(args.eval_batches),
                "--out",
                str(eval_out),
            ],
            PROJECT_ROOT,
        )
        if has_ood:
            run_cmd(
                [
                    args.python,
                    "scripts/eval_ood_baselines.py",
                    "--config",
                    str(seed_cfg_path),
                    "--structured-ckpt",
                    str(structured_ckpt),
                    "--random-probe-ckpt",
                    str(random_probe_ckpt),
                    "--no-rules-ckpt",
                    str(no_rules_ckpt),
                    "--recurrent-ckpt",
                    str(recurrent_ckpt),
                    "--num-batches",
                    str(args.eval_batches),
                    "--out",
                    str(eval_ood_out),
                ],
                PROJECT_ROOT,
            )

        per_seed[str(seed)] = json.loads(eval_out.read_text())
        if has_ood:
            per_seed_ood[str(seed)] = json.loads(eval_ood_out.read_text())

    summary = {
        "tag": args.tag,
        "config": args.config,
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate_overall": aggregate_seed_results(per_seed),
        "structured_vs_baselines": summarize_structured_vs_baselines(per_seed, None),
    }
    if has_ood:
        summary["per_seed_ood"] = per_seed_ood
        summary["aggregate_iid"] = aggregate_ood_seed_results(per_seed_ood, split="iid")
        summary["aggregate_ood"] = aggregate_ood_seed_results(per_seed_ood, split="ood")
        summary["structured_vs_baselines"] = summarize_structured_vs_baselines(per_seed, per_seed_ood)

    summary_path = root_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
