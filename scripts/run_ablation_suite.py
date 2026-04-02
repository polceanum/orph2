import argparse
import json
import math
import subprocess
import sys
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

        per_seed[str(seed)] = json.loads(eval_out.read_text())

    summary = {
        "tag": args.tag,
        "config": args.config,
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate_overall": aggregate_seed_results(per_seed),
    }

    summary_path = root_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
