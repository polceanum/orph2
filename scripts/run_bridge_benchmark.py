import argparse
import json
import subprocess
import sys
from pathlib import Path as _Path
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_benchmark(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_csv(text: str | None) -> list[str]:
    if text is None:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_seed_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("No seeds provided.")
    return out


def cfg_set(cfg: dict, key: str, value):
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def parse_override(text: str) -> tuple[str, object]:
    if "=" not in text:
        raise ValueError(f"Invalid override {text!r}. Expected KEY=VALUE.")
    key, raw_val = text.split("=", 1)
    key = key.strip()
    raw_val = raw_val.strip()
    if not key:
        raise ValueError(f"Invalid override {text!r}. Empty key.")
    value = yaml.safe_load(raw_val)
    return key, value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    out = json.loads(json.dumps(cfg))
    for ov in overrides:
        key, value = parse_override(ov)
        cfg_set(out, key, value)
    return out


def run_cmd(cmd: list[str]) -> None:
    # Force unbuffered Python output so long runs stream progress immediately.
    if cmd and _Path(cmd[0]).name.startswith("python") and "-u" not in cmd[1:3]:
        cmd = [cmd[0], "-u", *cmd[1:]]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _get_nested(d: dict, path: str):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def build_row(name: str, summary: dict, oracle_summary: dict | None) -> dict:
    iid_struct = _get_nested(summary, "aggregate_iid.structured.seq_acc.mean")
    ood_struct = _get_nested(summary, "aggregate_ood.structured.seq_acc.mean")
    gap_struct = None if iid_struct is None or ood_struct is None else (ood_struct - iid_struct)

    row = {
        "scenario": name,
        "structured_iid_seq_acc": iid_struct,
        "structured_ood_seq_acc": ood_struct,
        "structured_ood_gap": gap_struct,
        "structured_chance_floor": 1.0 / 16.0,
        "structured_vs_recurrent_ood_delta_ci95": _get_nested(
            summary, "structured_vs_baselines.recurrent.ood_seq_acc_delta.batch_bootstrap_95ci"
        ),
        "structured_vs_no_rules_ood_delta_ci95": _get_nested(
            summary, "structured_vs_baselines.no_rules.ood_seq_acc_delta.batch_bootstrap_95ci"
        ),
        "structured_vs_random_probe_ood_delta_ci95": _get_nested(
            summary, "structured_vs_baselines.random_probe.ood_seq_acc_delta.batch_bootstrap_95ci"
        ),
    }

    if oracle_summary is not None:
        oracle_struct = _get_nested(oracle_summary, "aggregate_overall.structured.seq_acc.mean")
        row["oracle_structured_seq_acc"] = oracle_struct
        row["structured_adaptation_efficiency"] = (
            None if oracle_struct in (None, 0.0) or ood_struct is None else (ood_struct / oracle_struct)
        )
        row["structured_ood_oracle_gap"] = None if oracle_struct is None or ood_struct is None else (ood_struct - oracle_struct)
    return row


def evaluate_oracle_on_target_ood(
    python_exec: str,
    scenario_config: str,
    oracle_structured_ckpt: Path,
    out_path: Path,
    num_batches: int,
) -> dict:
    run_cmd(
        [
            python_exec,
            "scripts/eval_ood_baselines.py",
            "--config",
            scenario_config,
            "--structured-ckpt",
            str(oracle_structured_ckpt),
            "--num-batches",
            str(num_batches),
            "--out",
            str(out_path),
        ]
    )
    return json.loads(out_path.read_text())


def materialize_config_with_overrides(
    source_path: str,
    scenario_name: str,
    tag_prefix: str,
    overrides: list[str],
    suffix: str = "",
) -> str:
    if not overrides:
        return source_path
    with open(source_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, overrides)
    out_dir = PROJECT_ROOT / "artifacts" / "bridge_benchmark" / "generated_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag_prefix}__{scenario_name}{suffix}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", type=str, default="configs/bridges/benchmark.yaml")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--tag-prefix", type=str, default="bridge_benchmark")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--with-oracle", action="store_true")
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="Override applied to scenario configs before running (repeatable): train.foo=bar",
    )
    parser.add_argument(
        "--oracle-train-override",
        action="append",
        default=[],
        help="Override applied to oracle configs before running (repeatable): train.foo=bar",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Optional comma-separated scenario names to run from benchmark config.",
    )
    args = parser.parse_args()
    seed_list = parse_seed_list(args.seeds)

    bench = load_benchmark(args.benchmark_config)
    scenarios = bench.get("scenarios", [])
    wanted = set(parse_csv(args.scenarios))
    if wanted:
        scenarios = [s for s in scenarios if s.get("name") in wanted]
    if not scenarios:
        raise ValueError("No scenarios selected.")

    report_rows = []
    report = {
        "benchmark_config": args.benchmark_config,
        "seeds": args.seeds,
        "eval_batches": args.eval_batches,
        "with_oracle": args.with_oracle,
        "train_override": args.train_override,
        "oracle_train_override": args.oracle_train_override,
        "scenarios": {},
    }

    for scenario in scenarios:
        name = scenario["name"]
        config = materialize_config_with_overrides(
            source_path=scenario["config"],
            scenario_name=name,
            tag_prefix=args.tag_prefix,
            overrides=args.train_override,
            suffix="",
        )
        tag = f"{args.tag_prefix}__{name}"
        run_cmd(
            [
                args.python,
                "scripts/run_ablation_suite.py",
                "--config",
                config,
                "--seeds",
                args.seeds,
                "--tag",
                tag,
                "--eval-batches",
                str(args.eval_batches),
            ]
        )
        summary_path = PROJECT_ROOT / "artifacts" / "sweeps" / tag / "summary.json"
        summary = json.loads(summary_path.read_text())

        oracle_summary = None
        oracle_on_target_ood = None
        if args.with_oracle and scenario.get("oracle_config"):
            oracle_tag = f"{args.tag_prefix}__{name}__oracle"
            oracle_config = materialize_config_with_overrides(
                source_path=scenario["oracle_config"],
                scenario_name=name,
                tag_prefix=args.tag_prefix,
                overrides=args.oracle_train_override or args.train_override,
                suffix="__oracle",
            )
            run_cmd(
                [
                    args.python,
                    "scripts/run_ablation_suite.py",
                    "--config",
                    oracle_config,
                    "--seeds",
                    args.seeds,
                    "--tag",
                    oracle_tag,
                    "--eval-batches",
                    str(args.eval_batches),
                ]
            )
            oracle_path = PROJECT_ROOT / "artifacts" / "sweeps" / oracle_tag / "summary.json"
            oracle_summary = json.loads(oracle_path.read_text())
            oracle_structured_ckpt = (
                PROJECT_ROOT / "artifacts" / "sweeps" / oracle_tag / f"seed_{seed_list[0]}" / "structured.pt"
            )
            oracle_eval_out = (
                PROJECT_ROOT / "artifacts" / "bridge_benchmark" / f"{args.tag_prefix}__{name}__oracle_on_target_ood.json"
            )
            oracle_eval_out.parent.mkdir(parents=True, exist_ok=True)
            oracle_on_target_ood = evaluate_oracle_on_target_ood(
                python_exec=args.python,
                scenario_config=config,
                oracle_structured_ckpt=oracle_structured_ckpt,
                out_path=oracle_eval_out,
                num_batches=args.eval_batches,
            )

        row = build_row(name, summary, oracle_summary)
        if oracle_on_target_ood is not None:
            oracle_target_ood_acc = _get_nested(oracle_on_target_ood, "ood.structured.overall.seq_acc")
            row["oracle_structured_target_ood_seq_acc"] = oracle_target_ood_acc
            if row.get("structured_ood_seq_acc") is not None and oracle_target_ood_acc not in (None, 0.0):
                row["structured_adaptation_efficiency"] = row["structured_ood_seq_acc"] / oracle_target_ood_acc
                row["structured_ood_oracle_gap"] = row["structured_ood_seq_acc"] - oracle_target_ood_acc
        report_rows.append(row)
        report["scenarios"][name] = {
            "description": scenario.get("description"),
            "summary_path": str(summary_path),
            "oracle_summary_path": None
            if oracle_summary is None
            else str(PROJECT_ROOT / "artifacts" / "sweeps" / f"{args.tag_prefix}__{name}__oracle" / "summary.json"),
            "headline": row,
        }

    out_dir = PROJECT_ROOT / "artifacts" / "bridge_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag_prefix}_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print("\nBridge benchmark headline:")
    for row in report_rows:
        eff = row.get("structured_adaptation_efficiency")
        eff_txt = "NA" if eff is None else f"{eff:.3f}"
        print(
            f"{row['scenario']}: "
            f"iid={row['structured_iid_seq_acc']:.4f} "
            f"ood={row['structured_ood_seq_acc']:.4f} "
            f"gap={row['structured_ood_gap']:.4f} "
            f"eff={eff_txt}"
        )
    print(f"\nWrote benchmark report to {out_path}")


if __name__ == "__main__":
    main()
