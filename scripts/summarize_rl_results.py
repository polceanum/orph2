import argparse
import json
from pathlib import Path

import yaml


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_mean(blob: dict, key: str):
    if key not in blob.get("aggregate", {}):
        return None
    return blob["aggregate"][key]["mean"]


def fmt(x):
    if x is None:
        return "n/a"
    return f"{x:.4f}"


def build_summary(primary: dict, oracle: dict | None, random_chance: float) -> dict:
    out = {
        "config": primary.get("config"),
        "seeds": primary.get("seeds"),
        "random_chance_token_acc": random_chance,
        "metrics": {},
    }

    struct_ood = maybe_mean(primary, "structured_ood")
    recur_ood = maybe_mean(primary, "recurrent_ood")
    stdac_ood = maybe_mean(primary, "standard_ac_ood")
    delta_sr = primary.get("delta_structured_minus_recurrent", {}).get("ood", {}).get("mean")
    delta_ss = primary.get("delta_structured_minus_standard_actor_critic", {}).get("ood", {}).get("mean")

    out["metrics"]["structured_ood"] = struct_ood
    out["metrics"]["recurrent_ood"] = recur_ood
    out["metrics"]["standard_ac_ood"] = stdac_ood
    out["metrics"]["delta_structured_minus_recurrent_ood"] = delta_sr
    out["metrics"]["delta_structured_minus_standard_ac_ood"] = delta_ss

    if struct_ood is not None:
        out["metrics"]["structured_minus_random"] = struct_ood - random_chance
        out["metrics"]["structured_over_random_ratio"] = struct_ood / max(random_chance, 1e-12)
    if recur_ood is not None:
        out["metrics"]["recurrent_minus_random"] = recur_ood - random_chance
    if stdac_ood is not None:
        out["metrics"]["standard_ac_minus_random"] = stdac_ood - random_chance

    if oracle is not None:
        oracle_struct_ood = maybe_mean(oracle, "structured_ood")
        oracle_recur_ood = maybe_mean(oracle, "recurrent_ood")
        oracle_stdac_ood = maybe_mean(oracle, "standard_ac_ood")
        out["oracle"] = {
            "structured_ood": oracle_struct_ood,
            "recurrent_ood": oracle_recur_ood,
            "standard_ac_ood": oracle_stdac_ood,
        }
        if struct_ood is not None and oracle_struct_ood is not None:
            out["metrics"]["structured_transfer_ratio_vs_oracle"] = struct_ood / max(oracle_struct_ood, 1e-12)
            out["metrics"]["structured_oracle_gap"] = oracle_struct_ood - struct_ood
        if recur_ood is not None and oracle_recur_ood is not None:
            out["metrics"]["recurrent_transfer_ratio_vs_oracle"] = recur_ood / max(oracle_recur_ood, 1e-12)
        if stdac_ood is not None and oracle_stdac_ood is not None:
            out["metrics"]["standard_ac_transfer_ratio_vs_oracle"] = stdac_ood / max(oracle_stdac_ood, 1e-12)

    return out


def write_markdown(summary: dict, out_md: Path) -> None:
    m = summary["metrics"]
    lines = [
        "# RL Summary",
        "",
        f"- config: `{summary.get('config')}`",
        f"- seeds: `{summary.get('seeds')}`",
        f"- random chance token acc: `{fmt(summary.get('random_chance_token_acc'))}`",
        "",
        "## OOD",
        "",
        f"- structured: `{fmt(m.get('structured_ood'))}`",
        f"- recurrent: `{fmt(m.get('recurrent_ood'))}`",
        f"- standard_actor_critic: `{fmt(m.get('standard_ac_ood'))}`",
        f"- structured - recurrent: `{fmt(m.get('delta_structured_minus_recurrent_ood'))}`",
        f"- structured - standard_actor_critic: `{fmt(m.get('delta_structured_minus_standard_ac_ood'))}`",
        "",
        "## Chance/Oracle",
        "",
        f"- structured - random: `{fmt(m.get('structured_minus_random'))}`",
        f"- structured / random: `{fmt(m.get('structured_over_random_ratio'))}`",
        f"- structured oracle gap: `{fmt(m.get('structured_oracle_gap'))}`",
        f"- structured transfer ratio vs oracle: `{fmt(m.get('structured_transfer_ratio_vs_oracle'))}`",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="Primary result json (e.g., structured-vs-baselines).")
    ap.add_argument("--oracle", default=None, help="Optional oracle result json (trained on target OOD).")
    ap.add_argument("--config", default=None, help="Optional config path for chance baseline; defaults to primary['config'].")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    primary = load_json(args.primary)
    oracle = load_json(args.oracle) if args.oracle else None
    cfg_path = args.config or primary.get("config")
    if cfg_path is None:
        raise ValueError("No config path available for chance baseline.")
    cfg = load_yaml(cfg_path)
    vocab_size = int(cfg.get("env", {}).get("vocab_size", 1))
    random_chance = 1.0 / max(vocab_size, 1)

    summary = build_summary(primary, oracle, random_chance)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.out_md:
        write_markdown(summary, Path(args.out_md))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
