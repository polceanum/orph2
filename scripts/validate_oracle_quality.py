import argparse
import json


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_mean(blob: dict, key: str) -> float | None:
    v = blob.get("aggregate", {}).get(key)
    return None if v is None else float(v["mean"])


def check_oracle(primary: dict, oracle: dict, min_margin: float) -> tuple[bool, list[str]]:
    checks = []
    ok = True
    for p_key, label in (
        ("structured_ood", "structured"),
        ("recurrent_ood", "recurrent"),
        ("standard_ac_ood", "standard_actor_critic"),
    ):
        p = get_mean(primary, p_key)
        o = get_mean(oracle, p_key)
        if p is None or o is None:
            continue
        margin = o - p
        passed = margin >= min_margin
        checks.append(f"{label}: primary={p:.4f}, oracle={o:.4f}, margin={margin:.4f}, pass={passed}")
        ok = ok and passed
    return ok, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--min-margin", type=float, default=0.005)
    args = ap.parse_args()

    primary = load_json(args.primary)
    oracle = load_json(args.oracle)
    ok, checks = check_oracle(primary, oracle, args.min_margin)
    print("Oracle quality checks:")
    for line in checks:
        print(f"- {line}")
    if not checks:
        print("- no overlapping keys to compare; treating as failure")
        raise SystemExit(2)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
