#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _extract_generic_block(agent_text: str) -> str:
    start = agent_text.find("def _symbolic_solve_generic(question: str) -> str | None:")
    if start < 0:
        raise ValueError("Could not find _symbolic_solve_generic in agent file")
    end = agent_text.find("\ndef _symbolic_solve(question: str) -> str | None:", start)
    if end < 0:
        raise ValueError("Could not find _symbolic_solve boundary in agent file")
    return agent_text[start:end]


def _extract_rule_ids(generic_block: str) -> set[str]:
    return set(re.findall(r"RULE_ID:\s*([a-z0-9_]+)", generic_block))


def _load_registry(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values: set[str] = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.add(line)
    if not values:
        raise ValueError(f"Registry is empty: {path}")
    return values


def _extract_ood_only_names(benchmark_path: Path) -> set[str]:
    excluded = {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Calculate",
        "Compute",
        "Evaluate",
        "What",
        "How",
        "If",
        "The",
        "There",
        "Then",
        "And",
        "For",
        "From",
        "With",
        "Without",
        "Her",
        "His",
        "She",
        "He",
        "They",
        "Math",
        "Every",
        "Profit",
        "Red",
        "Blue",
        "Green",
        "Black",
        "White",
        "Bill",
        "Last",
        "Four",
    }
    iid_names: set[str] = set()
    ood_names: set[str] = set()
    for raw in benchmark_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(raw)
        q = item.get("question", "")
        split = item.get("split", "")
        # Capitalized words (>=3 chars) are a useful leakage signal for
        # benchmark-instance keyed templates.
        names = set(re.findall(r"\b[A-Z][a-z]{2,}\b", q))
        if split == "iid":
            iid_names |= names
        elif split == "ood":
            ood_names |= names
    return {n for n in ood_names if n not in iid_names and n not in excluded}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-path", required=True)
    ap.add_argument("--registry-path", required=True)
    ap.add_argument("--benchmark-path", required=True)
    args = ap.parse_args()

    agent_path = Path(args.agent_path)
    registry_path = Path(args.registry_path)
    benchmark_path = Path(args.benchmark_path)

    generic_block = _extract_generic_block(agent_path.read_text(encoding="utf-8"))

    found_ids = _extract_rule_ids(generic_block)
    allowed_ids = _load_registry(registry_path)

    if not found_ids:
        raise SystemExit(
            "IID-wall validation failed: no RULE_ID tags found in "
            "_symbolic_solve_generic. Add RULE_ID comments for each strict rule."
        )

    unknown = sorted(found_ids - allowed_ids)
    if unknown:
        raise SystemExit(
            "IID-wall validation failed: strict generic rules include unregistered "
            f"RULE_ID(s): {', '.join(unknown)}"
        )

    ood_only_names = _extract_ood_only_names(benchmark_path)
    generic_low = generic_block.lower()
    leaked_names = sorted(
        [
            n
            for n in ood_only_names
            if re.search(rf"\b{re.escape(n.lower())}\b", generic_low)
        ]
    )
    if leaked_names:
        raise SystemExit(
            "IID-wall validation failed: OOD-only names found in strict generic "
            f"solver block: {', '.join(leaked_names)}"
        )

    print("IID-wall validation passed")
    print(f"  Registered RULE_ID count: {len(found_ids)}")
    print(f"  Registry path: {registry_path}")
    print(f"  Benchmark checked: {benchmark_path}")


if __name__ == "__main__":
    main()
