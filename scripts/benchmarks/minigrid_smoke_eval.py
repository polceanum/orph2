#!/usr/bin/env python3
"""Lightweight Minigrid IID/OOD smoke evaluation.

This script is intentionally simple: it evaluates a random policy on
config-defined train (IID) and test (OOD) env splits so we can verify
benchmark wiring and reporting before plugging in full training.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class SplitResult:
    episodes: int
    mean_return: float
    success_rate: float
    mean_len: float


def evaluate_random(
    env_id: str,
    episodes: int,
    max_steps: int,
    seed: int,
) -> SplitResult:
    try:
        import gymnasium as gym  # type: ignore
        import minigrid  # noqa: F401  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "gymnasium is required for Minigrid benchmark runs. "
            "Install with: `conda run -n orpheus python -m pip install gymnasium minigrid`"
        ) from exc

    returns: list[float] = []
    successes = 0
    lengths: list[int] = []
    for ep in range(episodes):
        env = gym.make(env_id)
        obs, info = env.reset(seed=seed + ep)
        ep_ret = 0.0
        done = False
        t = 0
        while not done and t < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            t += 1
            done = bool(terminated or truncated)
            if terminated and float(reward) > 0.0:
                successes += 1
        returns.append(ep_ret)
        lengths.append(t)
        env.close()

    return SplitResult(
        episodes=episodes,
        mean_return=float(np.mean(returns)) if returns else 0.0,
        success_rate=float(successes / max(episodes, 1)),
        mean_len=float(np.mean(lengths)) if lengths else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmarks/minigrid_door_key_smoke.yaml",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/benchmarks/minigrid_smoke_random.json",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(_cfg_get(cfg, "seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    iid_env_ids = list(_cfg_get(cfg, "benchmark.minigrid.iid_env_ids", []))
    ood_env_ids = list(_cfg_get(cfg, "benchmark.minigrid.ood_env_ids", []))
    episodes = int(_cfg_get(cfg, "benchmark.minigrid.eval_episodes", 100))
    max_steps = int(_cfg_get(cfg, "benchmark.minigrid.max_steps", 200))

    if not iid_env_ids or not ood_env_ids:
        raise ValueError("Config must define non-empty iid_env_ids and ood_env_ids.")

    per_env: dict[str, dict[str, Any]] = {}
    iid_srs: list[float] = []
    ood_srs: list[float] = []
    for env_id in iid_env_ids:
        res = evaluate_random(env_id=env_id, episodes=episodes, max_steps=max_steps, seed=seed)
        per_env[env_id] = {"split": "iid", **res.__dict__}
        iid_srs.append(res.success_rate)
    for env_id in ood_env_ids:
        res = evaluate_random(env_id=env_id, episodes=episodes, max_steps=max_steps, seed=seed + 10_000)
        per_env[env_id] = {"split": "ood", **res.__dict__}
        ood_srs.append(res.success_rate)

    out = {
        "config": args.config,
        "seed": seed,
        "policy": "random",
        "summary": {
            "iid_success_rate_mean": float(np.mean(iid_srs)),
            "ood_success_rate_mean": float(np.mean(ood_srs)),
            "ood_minus_iid_success_rate": float(np.mean(ood_srs) - np.mean(iid_srs)),
        },
        "per_env": per_env,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote Minigrid smoke report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
