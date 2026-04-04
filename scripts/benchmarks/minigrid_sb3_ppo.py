#!/usr/bin/env python3
"""Train/evaluate a standard SB3 PPO baseline on Minigrid IID/OOD splits."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover
    gym = None


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _build_env(env_id: str, seed: int, obs_mode: str):
    import gymnasium as gym  # type: ignore
    import minigrid  # noqa: F401  # type: ignore
    from gymnasium.wrappers import FlattenObservation
    from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

    env = gym.make(env_id)
    mode = str(obs_mode).lower()
    if mode == "full":
        env = FullyObsWrapper(env)
    elif mode != "partial":
        raise ValueError(f"Unsupported obs_mode={obs_mode!r}; expected 'partial' or 'full'.")
    env = ImgObsWrapper(env)  # image-only observation
    env = FlattenObservation(env)
    env.reset(seed=seed)
    return env


def _infer_obs_dim(env_id: str, seed: int, obs_mode: str) -> int:
    env = _build_env(env_id=env_id, seed=seed, obs_mode=obs_mode)
    try:
        shape = tuple(env.observation_space.shape)
        if len(shape) != 1:
            raise ValueError(f"Expected 1D flattened observation, got shape={shape}")
        return int(shape[0])
    finally:
        env.close()


def _pad_obs(obs: np.ndarray, target_dim: int) -> np.ndarray:
    cur = int(obs.shape[0])
    if cur == target_dim:
        return obs
    if cur > target_dim:
        return obs[:target_dim]
    out = np.zeros((target_dim,), dtype=obs.dtype)
    out[:cur] = obs
    return out


class MinigridIIDSamplerEnv(gym.Env if gym is not None else object):
    """A single-env sampler that switches among IID env IDs each episode."""

    metadata = {"render_modes": []}

    def __init__(self, env_ids: list[str], seed: int, obs_mode: str, obs_dim: int | None = None):
        import gymnasium as gym  # type: ignore

        if not env_ids:
            raise ValueError("env_ids cannot be empty")
        self._env_ids = list(env_ids)
        self._seed = int(seed)
        self._obs_mode = str(obs_mode)
        self._obs_dim = int(obs_dim) if obs_dim is not None else None
        self._episode_idx = 0
        self._rng = random.Random(seed)
        self._env = _build_env(self._rng.choice(self._env_ids), seed=seed, obs_mode=self._obs_mode)
        self.action_space = self._env.action_space
        if self._obs_dim is None:
            self.observation_space = self._env.observation_space
        else:
            import gymnasium as gym  # type: ignore

            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._obs_dim,),
                dtype=np.float32,
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = int(seed)
            self._rng = random.Random(self._seed)
        self._episode_idx += 1
        env_id = self._rng.choice(self._env_ids)
        if self._env is not None:
            self._env.close()
        self._env = _build_env(env_id, seed=self._seed + self._episode_idx, obs_mode=self._obs_mode)
        obs, info = self._env.reset(seed=self._seed + self._episode_idx, options=options)
        if self._obs_dim is not None:
            obs = _pad_obs(np.asarray(obs), self._obs_dim)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._obs_dim is not None:
            obs = _pad_obs(np.asarray(obs), self._obs_dim)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()


@dataclass
class EvalResult:
    episodes: int
    mean_return: float
    success_rate: float
    mean_len: float


def evaluate_policy_on_env_id(
    model,
    env_id: str,
    episodes: int,
    max_steps: int,
    seed: int,
    obs_mode: str,
    obs_dim: int | None = None,
) -> EvalResult:
    env = _build_env(env_id, seed=seed, obs_mode=obs_mode)
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if obs_dim is not None:
            obs = _pad_obs(np.asarray(obs), obs_dim)
        done = False
        ep_ret = 0.0
        t = 0
        while not done and t < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if obs_dim is not None:
                obs = _pad_obs(np.asarray(obs), obs_dim)
            ep_ret += float(reward)
            t += 1
            done = bool(terminated or truncated)
            if terminated and float(reward) > 0.0:
                successes += 1
        returns.append(ep_ret)
        lengths.append(t)
    env.close()
    return EvalResult(
        episodes=episodes,
        mean_return=float(np.mean(returns)) if returns else 0.0,
        success_rate=float(successes / max(episodes, 1)),
        mean_len=float(np.mean(lengths)) if lengths else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/benchmarks/minigrid_door_key_ppo.yaml")
    parser.add_argument("--out", type=str, default="artifacts/benchmarks/minigrid_ppo_s0.json")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "stable-baselines3 is required for PPO benchmark runs. "
            "Install with: `conda run -n orpheus python -m pip install stable-baselines3`"
        ) from exc

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(args.seed) if args.seed is not None else int(_cfg_get(cfg, "seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    iid_env_ids = list(_cfg_get(cfg, "benchmark.minigrid.iid_env_ids", []))
    ood_env_ids = list(_cfg_get(cfg, "benchmark.minigrid.ood_env_ids", []))
    if not iid_env_ids or not ood_env_ids:
        raise ValueError("Config must define non-empty iid_env_ids and ood_env_ids.")

    train_steps = int(_cfg_get(cfg, "benchmark.train.total_timesteps", 200_000))
    eval_episodes = int(_cfg_get(cfg, "benchmark.eval.episodes", 100))
    max_steps = int(_cfg_get(cfg, "benchmark.eval.max_steps", 200))
    obs_mode = str(_cfg_get(cfg, "benchmark.observation_mode", "partial"))
    fixed_obs_dim = bool(_cfg_get(cfg, "benchmark.force_fixed_obs_dim", True))
    obs_dim = None
    if fixed_obs_dim:
        all_env_ids = list(dict.fromkeys(iid_env_ids + ood_env_ids))
        obs_dim = max(_infer_obs_dim(env_id=eid, seed=seed, obs_mode=obs_mode) for eid in all_env_ids)

    train_env = MinigridIIDSamplerEnv(iid_env_ids, seed=seed, obs_mode=obs_mode, obs_dim=obs_dim)
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=seed,
        learning_rate=float(_cfg_get(cfg, "benchmark.train.learning_rate", 3e-4)),
        n_steps=int(_cfg_get(cfg, "benchmark.train.n_steps", 256)),
        batch_size=int(_cfg_get(cfg, "benchmark.train.batch_size", 256)),
        n_epochs=int(_cfg_get(cfg, "benchmark.train.n_epochs", 4)),
        gamma=float(_cfg_get(cfg, "benchmark.train.gamma", 0.99)),
        gae_lambda=float(_cfg_get(cfg, "benchmark.train.gae_lambda", 0.95)),
        clip_range=float(_cfg_get(cfg, "benchmark.train.clip_range", 0.2)),
        ent_coef=float(_cfg_get(cfg, "benchmark.train.ent_coef", 0.01)),
        vf_coef=float(_cfg_get(cfg, "benchmark.train.vf_coef", 0.5)),
        verbose=int(_cfg_get(cfg, "benchmark.train.verbose", 1)),
        device=str(_cfg_get(cfg, "benchmark.train.device", "cpu")),
    )
    model.learn(total_timesteps=train_steps, progress_bar=False)
    train_env.close()

    per_env: dict[str, dict[str, Any]] = {}
    iid_srs: list[float] = []
    ood_srs: list[float] = []

    for env_id in iid_env_ids:
        r = evaluate_policy_on_env_id(
            model,
            env_id,
            episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed + 1_000,
            obs_mode=obs_mode,
            obs_dim=obs_dim,
        )
        per_env[env_id] = {"split": "iid", **r.__dict__}
        iid_srs.append(r.success_rate)
    for env_id in ood_env_ids:
        r = evaluate_policy_on_env_id(
            model,
            env_id,
            episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed + 2_000,
            obs_mode=obs_mode,
            obs_dim=obs_dim,
        )
        per_env[env_id] = {"split": "ood", **r.__dict__}
        ood_srs.append(r.success_rate)

    out = {
        "config": args.config,
        "seed": seed,
        "algorithm": "stable_baselines3.PPO",
        "observation_mode": obs_mode,
        "fixed_obs_dim": obs_dim,
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
    print(f"Wrote Minigrid PPO report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
