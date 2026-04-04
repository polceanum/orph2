#!/usr/bin/env python3
"""Train/evaluate a standard SB3 PPO baseline on Minigrid IID/OOD splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import torch as th
import torch.nn as nn

try:
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover
    gym = None

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # type: ignore
except Exception:  # pragma: no cover
    BaseFeaturesExtractor = object


class SmallGridCNN(BaseFeaturesExtractor):
    """CNN extractor suitable for tiny Minigrid images (e.g. 7x7)."""

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = int(observation_space.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


class NavigationOnlyActionWrapper(gym.ActionWrapper if gym is not None else object):
    """Restrict MiniGrid actions to left/right/forward for pure navigation tasks."""

    ACTION_MAP = (0, 1, 2)

    def __init__(self, env):
        import gymnasium as gym  # type: ignore

        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.ACTION_MAP))

    def action(self, action):
        idx = int(action)
        if idx < 0 or idx >= len(self.ACTION_MAP):
            raise ValueError(f"Invalid reduced action index {idx}")
        return int(self.ACTION_MAP[idx])


class SimpleFrameStackObsWrapper(gym.ObservationWrapper if gym is not None else object):
    """Stack last K observations along the feature axis for single-env PPO."""

    def __init__(self, env, num_stack: int):
        import gymnasium as gym  # type: ignore

        super().__init__(env)
        self.num_stack = int(num_stack)
        if self.num_stack < 1:
            raise ValueError("num_stack must be >= 1")
        shape = tuple(self.observation_space.shape)
        if len(shape) != 1:
            raise ValueError(f"SimpleFrameStackObsWrapper expects 1D obs, got shape={shape}")
        self._base_dim = int(shape[0])
        self._frames: deque[np.ndarray] = deque(maxlen=self.num_stack)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._base_dim * self.num_stack,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_arr = np.asarray(obs)
        self._frames.clear()
        for _ in range(self.num_stack):
            self._frames.append(obs_arr.copy())
        return self.observation(obs_arr), info

    def observation(self, observation):
        obs_arr = np.asarray(observation)
        self._frames.append(obs_arr.copy())
        if len(self._frames) < self.num_stack:
            while len(self._frames) < self.num_stack:
                self._frames.append(obs_arr.copy())
        return np.concatenate(list(self._frames), axis=0)


def _cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _build_env(
    env_id: str,
    seed: int,
    obs_mode: str,
    flatten_obs: bool,
    navigation_only_actions: bool,
    frame_stack: int,
):
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
    if navigation_only_actions:
        env = NavigationOnlyActionWrapper(env)
    if flatten_obs:
        env = FlattenObservation(env)
    if flatten_obs and int(frame_stack) > 1:
        env = SimpleFrameStackObsWrapper(env, num_stack=int(frame_stack))
    env.reset(seed=seed)
    return env


def _infer_obs_dim(
    env_id: str,
    seed: int,
    obs_mode: str,
    flatten_obs: bool,
    navigation_only_actions: bool,
    frame_stack: int,
) -> int:
    env = _build_env(
        env_id=env_id,
        seed=seed,
        obs_mode=obs_mode,
        flatten_obs=flatten_obs,
        navigation_only_actions=navigation_only_actions,
        frame_stack=frame_stack,
    )
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


def _pad_obs_strict(obs: np.ndarray, target_dim: int, allow_truncate: bool) -> np.ndarray:
    cur = int(obs.shape[0])
    if cur > target_dim and not allow_truncate:
        raise ValueError(
            f"Observation dim {cur} exceeds train-time dim {target_dim}. "
            "Set benchmark.allow_ood_shape_truncation=true only if you explicitly accept this."
        )
    return _pad_obs(obs, target_dim)


class MinigridIIDSamplerEnv(gym.Env if gym is not None else object):
    """A single-env sampler that switches among IID env IDs each episode."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_ids: list[str],
        seed: int,
        obs_mode: str,
        flatten_obs: bool,
        navigation_only_actions: bool,
        frame_stack: int,
        obs_dim: int | None = None,
        allow_truncate: bool = False,
    ):
        import gymnasium as gym  # type: ignore

        if not env_ids:
            raise ValueError("env_ids cannot be empty")
        self._env_ids = list(env_ids)
        self._seed = int(seed)
        self._obs_mode = str(obs_mode)
        self._flatten_obs = bool(flatten_obs)
        self._navigation_only_actions = bool(navigation_only_actions)
        self._frame_stack = int(frame_stack)
        self._obs_dim = int(obs_dim) if obs_dim is not None else None
        self._allow_truncate = bool(allow_truncate)
        self._episode_idx = 0
        self._rng = random.Random(seed)
        self._env = _build_env(
            self._rng.choice(self._env_ids),
            seed=seed,
            obs_mode=self._obs_mode,
            flatten_obs=self._flatten_obs,
            navigation_only_actions=self._navigation_only_actions,
            frame_stack=self._frame_stack,
        )
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
        self._env = _build_env(
            env_id,
            seed=self._seed + self._episode_idx,
            obs_mode=self._obs_mode,
            flatten_obs=self._flatten_obs,
            navigation_only_actions=self._navigation_only_actions,
            frame_stack=self._frame_stack,
        )
        obs, info = self._env.reset(seed=self._seed + self._episode_idx, options=options)
        if self._obs_dim is not None:
            obs = _pad_obs_strict(np.asarray(obs), self._obs_dim, allow_truncate=self._allow_truncate)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._obs_dim is not None:
            obs = _pad_obs_strict(np.asarray(obs), self._obs_dim, allow_truncate=self._allow_truncate)
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
    flatten_obs: bool,
    navigation_only_actions: bool,
    frame_stack: int,
    obs_dim: int | None = None,
    allow_truncate: bool = False,
    deterministic: bool = True,
) -> EvalResult:
    env = _build_env(
        env_id,
        seed=seed,
        obs_mode=obs_mode,
        flatten_obs=flatten_obs,
        navigation_only_actions=navigation_only_actions,
        frame_stack=frame_stack,
    )
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if obs_dim is not None:
            obs = _pad_obs_strict(np.asarray(obs), obs_dim, allow_truncate=allow_truncate)
        done = False
        ep_ret = 0.0
        t = 0
        while not done and t < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            if obs_dim is not None:
                obs = _pad_obs_strict(np.asarray(obs), obs_dim, allow_truncate=allow_truncate)
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
    policy_type = str(_cfg_get(cfg, "benchmark.train.policy", "MlpPolicy"))
    flatten_obs = bool(_cfg_get(cfg, "benchmark.flatten_obs", policy_type.lower() != "cnnpolicy"))
    fixed_obs_dim = bool(_cfg_get(cfg, "benchmark.force_fixed_obs_dim", True))
    allow_ood_dim_leakage = bool(_cfg_get(cfg, "benchmark.allow_ood_shape_leakage_for_padding", False))
    allow_ood_shape_trunc = bool(_cfg_get(cfg, "benchmark.allow_ood_shape_truncation", False))
    navigation_only_actions = bool(_cfg_get(cfg, "benchmark.minigrid.navigation_only_actions", False))
    frame_stack = int(_cfg_get(cfg, "benchmark.minigrid.frame_stack", 1))
    obs_dim = None
    if fixed_obs_dim and flatten_obs:
        ids_for_dim = list(iid_env_ids)
        if allow_ood_dim_leakage:
            ids_for_dim = list(dict.fromkeys(iid_env_ids + ood_env_ids))
        obs_dim = max(
            _infer_obs_dim(
                env_id=eid,
                seed=seed,
                obs_mode=obs_mode,
                flatten_obs=flatten_obs,
                navigation_only_actions=navigation_only_actions,
                frame_stack=frame_stack,
            )
            for eid in ids_for_dim
        )

    train_env = MinigridIIDSamplerEnv(
        iid_env_ids,
        seed=seed,
        obs_mode=obs_mode,
        flatten_obs=flatten_obs,
        navigation_only_actions=navigation_only_actions,
        frame_stack=frame_stack,
        obs_dim=obs_dim,
        allow_truncate=allow_ood_shape_trunc,
    )
    policy_kwargs_cfg = _cfg_get(cfg, "benchmark.train.policy_kwargs", {})
    if policy_kwargs_cfg is None:
        policy_kwargs_cfg = {}
    if not isinstance(policy_kwargs_cfg, dict):
        raise ValueError("benchmark.train.policy_kwargs must be a mapping/dict when provided.")
    policy_kwargs: dict[str, Any] | None
    if policy_type.lower() == "cnnpolicy":
        policy_kwargs = {
            **policy_kwargs_cfg,
            "features_extractor_class": SmallGridCNN,
            "features_extractor_kwargs": {
                "features_dim": int(_cfg_get(cfg, "benchmark.train.cnn_features_dim", 128)),
                **dict(policy_kwargs_cfg.get("features_extractor_kwargs", {})),
            },
        }
    else:
        policy_kwargs = dict(policy_kwargs_cfg) if policy_kwargs_cfg else None

    model = PPO(
        policy_type,
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
        policy_kwargs=policy_kwargs,
        verbose=int(_cfg_get(cfg, "benchmark.train.verbose", 1)),
        device=str(_cfg_get(cfg, "benchmark.train.device", "cpu")),
    )
    eval_every = int(_cfg_get(cfg, "benchmark.train.eval_every_timesteps", 0))
    train_eval_episodes = int(_cfg_get(cfg, "benchmark.train.eval_episodes", 20))
    patience_evals = int(_cfg_get(cfg, "benchmark.train.early_stop_patience_evals", 0))
    use_best_model = bool(_cfg_get(cfg, "benchmark.train.use_best_model", True))
    best_iid_success = float("-inf")
    best_step = 0
    evals_without_improve = 0
    best_model_path = Path("artifacts/benchmarks/_tmp_minigrid_best_model")
    best_model_saved = False

    if eval_every > 0:
        steps_done = 0
        while steps_done < train_steps:
            chunk = min(eval_every, train_steps - steps_done)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
            steps_done += chunk
            iid_eval_scores: list[float] = []
            for env_id in iid_env_ids:
                r = evaluate_policy_on_env_id(
                    model,
                    env_id,
                    episodes=train_eval_episodes,
                    max_steps=max_steps,
                    seed=seed + 30_000 + steps_done,
                    obs_mode=obs_mode,
                    flatten_obs=flatten_obs,
                    navigation_only_actions=navigation_only_actions,
                    frame_stack=frame_stack,
                    obs_dim=obs_dim,
                    allow_truncate=allow_ood_shape_trunc,
                )
                iid_eval_scores.append(r.success_rate)
            iid_eval_mean = float(np.mean(iid_eval_scores)) if iid_eval_scores else 0.0
            print(
                f"[chunk-eval] step={steps_done} iid_success={iid_eval_mean:.3f} "
                f"best={best_iid_success if best_iid_success > -1e8 else float('nan'):.3f}",
                flush=True,
            )
            if iid_eval_mean > best_iid_success + 1e-12:
                best_iid_success = iid_eval_mean
                best_step = steps_done
                evals_without_improve = 0
                model.save(str(best_model_path))
                best_model_saved = True
            else:
                evals_without_improve += 1
                if patience_evals > 0 and evals_without_improve >= patience_evals:
                    print(
                        f"[early-stop] no IID improvement for {patience_evals} evals; stopping at step={steps_done}",
                        flush=True,
                    )
                    break
        if use_best_model and best_model_saved:
            model = PPO.load(str(best_model_path))
    else:
        model.learn(total_timesteps=train_steps, progress_bar=False)
    train_env.close()

    per_env: dict[str, dict[str, Any]] = {}
    iid_srs: list[float] = []
    ood_srs: list[float] = []
    eval_deterministic = bool(_cfg_get(cfg, "benchmark.eval.deterministic", True))
    include_stochastic_eval = bool(_cfg_get(cfg, "benchmark.eval.include_stochastic", False))

    for env_id in iid_env_ids:
        r = evaluate_policy_on_env_id(
            model,
            env_id,
            episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed + 1_000,
            obs_mode=obs_mode,
            flatten_obs=flatten_obs,
            navigation_only_actions=navigation_only_actions,
            frame_stack=frame_stack,
            obs_dim=obs_dim,
            allow_truncate=allow_ood_shape_trunc,
            deterministic=eval_deterministic,
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
            flatten_obs=flatten_obs,
            navigation_only_actions=navigation_only_actions,
            frame_stack=frame_stack,
            obs_dim=obs_dim,
            allow_truncate=allow_ood_shape_trunc,
            deterministic=eval_deterministic,
        )
        per_env[env_id] = {"split": "ood", **r.__dict__}
        ood_srs.append(r.success_rate)

    stochastic_block: dict[str, Any] | None = None
    if include_stochastic_eval:
        per_env_stochastic: dict[str, dict[str, Any]] = {}
        iid_srs_stoch: list[float] = []
        ood_srs_stoch: list[float] = []
        for env_id in iid_env_ids:
            r = evaluate_policy_on_env_id(
                model,
                env_id,
                episodes=eval_episodes,
                max_steps=max_steps,
                seed=seed + 11_000,
                obs_mode=obs_mode,
                flatten_obs=flatten_obs,
                navigation_only_actions=navigation_only_actions,
                frame_stack=frame_stack,
                obs_dim=obs_dim,
                allow_truncate=allow_ood_shape_trunc,
                deterministic=False,
            )
            per_env_stochastic[env_id] = {"split": "iid", **r.__dict__}
            iid_srs_stoch.append(r.success_rate)
        for env_id in ood_env_ids:
            r = evaluate_policy_on_env_id(
                model,
                env_id,
                episodes=eval_episodes,
                max_steps=max_steps,
                seed=seed + 12_000,
                obs_mode=obs_mode,
                flatten_obs=flatten_obs,
                navigation_only_actions=navigation_only_actions,
                frame_stack=frame_stack,
                obs_dim=obs_dim,
                allow_truncate=allow_ood_shape_trunc,
                deterministic=False,
            )
            per_env_stochastic[env_id] = {"split": "ood", **r.__dict__}
            ood_srs_stoch.append(r.success_rate)
        stochastic_block = {
            "summary": {
                "iid_success_rate_mean": float(np.mean(iid_srs_stoch)),
                "ood_success_rate_mean": float(np.mean(ood_srs_stoch)),
                "ood_minus_iid_success_rate": float(np.mean(ood_srs_stoch) - np.mean(iid_srs_stoch)),
            },
            "per_env": per_env_stochastic,
        }

    out = {
        "config": args.config,
        "seed": seed,
        "algorithm": "stable_baselines3.PPO",
        "policy_type": policy_type,
        "policy_kwargs": _json_safe(policy_kwargs if policy_kwargs is not None else {}),
        "observation_mode": obs_mode,
        "navigation_only_actions": navigation_only_actions,
        "frame_stack": frame_stack,
        "flatten_obs": flatten_obs,
        "fixed_obs_dim": obs_dim,
        "allow_ood_shape_leakage_for_padding": allow_ood_dim_leakage,
        "allow_ood_shape_truncation": allow_ood_shape_trunc,
        "summary": {
            "iid_success_rate_mean": float(np.mean(iid_srs)),
            "ood_success_rate_mean": float(np.mean(ood_srs)),
            "ood_minus_iid_success_rate": float(np.mean(ood_srs) - np.mean(iid_srs)),
        },
        "eval_mode": {
            "deterministic": eval_deterministic,
            "include_stochastic": include_stochastic_eval,
        },
        "train_selection": {
            "eval_every_timesteps": eval_every,
            "train_eval_episodes": train_eval_episodes,
            "early_stop_patience_evals": patience_evals,
            "use_best_model": use_best_model,
            "best_iid_success_during_training": (
                None if best_iid_success == float("-inf") else float(best_iid_success)
            ),
            "best_step": int(best_step),
        },
        "per_env": per_env,
    }
    if stochastic_block is not None:
        out["stochastic_eval"] = stochastic_block
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Wrote Minigrid PPO report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
