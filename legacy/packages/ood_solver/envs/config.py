from __future__ import annotations

from ood_solver.envs.hidden_mechanism_seq import HiddenMechanismSequenceEnv


def cfg_get(cfg: dict, key: str, default=None):
    parts = key.split(".")
    cur = cfg
    ok = True
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            ok = False
            break
    if ok:
        return cur
    return cfg.get(parts[-1], default)


def build_env_from_cfg(
    cfg: dict,
    section: str = "env",
    seed: int | None = None,
    seed_offset: int = 0,
):
    prefix = section
    base_seed = int(cfg_get(cfg, "seed", 0)) if seed is None else int(seed)
    env_seed = base_seed + int(seed_offset)

    mechanisms = cfg_get(cfg, f"{prefix}.mechanisms", None)
    if mechanisms is not None:
        mechanisms = [int(m) for m in mechanisms]
    mechanism_sampling_weights = cfg_get(cfg, f"{prefix}.mechanism_sampling_weights", None)
    if mechanism_sampling_weights is not None:
        mechanism_sampling_weights = [float(w) for w in mechanism_sampling_weights]

    param_ranges = cfg_get(cfg, f"{prefix}.param_ranges", None)
    if param_ranges is not None:
        param_ranges = [(int(lo), int(hi)) for lo, hi in param_ranges]
    input_value_range = cfg_get(cfg, f"{prefix}.input_value_range", None)
    if input_value_range is not None:
        input_value_range = (int(input_value_range[0]), int(input_value_range[1]))
    probe_value_range = cfg_get(cfg, f"{prefix}.probe_value_range", None)
    if probe_value_range is not None:
        probe_value_range = (int(probe_value_range[0]), int(probe_value_range[1]))
    recursive_mode = str(cfg_get(cfg, f"{prefix}.recursive_mode", cfg_get(cfg, "env.recursive_mode", "sum_halves")))

    return HiddenMechanismSequenceEnv(
        vocab_size=int(cfg_get(cfg, f"{prefix}.vocab_size", cfg_get(cfg, "env.vocab_size", 16))),
        seq_len=int(cfg_get(cfg, f"{prefix}.seq_len", cfg_get(cfg, "env.seq_len", 12))),
        num_probe_steps=int(cfg_get(cfg, f"{prefix}.num_probe_steps", cfg_get(cfg, "env.num_probe_steps", 4))),
        num_candidate_probes=int(
            cfg_get(cfg, f"{prefix}.num_candidate_probes", cfg_get(cfg, "env.num_candidate_probes", 8))
        ),
        num_demos=int(cfg_get(cfg, f"{prefix}.num_demos", cfg_get(cfg, "env.num_demos", 3))),
        seed=env_seed,
        mechanisms=mechanisms,
        mechanism_sampling_weights=mechanism_sampling_weights,
        param_ranges=param_ranges,
        recursive_mode=recursive_mode,
        input_value_range=input_value_range,
        probe_value_range=probe_value_range,
        local_use_left_context=bool(
            cfg_get(
                cfg,
                f"{prefix}.local_use_left_context",
                cfg_get(cfg, "env.local_use_left_context", True),
            )
        ),
    )
