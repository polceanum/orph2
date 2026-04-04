# OOD Solver (RL-First)

This repository is now a minimal RL-first codebase for testing structured OOD adaptation.

## Goal

Train and compare:

- `structured` RL controller (approaches + rules + probe policy + belief updates)
- `recurrent_rl` baseline (single-latent recurrent policy)

under explicit IID vs OOD splits.

## Definitions

- `IID`: same distribution as training.
- `OOD`: explicit shift defined in `eval.ood` (for bridge-10 this adds an unseen mechanism).
- `oracle`: same model class trained directly on target OOD split (reference upper bound).

## Run

Quick smoke:

```bash
conda run -n orpheus python scripts/train_rl_compare.py \
  --config configs/bridges/bridge_10_mechanism_addition_rl_quick.yaml \
  --seeds 0 \
  --out artifacts/rl/bridge10_rl_quick_s0.json \
  --log-jsonl artifacts/rl/bridge10_rl_quick_s0.log.jsonl
```

Main run:

```bash
conda run -n orpheus python scripts/train_rl_compare.py \
  --config configs/bridges/bridge_10_mechanism_addition_rl.yaml \
  --seeds 0,1,2 \
  --out artifacts/rl/bridge10_rl_full_s012.json \
  --log-jsonl artifacts/rl/bridge10_rl_full_s012.log.jsonl
```

Live monitor:

```bash
tail -f artifacts/rl/bridge10_rl_full_s012_seed0.log.jsonl
```

## Current active files

- `scripts/train_rl_compare.py`
- `configs/bridges/bridge_10_mechanism_addition_rl.yaml`
- `configs/bridges/bridge_10_mechanism_addition_rl_quick.yaml`
- `docs/RL_DIRECTION.md`

## Environment

Use the `orpheus` conda environment (custom macOS PyTorch build).

## Benchmark On-Ramp (Minigrid, Mac-Friendly)

Install optional benchmark deps:

```bash
conda run -n orpheus python -m pip install -r requirements-benchmarks.txt
```

Run Minigrid IID/OOD smoke evaluation (random policy):

```bash
conda run -n orpheus python scripts/benchmarks/minigrid_smoke_eval.py \
  --config configs/benchmarks/minigrid_door_key_smoke.yaml \
  --out artifacts/benchmarks/minigrid_smoke_random_s0.json
```

Benchmark roadmap and scientific protocol:
- `docs/BENCHMARK_PLAN.md`

Run standardized PPO baseline (first non-trivial Minigrid OOD tier):

```bash
conda run -n orpheus python scripts/benchmarks/minigrid_sb3_ppo.py \
  --config configs/benchmarks/minigrid_empty_random_ppo_quick.yaml \
  --seed 0 \
  --out artifacts/benchmarks/minigrid_empty_random_ppo_quick_s0.json
```
