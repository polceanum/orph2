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
