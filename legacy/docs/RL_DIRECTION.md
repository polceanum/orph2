# RL Direction Cleanup Map

This document records the current repository direction after shifting to an RL-centered OOD adaptation program.

## Primary path (active)

- `scripts/train_rl_compare.py`
- `configs/bridges/bridge_10_mechanism_addition_rl.yaml`
- `configs/bridges/bridge_10_mechanism_addition_rl_quick.yaml`

Use this path for claims about:
- structured vs classical RL baseline
- IID vs OOD adaptation under probe-action interaction

## Core evaluation semantics

- `IID`: same distribution as training.
- `OOD`: explicit shifted distribution in `eval.ood`.
- `oracle`: same model class trained on target OOD split.
- `adaptation efficiency`: structured OOD relative to oracle OOD.

## Legacy path

Legacy supervised/ablation scripts were removed in this cleanup.
The active evaluation surface is RL-first only.

## Live monitoring standard

For long runs, always pass `--log-jsonl` and monitor with `tail -f`.

Example:

```bash
python scripts/train_rl_compare.py \
  --config configs/bridges/bridge_10_mechanism_addition_rl.yaml \
  --seeds 0,1,2 \
  --out artifacts/rl/bridge10_rl_full_s012.json \
  --log-jsonl artifacts/rl/bridge10_rl_full_s012.log.jsonl

tail -f artifacts/rl/bridge10_rl_full_s012_seed0.log.jsonl
```

## Next cleanup steps (recommended)

1. Add one external benchmark adapter (Procgen or MiniGrid/BabyAI first).
2. Create a unified report script that prints IID/OOD/oracle/adaptation-efficiency tables.
3. Mark deprecated experiment configs explicitly once external benchmark path is stable.
