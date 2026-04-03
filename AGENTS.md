# AGENTS.md

## Mission

Optimize OOD adaptation in the RL-first structured-vs-recurrent comparison.

## Canonical command

```bash
conda run -n orpheus python scripts/train_rl_compare.py \
  --config configs/bridges/bridge_10_mechanism_addition_rl.yaml \
  --seeds 0,1,2 \
  --out artifacts/rl/bridge10_rl_full_s012.json \
  --log-jsonl artifacts/rl/bridge10_rl_full_s012.log.jsonl
```

## Rules

- Always report IID and OOD metrics.
- Always report structured-minus-recurrent deltas.
- Use `--log-jsonl` for long runs and monitor live.
- Use 1 seed for iteration, 3 seeds for conclusions.
