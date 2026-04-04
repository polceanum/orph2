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
- Always report structured-minus-standard-actor-critic deltas when enabled.
- Use `--log-jsonl` for long runs and monitor live.
- Use 1 seed for iteration, 3 seeds for conclusions.
- Use 5 seeds for stronger claims.
- Include random-chance and oracle context in summary tables.

## Required Baselines

- `recurrent_rl` baseline
- `standard_actor_critic` baseline (mainstream comparator)
- random chance reference
- oracle reference (trained on target OOD split)

## Reporting Command (Summary)

```bash
conda run -n orpheus python scripts/summarize_rl_results.py \
  --primary <result.json> \
  --oracle <oracle_result.json> \
  --out-json <summary.json> \
  --out-md <summary.md>
```
