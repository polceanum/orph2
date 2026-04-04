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
- After each meaningful experiment batch, append `docs/EXPERIMENT_LOG.md` with:
  - Question, Hypothesis, Controls, Runs, Result, Interpretation, Decision, Next step.
- Log negative/null results explicitly (do not omit failed variants).
- Mark 1-seed outcomes as exploratory; do not present as conclusions.

## Required Baselines

- `recurrent_rl` baseline
- `standard_actor_critic` baseline (mainstream comparator)
- random chance reference
- oracle reference (trained on target OOD split)

## Oracle Quality Gate

- Generate oracle config from primary with `scripts/make_oracle_config.py`.
- Validate oracle quality with `scripts/validate_oracle_quality.py`.
- If oracle does not beat primary by minimum margin, mark oracle as underfit and do not use transfer-ratio claims.

## Reporting Command (Summary)

```bash
conda run -n orpheus python scripts/summarize_rl_results.py \
  --primary <result.json> \
  --oracle <oracle_result.json> \
  --out-json <summary.json> \
  --out-md <summary.md>
```

## Research Log Requirement

- `docs/EXPERIMENT_LOG.md` is the canonical research journal.
- Keep entries chronological and evidence-linked (artifact paths required).
- If oracle quality gate fails, mark oracle as underfit in the log and avoid transfer-ratio claims.
