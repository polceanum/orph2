# Benchmark Plan (Gradual Standardization)

## Goals

- Preserve current bridge-mechanism experiments for continuity.
- Add progressively more standard, literature-aligned benchmarks.
- Keep apple-to-apple reporting: same metrics, same seed policy, same baseline set.

## Required Comparators (every benchmark)

- `structured` (our method)
- `recurrent_rl` baseline
- `standard_actor_critic` baseline
- `random chance` reference
- `oracle` reference (same model class trained on target OOD distribution)

## Required Metrics

- IID token/sequence accuracy
- OOD token/sequence accuracy
- Delta:
  - structured - recurrent
  - structured - standard actor-critic
- Chance gap:
  - structured - random
- Oracle gap + transfer ratio:
  - oracle - structured
  - structured / oracle

## Seed Policy

- Iteration: 1 seed
- Candidate acceptance: 3 seeds
- Main claims: 5 seeds

## Benchmark Ladder

1. Current bridge mechanism benchmarks (keep)
2. Harder synthetic OOD variants:
   - unseen stochastic mechanism
   - wider param ranges
   - fewer demos / more probes
3. Standardized toy benchmarks from literature:
   - contextual bandit-style OOD shifts
   - small POMDP/meta-RL suites with train/test task splits
4. Mid-scale benchmark suite:
   - at least one public benchmark with widely used baselines and published metrics

## Fairness Rules

- Same training budget across compared methods unless explicitly reported.
- Same environment generation seeds for each compared method.
- Same optimizer class unless benchmark protocol requires otherwise.
- No method-specific tuning on test/OOD split.

## Reporting Artifacts

- Raw run JSON in `artifacts/rl/`
- Summary JSON + markdown via `scripts/summarize_rl_results.py`
- Planned: standard plotting script output for all accepted runs
