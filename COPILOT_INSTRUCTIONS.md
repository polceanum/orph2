# Copilot Instructions

This repository is intentionally narrowed to an RL-first OOD adaptation track.

## Preserve

- Explicit structured reasoning state (approaches/rules/belief updates).
- Comparable recurrent RL baseline.
- Explicit IID/OOD split evaluation.

## Prioritize

- OOD accuracy delta: `structured - recurrent_rl`.
- OOD accuracy delta: `structured - standard_actor_critic`.
- Stable multi-seed evidence.
- Fast failure detection via live JSONL logs.
- Chance and oracle context in result summaries.
- Oracle quality gate must pass before using oracle transfer claims.

## Scientific Protocol

- Always report IID + OOD for every compared method.
- Always include:
  - `structured`
  - `recurrent_rl`
  - `standard_actor_critic`
  - random-chance reference
  - oracle reference (when available)
- Use:
  - 1 seed for exploratory iteration
  - 3 seeds for candidate acceptance
  - 5 seeds for stronger claims
- Use `scripts/make_oracle_config.py` + `scripts/validate_oracle_quality.py` for oracle runs.
- Reject variants that degrade either:
  - absolute structured OOD
  - or structured-vs-baseline OOD delta
- After each experiment batch, update `docs/EXPERIMENT_LOG.md` with:
  - Question, Hypothesis, Controls, Runs, Result, Interpretation, Decision, Next step.
- Include failed/null findings in the log to avoid selection bias.
- Treat 1-seed findings as exploratory; require 3/5 seeds for acceptance/strong claims.

## Standardization Path

- Keep current bridge tasks for continuity.
- Add standardized benchmarks gradually; do not replace existing tests abruptly.
- Prefer controlled benchmark additions where all methods share:
  - same train budget
  - same seeds
  - same eval protocol

## Avoid

- Re-introducing broad legacy experiment paths.
- Implicit OOD claims without explicit `eval.ood` shift.
