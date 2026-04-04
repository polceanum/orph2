# Copilot Instructions

This repository is now intentionally narrowed to an LLM-agent benchmark track.

## Preserve

- Reproducible benchmark configs (`configs/llm_agent/`)
- Agent orchestration logic (`llm_agent/`)
- Traceable evaluation outputs (`artifacts/llm_agent/`)

## Prioritize

- Benchmark-grounded metrics with transparent settings
- Controlled comparisons of orchestration policies (`direct` vs `plan_then_solve`)
- Fast iteration on Mac (mock/local first, hosted API runs second)
- Explicit experiment logging in `docs/EXPERIMENT_LOG.md`

## Scientific Protocol

- Keep model, benchmark split, and task budget fixed when comparing methods.
- Record provider/model/version in result artifacts.
- Use:
  - 1 seed for exploratory iteration
  - 3+ seeds for acceptance claims
- Treat 1-seed findings as exploratory.
- Include failed/null findings in the log.
- Do not claim SOTA from local fixtures; use them as smoke/on-ramp only.

## Standardization Path

- Keep `gaia_lite_v0` as local runner sanity check.
- Add adapters/configs for modern agent benchmarks incrementally:
  - SWE-bench Pro (coding agents)
  - GAIA/Gaia2 (general assistant tasks)
  - WebArena/VisualWebArena/WorkArena (web agents)

## Avoid

- Re-activating RL-first legacy code as the main research path.
- Mixing legacy and active benchmarks in one headline metric.
