# Research Spec (Current)

## Thesis

For 2026-relevant progress, OOD robustness should be tested on **LLM-agent tasks** (tool use, planning, execution reliability), not only synthetic RL environments.

## Protocol

- Use benchmark-defined task sets with explicit train/dev/test or public/eval splits.
- Evaluate at least one orchestrated agent policy:
  - `direct` (single-pass answer)
  - `plan_then_solve` (two-stage orchestration)
- Keep model and budget controlled across compared methods.

## Required reporting

- Accuracy or pass@1 on the selected split
- Per-task predictions/traces
- Multi-seed mean/std where stochasticity is involved
- Cost/runtime context when available

## Benchmark Direction

- Near-term local/on-ramp: `gaia_lite_v0` fixture
- Target modern external benchmarks: SWE-bench Pro, GAIA/Gaia2, WebArena/VisualWebArena/WorkArena

## Legacy

Previous RL-first bridge/minigrid track is preserved under `legacy/` for reproducibility and historical comparison.
