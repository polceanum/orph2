# AGENTS.md

## Mission

Optimize LLM-agent OOD performance with reproducible, benchmark-driven **local-only** evaluation.

## Canonical command

```bash
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/gaia_lite_mock.yaml \
  --out artifacts/llm_agent/gaia_lite_mock_s0.json
```

## Rules

- Keep active work in `llm_agent/`, `scripts/run_llm_agent_eval.py`, `configs/llm_agent/`, and `benchmarks/`.
- Treat `legacy/` as archived: no new feature development there.
- Always log experiment batches in `docs/EXPERIMENT_LOG.md`:
  - Question, Hypothesis, Controls, Runs, Result, Interpretation, Decision, Next step.
- Log negative/null findings explicitly.
- Use 1 seed for fast iteration, 3+ seeds for claims.
- Use 5 seeds for stronger claims when compute allows.
- Keep benchmark/model budgets comparable when comparing agent policies.
- Always report IID and OOD separately when benchmark includes both splits.
- Always include random-chance context in summary tables.
- Include oracle/reference context when available (same task family, target-split access allowed).

## Required Comparisons (LLM track)

- `direct` baseline
- `sota_sc_verifier` baseline (rewrite + SC + verifier)
- `adaptive_router` baseline
- `adaptive_router + tools` (if tools enabled)
- `symbolic_only` baseline when tools are enabled
- `learned_program` baseline (train on IID, evaluate IID+OOD) when claiming learning-driven gains
- Same model, same benchmark split, same max task count
- Local-only rule: use `model.provider: mock` for all active runs.

## Research Log Requirement

- `docs/EXPERIMENT_LOG.md` remains the canonical journal.
- Include artifact paths for every claim.
- Mark 1-seed results as exploratory.

## Definitions (LLM track)

- IID: same template family and operator distribution as development set.
- OOD: paraphrase/format/operator-composition shift with same answer space.
- Random chance: uniform over unique gold answers in evaluated split.
- Oracle/reference: evaluation using privileged target-split adaptation/data access; do not compare transfer ratios if oracle underperforms expected ceiling.
