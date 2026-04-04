# Benchmark Reality Protocol (LLM OOD, 2026)

This file keeps experiments grounded in externally relevant practice rather than only local toy wins.

## Goal

Demonstrate reproducible OOD gains from orchestration/tooling, not just prompt luck on one fixture.

## Minimum Evidence For A Claim

1. Controlled baselines on the same benchmark split and same model budget:
- `direct`
- `sota_sc_verifier`
- `adaptive_router`
- `adaptive_router + tools` (if tools are part of the method)
- `symbolic_only` when tools are part of the method (to isolate orchestration value from tool value)

2. Split-aware reporting:
- overall accuracy
- IID accuracy
- OOD accuracy
- structured delta vs baseline methods

3. Reference context:
- random-chance accuracy
- oracle/reference run where possible

4. Statistical hygiene:
- 1 seed = exploratory
- 3 seeds = preliminary claim
- 5 seeds = stronger claim

## Practical Benchmark Ladder (Mac-Friendly First)

1. Local reasoning OOD fixtures (`v1`-`v6`)
- Purpose: rapid debugging of routing/tool correctness
- Not sufficient for SOTA claims

2. Open local-model benchmarks (Ollama backend)
- Same benchmark files, but with real local LLM behavior
- Confirms effects beyond synthetic mock behavior

3. External literature-facing benchmarks (next)
- Candidate families:
  - GSM8K-style math reasoning splits
  - MMLU-style knowledge QA with held-out domains
  - BIG-Bench Hard style compositional tasks
  - agentic/tool benchmarks with explicit task decompositions
- Requirement: preserve IID/OOD split protocol and comparable inference budget.

## SOTA-Inspired Methods To Keep As Comparators

- Self-consistency sampling
- Verifier/reranker selection
- Query rewriting/paraphrase normalization
- Uncertainty-triggered adaptive compute
- Tool-augmented reasoning (symbolic/external actions)

## Anti-Patterns

- Claiming "works" from one seed only
- Comparing methods with different model sizes or token budgets
- Reporting only aggregate accuracy without IID/OOD split
- Ignoring failed/negative variants

## Required Logging

After each meaningful batch, append `docs/EXPERIMENT_LOG.md` with:
- Question
- Hypothesis
- Controls
- Runs (artifact paths)
- Result
- Interpretation
- Decision
- Next step

## Claim Gate

Before any "SOTA" wording, run:

```bash
conda run -n orpheus python scripts/validate_sota_claim_readiness.py \
  --benchmark-key <benchmark_family_key> \
  --local-benchmark <local_benchmark_tag> \
  --seeds 0,1,2 \
  --methods sota,learned_program,symbolic_only \
  --out-json <claim_check.json>
```

Reference files:
- `docs/SOTA_BASELINE_POLICY.md`
- `docs/LITERATURE_BASELINE_REGISTRY.json`
