# LLM-Agent Direction (Active)

## Why the Pivot

The active target is benchmark-relevant LLM-agent performance (2026 landscape), not only synthetic RL OOD behavior.

## Active Loop

1. Pick benchmark split/config.
2. Run agent policy (`direct` or `plan_then_solve`).
3. Save per-task traces and aggregate metrics.
4. Iterate on orchestration, prompting, and tool strategy.
5. Log result in `docs/EXPERIMENT_LOG.md`.

## Current Runner

- Script: `scripts/run_llm_agent_eval.py`
- Package: `llm_agent/`
- On-ramp benchmark: `benchmarks/gaia_lite_v0.jsonl`
- Configs: `configs/llm_agent/`

## Current SOTA-Inspired Inference Stack

- Query rewriting for paraphrase robustness (`use_query_rewrite`)
- Self-consistency sampling (`self_consistency_k`)
- Verifier-guided answer selection (`use_verifier`)
- Majority-vote fallback when verifier selection is invalid
- Optional staged planning (`plan_then_solve`) vs direct answering
- Uncertainty-gated adaptive routing (`adaptive_router`):
  - fast cheap pass first
  - escalate to full SOTA stack only when confidence is low

## Scientific Guardrails

- Keep comparisons controlled (same tasks, same model, same budget).
- Use 1 seed for exploratory passes, 3+ seeds for acceptance.
- Store full prediction-level outputs to support error analysis.
- Record negative results.
- Use component-contribution deltas from `docs/SYSTEM_COMPONENTS_AND_CONTRIBUTIONS.md`.
- Run `scripts/validate_sota_claim_readiness.py` before any SOTA wording.

## Benchmark Roadmap

- Phase 0: local smoke (`gaia_lite_v0`)
- Phase 1: local OOD debug fixtures (`local_reasoning_ood_v1/v2/v3`)
- Phase 2: local-model (Ollama) replication on the same OOD fixtures
- Phase 3: add adapter/config for one external benchmark family
- Phase 4: compare orchestration variants at fixed budget
- Phase 5: multi-seed and cost-aware comparisons

See `docs/BENCHMARK_REALITY_PROTOCOL.md` for claim-quality requirements.
