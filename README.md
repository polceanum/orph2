# OOD Solver (LLM-Agent Pivot)

This repository now focuses on **LLM-agent orchestration and evaluation** for modern OOD-relevant benchmarks, with a strict **local-only** workflow.

## Active Goal

Build and evaluate orchestrated LLM agents (planning + solving loops) with explicit, reproducible benchmark reporting.

## Quick Start

Run local mock evaluation:

```bash
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/gaia_lite_mock.yaml \
  --out artifacts/llm_agent/gaia_lite_mock_s0.json
```

For long runs, prefer live progress and compact artifacts:

```bash
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/gsm8k_main_mock_sota.yaml \
  --progress-every 25 \
  --no-save-predictions \
  --out artifacts/llm_agent/gsm8k_main_mock_sota_s0_nopreds.json
```

Run local OOD-style benchmark (mock, no keys):

```bash
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/local_reasoning_ood_mock.yaml \
  --out artifacts/llm_agent/local_reasoning_ood_mock_s0.json
```

Run SOTA-style local inference stack (rewrite + self-consistency + verifier):

```bash
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/local_reasoning_ood_mock_sota.yaml \
  --out artifacts/llm_agent/local_reasoning_ood_mock_sota_s0.json
```

Run adaptive router (cheap pass + uncertainty-triggered escalation):

```bash
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/local_reasoning_ood_mock_adaptive.yaml \
  --out artifacts/llm_agent/local_reasoning_ood_mock_adaptive_s0.json
```

All active configs are local-only (`model.provider: mock`).

## Active Structure

- `scripts/run_llm_agent_eval.py`: benchmark runner
- `scripts/summarize_llm_agent_results.py`: result aggregator with random-chance/oracle reference fields
- `llm_agent/`: orchestration, model clients, eval utilities
- `benchmarks/gaia_lite_v0.jsonl`: minimal local benchmark fixture
- `benchmarks/local_reasoning_ood_v1|v2|v3|v4|v5|v6.jsonl`: local IID/OOD debugging ladder (v5/v6 add stronger distractor/noise robustness checks)
- `benchmarks/external/gsm8k_main_test_oodheuristic_v0.jsonl`: GSM8K local adapter with heuristic IID/OOD split
- `benchmarks/external/gsm8k_main_test_ood_typeholdout_v1.jsonl`: stricter type/complexity holdout split
- `benchmarks/external/gsm8k_main_test_ood_lengthholdout_v1.jsonl`: length-based OOD split
- `configs/llm_agent/`: runner configs
- `scripts/compare_llm_agent_matrix.py`: cross-benchmark multi-seed comparison vs reference method
- `scripts/compare_llm_agent_benchmark.py`: single-benchmark multi-method comparison (IID/OOD + deltas + random-chance)
- `scripts/report_component_contributions.py`: ablation-style component contribution deltas
- `scripts/validate_sota_claim_readiness.py`: SOTA-claim guardrail checker
- `scripts/prepare_gsm8k_benchmark.py`: external GSM8K JSONL adapter/prep
- `scripts/compare_with_reported_baselines.py`: compare run metrics to paper-reported baselines
- `docs/SYSTEM_COMPONENTS_AND_CONTRIBUTIONS.md`: component map + contribution semantics
- `docs/SOTA_BASELINE_POLICY.md`: strict claim policy
- `docs/LITERATURE_BASELINE_REGISTRY.json`: paper-grounded comparator registry
- `docs/EXTERNAL_BENCHMARK_ONRAMP.md`: external benchmark workflow

## Reality Grounding

Use `docs/BENCHMARK_REALITY_PROTOCOL.md` as the scientific guardrail for claims:
- always include controlled baselines
- always report IID/OOD split metrics
- include random-chance and oracle/reference context
- do not treat 1-seed outcomes as conclusions

## Notes

- This pivot is intentionally practical for Mac hardware: evaluate agent logic with local-only, reproducible mock-backed runs.
- The current `gaia_lite` benchmark is a smoke/on-ramp fixture; it is not intended as a final SOTA claim benchmark.
- The local `local_reasoning_ood_v*` ladder is for debugging/regression; it is not by itself valid external SOTA evidence.
- External API/provider runs are intentionally removed from active workflow.
