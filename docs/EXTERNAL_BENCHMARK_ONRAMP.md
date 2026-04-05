# External Benchmark On-Ramp

This on-ramp keeps external comparisons reproducible and paper-grounded.

## 1) Prepare GSM8K JSONL

From HuggingFace (requires internet + `datasets` package):

```bash
conda run -n orpheus python scripts/prepare_gsm8k_benchmark.py \
  --source hf \
  --hf-split test \
  --ood-split-mode heuristic \
  --out benchmarks/external/gsm8k_main_test_oodheuristic_v0.jsonl
```

Notes:
- `heuristic` split is for internal IID/OOD probing only; it is not canonical GSM8K.
- For paper-comparable reporting, also run with `--ood-split-mode none` and report test accuracy directly.

## 2) Run Baselines (Example)

```bash
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/gsm8k_main_mock_sota.yaml \
  --seed 0 \
  --out artifacts/llm_agent/gsm8k_main_mock_sota_s0.json
```

```bash
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/gsm8k_main_mock_symbolic_only.yaml \
  --seed 0 \
  --out artifacts/llm_agent/gsm8k_main_mock_symbolic_only_s0.json
```

## 3) Compare Against Reported Paper Baselines

Populate `docs/LITERATURE_BASELINE_REGISTRY.json` first, then:

```bash
conda run -n orpheus python scripts/compare_with_reported_baselines.py \
  --benchmark-key gsm8k \
  --run-json artifacts/llm_agent/gsm8k_main_mock_sota_s0.json \
  --metric accuracy \
  --out-json artifacts/llm_agent/gsm8k_main_mock_sota_vs_reported.json
```

## 4) Claim Readiness Gate

```bash
conda run -n orpheus python scripts/validate_sota_claim_readiness.py \
  --benchmark-key gsm8k \
  --local-benchmark v6 \
  --seeds 0,1,2 \
  --methods sota,learned_program,symbolic_only \
  --out-json artifacts/llm_agent/sota_claim_readiness_gsm8k_check.json
```

Do not use SOTA wording if this check fails.
