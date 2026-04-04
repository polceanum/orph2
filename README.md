# OOD Solver (LLM-Agent Pivot)

This repository now focuses on **LLM-agent orchestration and evaluation** for modern OOD-relevant benchmarks, with a Mac-friendly API-first workflow.

Legacy RL-first experiments are preserved under `legacy/`.

## Active Goal

Build and evaluate orchestrated LLM agents (planning + solving loops) with explicit, reproducible benchmark reporting.

## Quick Start

Run local mock evaluation (no external API required):

```bash
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/gaia_lite_mock.yaml \
  --out artifacts/llm_agent/gaia_lite_mock_s0.json
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

Optional local model backend (Ollama, no API keys):

```bash
ollama serve
ollama pull llama3.1:8b
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/local_reasoning_ood_ollama.yaml \
  --out artifacts/llm_agent/local_reasoning_ood_ollama_s0.json
```

Run OpenAI-backed evaluation (requires `OPENAI_API_KEY`):

```bash
conda run -n orpheus python scripts/run_llm_agent_eval.py \
  --config configs/llm_agent/gaia_lite_openai.yaml \
  --out artifacts/llm_agent/gaia_lite_openai_s0.json
```

## Active Structure

- `scripts/run_llm_agent_eval.py`: benchmark runner
- `scripts/summarize_llm_agent_results.py`: result aggregator with random-chance/oracle reference fields
- `llm_agent/`: orchestration, model clients, eval utilities
- `benchmarks/gaia_lite_v0.jsonl`: minimal local benchmark fixture
- `benchmarks/local_reasoning_ood_v1|v2|v3.jsonl`: local IID/OOD debugging ladder
- `configs/llm_agent/`: runner configs

## Reality Grounding

Use `docs/BENCHMARK_REALITY_PROTOCOL.md` as the scientific guardrail for claims:
- always include controlled baselines
- always report IID/OOD split metrics
- include random-chance and oracle/reference context
- do not treat 1-seed outcomes as conclusions

## Legacy Track

RL/minigrid/bridge experiments are archived for traceability:

- `legacy/scripts/`
- `legacy/configs/`
- `legacy/packages/ood_solver/`
- `legacy/docs/`

## Notes

- This pivot is intentionally practical for Mac hardware: evaluate agent logic locally, call hosted models via API for capability.
- The current `gaia_lite` benchmark is a smoke/on-ramp fixture; it is not intended as a final SOTA claim benchmark.
- To run local-model benchmarks without API keys, start Ollama first (`ollama serve`) and pull a model (`ollama pull llama3.1:8b`).
