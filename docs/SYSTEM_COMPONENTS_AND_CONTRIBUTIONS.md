# System Components And Contribution Map

This file makes module intent, contribution, and fair-ablation interpretation explicit.

## Core Components

1. `direct`
- Single-pass answer generation.
- Purpose: minimum viable baseline.

2. `sota_sc_verifier`
- Rewrite/paraphrase normalization + self-consistency + verifier selection.
- Purpose: strong inference-time baseline inspired by current LLM practice.

3. `adaptive_router`
- Cheap first pass, escalate to `sota_sc_verifier` on low confidence.
- Purpose: efficiency-aware quality.

4. `adaptive_router + tools`
- Same routing, but solver can use symbolic/tool path.
- Purpose: isolate value from external reasoning tools.

5. `symbolic_only`
- Direct symbolic/tool execution without orchestration extras.
- Purpose: isolate tool value from orchestration value.

6. `learned_program`
- IID-trained type predictor + typed executor.
- Purpose: learned decomposition comparator to non-learned tool baselines.

## Contribution Deltas (Ablation Semantics)

Use these directional deltas to attribute gains:

- `rewrite_sc_verifier_over_direct = sota_sc_verifier - direct`
- `adaptive_over_sota = adaptive_router - sota_sc_verifier`
- `tools_over_adaptive = adaptive_router_tools - adaptive_router`
- `symbolic_over_tools = symbolic_only - adaptive_router_tools`
- `learned_over_sota = learned_program - sota_sc_verifier`
- `learned_over_symbolic = learned_program - symbolic_only`

Interpretation:
- Positive `tools_over_adaptive`: tool value exists beyond orchestration.
- Positive `learned_over_symbolic`: learning adds value beyond fixed symbolic logic.
- Near-zero `learned_over_symbolic`: parity; avoid claiming learned superiority.

## Reporting Command

```bash
conda run -n orpheus python scripts/report_component_contributions.py \
  --benchmarks v2 v3 v4 v5 v6 \
  --seeds 0,1,2 \
  --out-json artifacts/llm_agent/component_contrib_v2_v6_s012.json \
  --out-md artifacts/llm_agent/component_contrib_v2_v6_s012.md
```

## Fairness Rules

- Same benchmark split for every compared method.
- Same backend/provider for every compared method in a claim.
- Same inference budget class (SC k, routing policy, tool access clearly declared).
- 1 seed exploratory only; 3 seeds preliminary; 5 seeds stronger.
