# SOTA Baseline Policy (Claim Guard)

This policy defines when "SOTA" language is allowed.

## Required For Any SOTA Wording

1. Non-mock execution
- No SOTA claim from `model_provider: mock`.

2. Multiple strong baselines
- At minimum include: `direct`, `sota_sc_verifier`, and one tool-enabled comparator (`adaptive_tools` or `symbolic_only`).
- Include at least one learned comparator (`learned_program` or equivalent learned method) when claiming learning advantage.

3. Literature-grounded references
- For each external benchmark family, include at least 2 independent paper-reported baselines with citation links.
- Track these in `docs/LITERATURE_BASELINE_REGISTRY.json`.

4. Protocol match
- Match shot count / decoding / selection protocol to cited results, or clearly label as non-comparable.

5. Statistical minimum
- 3 seeds minimum for preliminary claim.
- 5 seeds for stronger claim.

## Validator

Run this before using SOTA wording:

```bash
conda run -n orpheus python scripts/validate_sota_claim_readiness.py \
  --benchmark-key gsm8k \
  --local-benchmark v6 \
  --seeds 0,1,2 \
  --methods sota,learned_program,symbolic_only \
  --out-json artifacts/llm_agent/sota_claim_readiness_gsm8k_check.json
```

If this check fails, do not use SOTA language in summaries.
