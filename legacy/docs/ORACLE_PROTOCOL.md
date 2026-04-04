# Oracle Protocol

## Purpose

Ensure oracle comparisons are valid upper bounds rather than underfit references.

## Definition

An oracle run is trained on the target OOD distribution itself (the same distribution used as OOD in the primary run).

## Generation

Use:

```bash
conda run -n orpheus python scripts/make_oracle_config.py \
  --primary <primary_config.yaml> \
  --out <oracle_config.yaml> \
  --epochs-mult 1.0 \
  --steps-mult 1.0
```

Optional stricter oracle:

- increase `--epochs-mult` and/or `--steps-mult` (e.g., `1.25x` or `1.5x`)

## Validation Gate

After oracle run completes:

```bash
conda run -n orpheus python scripts/validate_oracle_quality.py \
  --primary <primary_result.json> \
  --oracle <oracle_result.json> \
  --min-margin 0.005
```

Expected:

- oracle OOD should exceed primary OOD for each comparable method by at least margin.

If gate fails:

- treat oracle as underfit/inconclusive
- do not use transfer-ratio claims
- rerun oracle with increased budget and/or adjusted optimization

## Reporting

- Always include random chance and oracle context via:
  - `scripts/summarize_rl_results.py`
- Mark results clearly when oracle gate failed.
