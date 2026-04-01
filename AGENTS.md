# AGENTS.md

See RESEARCH_SPEC.md for experiment rationale.
This file defines how to run and evaluate experiments.

## Mission
Build a structured reasoning system for OOD generalization.

---

## Success Criteria

Short term:
- Structured > recurrent on small.yaml
- Gains visible per mechanism
- Entropy not collapsing immediately

Medium term:
- Recovery after surprise
- Archive usefulness

---

## Workflow

1. Debug (debug.yaml)
2. Overfit sanity
3. Train small.yaml
4. Compare structured vs recurrent
5. Add ablations

---

## Commands

Debug:
python scripts/overfit_tiny.py --config configs/debug.yaml

Small:
python scripts/train_structured.py --config configs/small.yaml
python scripts/train_recurrent.py --config configs/small.yaml
python scripts/eval_baselines.py --config configs/small.yaml

Train:
python scripts/train_structured.py --config configs/train.yaml
python scripts/train_recurrent.py --config configs/train.yaml

---

## Rules

- Do not collapse architecture into single latent
- Do not trust debug results
- Always compare against baselines
- Keep experiments controlled
