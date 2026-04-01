# Copilot Instructions for `ood_solver`

See RESEARCH_SPEC.md for high-level goals.

This file defines how code should be written to respect those goals.

This project tests a structured approach to OOD problem solving:

Maintain multiple approaches, probabilistic rules, targeted probing, and soft belief updates instead of collapsing early.

---

## Core Idea

- Multiple competing approaches (hypotheses)
- Probabilistic rules (constraints)
- Active probing (information gathering)
- Belief updates (non-monotonic reasoning)

---

## Components

### Environment
Synthetic hidden mechanism tasks requiring disambiguation via probes.

### Encoder
Encodes context → latent representation. Not the reasoning core.

### Approach Proposer
Maintains multiple candidate solution strategies.

### Rule Proposer
Maintains soft constraints / expectations.

### Probe Policy
Selects informative probes to reduce uncertainty.

### Belief Updater
Updates confidence over approaches/rules.

### Solver Head
Produces final prediction.

---

## Configs

### debug.yaml
- Fast debugging only

### small.yaml
- First real experiments

### train.yaml
- Larger runs

---

## Scripts

Train structured:
python scripts/train_structured.py --config configs/small.yaml

Train recurrent:
python scripts/train_recurrent.py --config configs/small.yaml

Evaluate:
python scripts/eval_baselines.py --config configs/small.yaml

---

## Evaluation priorities

- Structured > recurrent
- Performance by mechanism
- Probe usefulness
- Entropy behavior

---

## Known issue

- Overconfident collapse (low entropy, weak performance)

Focus:
- entropy regularization
- probe learning
- better training regime

---

## Development rules

- Keep modular structure
- Avoid collapsing to single latent
- Always add ablations
- Prioritize interpretability

---

## Next tasks

- random probe baseline
- entropy tuning
- better logging
