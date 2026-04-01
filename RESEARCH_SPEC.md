# OOD Solver — Research Specification

Implementation details: see COPILOT_INSTRUCTIONS.md  
Execution workflow: see AGENTS.md

## 1. Objective

Develop a model that can generalize to unseen tasks by:
- maintaining multiple competing hypotheses (approaches)
- using probabilistic, testable rules
- actively selecting informative probes
- updating beliefs without premature collapse

This is NOT a scaling experiment. It is an architecture + reasoning experiment.

---

## 2. Core Hypothesis

Standard models fail OOD because they:
- collapse too early to a single latent interpretation
- lack structured uncertainty
- do not actively gather information

We test whether the following improves OOD:

Approaches + Rules + Probes + Belief Updates

---

## 3. Architecture Overview

Pipeline:

1. Encode context
2. Initialize:
   - approach slots
   - rule slots
3. For each probe step:
   - select probe (probe policy)
   - observe result
   - update belief (belief updater)
4. Produce final prediction

Key properties:
- multiple competing hypotheses
- soft belief updates
- no hard rejection
- potential recovery

---

## 4. Components

### Approaches
- latent hypotheses
- represent different solution strategies

### Rules
- probabilistic constraints
- guide pruning and probe selection

### Probe Policy
- selects most informative probe
- reduces uncertainty

### Belief Updater
- updates scores for approaches/rules
- enables non-monotonic reasoning

---

## 5. Expected Failure Modes

Current likely issues:

1. Premature collapse
   - entropy → near zero too early
   - incorrect overconfidence

2. Probe inefficiency
   - probes not reducing uncertainty

3. Mode imitation
   - behaving like recurrent baseline

---

## 6. Experiments Roadmap

### Phase 1 (current)
- sanity + pipeline validation
- debug + overfit

### Phase 2 (now)
- train on small.yaml
- compare structured vs recurrent
- evaluate per mechanism

### Phase 3
- ablations:
  - random probes
  - no rules
  - no archive

### Phase 4
- improved probe learning
- entropy calibration
- recovery behavior

---

## 7. Metrics

Primary:
- sequence accuracy
- accuracy per mechanism

Secondary:
- approach entropy
- rule entropy
- probe efficiency

---

## 8. Success Criteria

Short-term:
- structured > recurrent (clear margin)
- gains on at least some mechanisms
- entropy not collapsing immediately

Mid-term:
- learned probes outperform random
- structured components show causal impact

---

## 9. Non-goals (for now)

- scaling model size
- integrating large pretrained models
- symbolic rule extraction
- language explanations

---

## 10. Development Principles

- modular components
- ablation-friendly design
- explicit structured state
- avoid shortcuts bypassing architecture

---

## 11. Canonical Workflow

Debug:
python scripts/overfit_tiny.py --config configs/debug.yaml

Train (small):
python scripts/train_structured.py --config configs/small.yaml
python scripts/train_recurrent.py --config configs/small.yaml

Evaluate:
python scripts/eval_baselines.py --config configs/small.yaml

---

## 12. Interpretation Guidelines

Do NOT trust:
- debug results
- tiny overfit results

Trust:
- small/train configs
- held-out evaluation
- per-mechanism breakdown

---

## 13. Next Critical Experiment

Train on small.yaml and compare:

structured vs recurrent vs random-probe structured

This will determine if:
- structure helps
- probes matter
- belief system is doing useful work
