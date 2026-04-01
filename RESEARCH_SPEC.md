# OOD Solver — Research Specification

Implementation details: see `COPILOT_INSTRUCTIONS.md`  
Execution workflow: see `AGENTS.md`

## 1. Problem Statement

Modern neural networks, especially transformers, can perform well in-distribution while remaining brittle out-of-distribution.

Common failure modes:

- early collapse to a single latent interpretation
- weak or implicit uncertainty tracking
- no explicit mechanism for information gathering
- poor recovery when early assumptions are wrong

This project asks:

> Can OOD generalization improve if the model reasons through explicit, revisable belief state instead of compressing everything into one latent guess?

This is not a scaling project first. It is a structured reasoning project first.

---

## 2. Core Hypothesis

The central hypothesis is:

> Robust OOD reasoning requires structured belief management, not just larger models or more data.

The system should:

- maintain multiple competing approaches
- maintain probabilistic, testable rules
- actively select informative probes
- update beliefs softly rather than through hard elimination
- support non-monotonic reasoning, including revisiting previously downweighted ideas

Shorthand:

Approaches + Rules + Probes + Belief Updates + Recovery

---

## 3. Conceptual Model of Human-Like Reasoning

This work is inspired by a process view of human problem-solving:

1. Start with a goal and partial evidence
2. Form multiple candidate interpretations
3. Derive soft constraints or expectations
4. Test ideas through probes or experiments
5. Detect contradiction, surprise, or lack of progress
6. Update confidence, switch strategies, or revisit archived possibilities

Important properties we want the model to reflect:

- reasoning is parallel over hypotheses
- hypotheses are internally coherent but compete globally
- contradictions often arise between hypotheses, not just inside one
- confidence should be graded, not binary
- rules should be probabilistic but testable
- wrong early commitment should be recoverable

---

## 4. Architecture Overview

### High-level pipeline

1. Encode context
2. Initialize:
   - approach slots
   - rule slots
3. For each probe step:
   - select a probe
   - observe the result
   - update belief over approaches and rules
4. Produce final prediction

### Key architectural commitments

- multiple competing hypotheses must exist explicitly
- belief state must remain inspectable
- updates should be soft, not hard rejection
- the system should be able to recover after misleading early evidence
- the structured path should not be quietly replaced by a single recurrent latent unless implementing a baseline

---

## 5. Components

### Approaches

- latent hypotheses representing different possible solution strategies
- probabilistically weighted rather than strictly exclusive

### Rules

- latent probabilistic expectations or constraints
- examples conceptually:
  - "if I probe X, Y should happen"
  - "under this framing, Z should be unlikely"
- used to guide belief updates and probe selection

### Probe Policy

- chooses experiments or queries that help discriminate between approaches
- should move the model toward useful uncertainty reduction, not merely action repetition

### Belief Updater

- updates confidence over approaches and rules
- must support soft updates and later recovery
- should allow surprise and stagnation to affect reasoning state

### Archive / Recovery Memory

- preserves useful but currently downweighted candidates
- intended to support recovery when current reasoning stalls or becomes contradicted

### Solver Head

- produces final output conditioned on context and current belief state
- should reflect the structured reasoning state rather than bypass it

---

## 6. Relation to Existing Work

### Transformers / attention

Attention already resembles fast contextual adaptation, but standard transformers typically do not maintain persistent competing hypotheses or explicit belief revision state.

### Meta-learning / learned optimizers

These learn adaptation procedures, but often collapse to heuristics and lack explicit hypothesis structure.

### MCTS / AlphaZero analogies

There is a useful analogy to branching hypothesis search, evaluation, and selection. The difference here is that the reasoning is amortized and mostly continuous rather than explicit discrete tree search.

### World models / predictive consistency work

These capture structure in prediction, but usually lack explicit hypothesis competition, active probing, and revisable belief tracking.

---

## 7. Synthetic Environment Rationale

The current benchmark is intentionally synthetic.

Why:

- it allows controlled ambiguity
- it makes probing necessary rather than decorative
- it exposes whether the model can disambiguate hidden mechanism families
- it provides interpretable failure modes and mechanism-level analysis

The synthetic setup is a testbed for the reasoning process, not the final destination.

---

## 8. Empirical Status So Far

### What already works

- the structured architecture is trainable
- the model can overfit small datasets
- entropy over approaches and rules can decrease in a learnable way
- the structured system can slightly outperform a simple recurrent baseline

This means the architecture is functional enough to study.

### What does not work yet

- held-out performance remains weak
- gains over baseline are still small
- approach entropy and rule entropy tend to collapse too quickly

This means the model has learned how to commit, but not when to commit.

---

## 9. Main Failure Diagnosis

The current central failure mode is:

## Premature Belief Collapse

Symptoms:

- approach entropy goes near zero too early
- rule entropy also collapses quickly
- the model becomes overconfident before the evidence justifies it
- probing does not create enough useful uncertainty reduction
- recovery behavior remains weak

Interpretation:

- this is more a calibration and belief-dynamics problem than a raw capacity problem
- the issue is not simply "make the model bigger"
- the issue is that belief updates are too eager and insufficiently calibrated

The current project should treat belief calibration as a first-class research target.

---

## 10. Metrics

### Primary metrics

- sequence accuracy
- per-mechanism sequence accuracy

### Secondary metrics

- approach entropy
- rule entropy
- probe efficiency or probe usefulness
- surprise behavior
- wrong-frame persistence
- recovery after misleading evidence

### Interpretation guidance

Do not treat raw accuracy alone as sufficient.

A result is more convincing if:

- structured beats recurrent
- some mechanism families improve clearly
- entropy remains useful for longer
- probes appear to matter
- ablations degrade behavior in the expected way

---

## 11. Success Criteria

### Short term

- structured > recurrent on `small.yaml`
- gains visible on at least some mechanisms
- entropy does not collapse immediately

### Medium term

- learned probes outperform random probes
- no-rules and no-archive ablations degrade performance in interpretable ways
- surprise and stagnation signals correlate with meaningful recovery behavior

### Long term

- robust adaptation to unseen task families
- meaningful recovery after incorrect early assumptions
- evidence that the system is learning a reasoning process, not just a better heuristic shortcut

---

## 12. Experimental Roadmap

### Phase 1

- sanity checks
- pipeline validation
- debug runs
- tiny overfit

Status: completed enough to proceed

### Phase 2

- train on `small.yaml`
- compare structured vs recurrent
- inspect per-mechanism behavior
- inspect entropy behavior

Status: current focus

### Phase 3

Critical ablations:

- random probes
- no rules
- no archive / recovery memory

Questions:

- does probe learning matter?
- do rules contribute causally?
- does recovery memory actually help?

### Phase 4

- entropy regularization
- improved probe objectives
- better surprise and stagnation usage
- stronger recovery mechanisms

---

## 13. Non-goals

For now, do not optimize around:

- scaling model size as the main lever
- integrating large pretrained models as the reasoning core
- symbolic rule extraction for its own sake
- natural-language explanations as the main objective

Large pretrained models may later serve as encoders, but not as a replacement for the structured controller.

---

## 14. Development Principles

- preserve explicit structured state
- keep modules separable and ablation-friendly
- optimize for interpretable failure modes
- prefer controlled experiments over architectural drift
- avoid shortcuts that bypass the reasoning hypothesis

If a change improves benchmark numbers by collapsing the architecture toward a single latent process, that is likely a research failure even if it is an engineering success.

---

## 15. Canonical Workflow

Debug:

```bash
python scripts/overfit_tiny.py --config configs/debug.yaml
```

Train structured on small:

```bash
python scripts/train_structured.py --config configs/small.yaml
```

Train recurrent on small:

```bash
python scripts/train_recurrent.py --config configs/small.yaml
```

Evaluate:

```bash
python scripts/eval_baselines.py --config configs/small.yaml \
  --structured-ckpt artifacts/structured_last.pt \
  --recurrent-ckpt artifacts/recurrent_last.pt
```

---

## 16. Interpretation Guidelines

Do not trust:

- debug results
- tiny overfit results
- single-run anecdotes without baseline comparison

Trust more:

- `small.yaml` and `train.yaml` runs
- held-out evaluation
- per-mechanism breakdowns
- ablation comparisons
- entropy and probe behavior alongside accuracy

If structured slightly beats recurrent but entropy collapses immediately and ablations do not matter, the result is weak.

---

## 17. Next Critical Experiment

Train on `small.yaml` and compare:

- structured
- recurrent
- random-probe structured

This should answer:

- does structure help?
- do learned probes matter?
- is the belief system doing useful causal work?

That experiment is the current gate for deciding whether the architecture is substantively promising.
