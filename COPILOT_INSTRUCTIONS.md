# Copilot Instructions for `ood_solver`

See `RESEARCH_SPEC.md` for the research thesis.

This file translates the thesis into implementation guidance.

## Purpose

This project is not trying to learn a better black-box predictor.

It is trying to learn a process of reasoning under uncertainty with:

- multiple approaches
- probabilistic rules
- active probing
- soft belief updates
- possible recovery after wrong early commitment

Code changes should preserve that intent.

---

## What The Architecture Must Preserve

### Multiple competing approaches

- keep explicit approach slots
- avoid merging them into one hidden state except inside a baseline

### Rules as first-class state

- rules should remain explicit latent objects or scores, not disappear into undifferentiated activations

### Probe loop

- probing is not decorative
- probe choice should matter for disambiguation

### Belief dynamics

- belief updates should be soft and revisable
- avoid designs that force irreversible early commitment

### Recovery support

- preserve archive or related recovery mechanisms
- surprise and stagnation should be able to influence future reasoning

---

## Mental Model For The Codebase

### Environment

Creates tasks where early evidence is ambiguous and probes are useful.

### Encoder

Encodes observations into latent features. It is not the reasoning core.

### Approach Proposer

Creates candidate solution framings.

### Rule Proposer

Creates soft constraints and expectations tied to the current reasoning state.

### Probe Policy

Chooses the most informative next query or experiment.

### Belief Updater

Revises confidence over approaches and rules after each probe result.

### Solver Head

Produces the final answer from the current belief state.

---

## What To Optimize For

Prioritize changes that improve:

- structured > recurrent comparison
- per-mechanism gains
- delayed or better-calibrated entropy collapse
- probe usefulness
- recovery after misleading evidence

Prefer metrics and logging that reveal:

- whether hypotheses remain alive long enough
- whether rules are informative or dead weight
- whether probing changes belief in meaningful ways
- whether archived candidates ever become useful again

---

## Current Failure Diagnosis

The main current failure mode is premature belief collapse.

Symptoms:

- approach entropy goes low too early
- rule entropy goes low too early
- the model commits before it has earned confidence
- held-out gains remain weak

Therefore, be especially cautious about changes that:

- improve training loss by encouraging earlier commitment
- reduce uncertainty without improving downstream robustness
- bypass the probe loop
- make the structured model behave like a recurrent baseline

---

## Coding Rules

- keep modules clean and separable
- keep ablations easy to implement
- prefer explicit state over hidden coupling
- do not quietly remove interpretability hooks
- log metrics that help diagnose belief dynamics

When adding a feature, ask:

1. Does it preserve explicit structured state?
2. Can it be ablated cleanly?
3. Does it help probing, calibration, or recovery?
4. Does it accidentally collapse the architecture into a single latent?

If the answer to 4 is yes, rethink it.

---

## Evaluation Priorities

Always compare against baselines.

Minimum useful evaluation questions:

- is structured better than recurrent?
- on which mechanisms?
- what happens to approach entropy?
- what happens to rule entropy?
- are probes helping?

Strong results should show both:

- performance gains
- evidence that the gains come from the intended structured mechanisms

---

## Preferred Ablations

If you add or change architecture, try to keep these comparisons available:

- random probes
- no rules
- no archive
- recurrent baseline

These are important because they tell us whether the structured components have causal impact.

---

## Config Expectations

### `debug.yaml`

- for fast iteration only
- never use as final evidence

### `small.yaml`

- first real comparison point
- primary near-term benchmark

### `train.yaml`

- larger runs after the experiment story is already coherent on `small.yaml`

---

## What Not To Do

- do not optimize only for debug overfit
- do not replace the structured controller with a larger generic sequence model
- do not hard-code mechanism-specific solutions into the main path
- do not use one-off tricks that make ablations impossible
- do not interpret slight gains as success if entropy collapses immediately

---

## Useful Next Tasks

- random probe baseline
- entropy regularization tuning
- better logging of belief trajectories
- explicit analysis of surprise and stagnation
- archive usefulness diagnostics
