# Research Spec (Current)

## Thesis

OOD adaptation improves when the agent explicitly manages competing hypotheses and updates beliefs through active probing, compared with a classical single-latent recurrent RL policy.

## Protocol

- Train on IID distribution (`env`).
- Evaluate on IID (`eval.id`) and OOD (`eval.ood`).
- Compare:
  - `structured`
  - `recurrent_rl`

## Required reporting

- IID seq accuracy
- OOD seq accuracy
- Delta (`structured - recurrent_rl`) on IID and OOD
- Multi-seed mean/std

## Current benchmark

Bridge-10 mechanism-addition split:

- Train mechanisms: `[0, 2]`
- OOD mechanisms: `[0, 1, 2]`

## Next step

Port the same protocol to at least one literature benchmark family (Procgen, Meta-World, or MiniGrid/BabyAI).
