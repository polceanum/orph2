# Insight Originality Review

Purpose: keep us honest about what is genuinely new from our hypothesis versus what is inherited from prior art.

## Core Project Insight (from our discussion)

Primary claim:
- Robust OOD performance requires **structured, revisable belief dynamics** (multiple hypotheses, uncertainty-aware updates, informative probing), not just larger monolithic predictors.

## Current Components: Origin Assessment

1. `sota_sc_verifier` (rewrite + self-consistency + verifier)
- Lineage: mainstream LLM inference-time scaling patterns.
- Status: not novel by itself; used as a strong comparator.

2. `adaptive_router`
- Lineage: conditional-compute / selective escalation patterns.
- Status: partly standard; implementation details are ours.

3. `symbolic_only` / `adaptive_tools`
- Lineage: tool-augmented reasoning and neuro-symbolic execution.
- Status: not novel conceptually; useful performance ceiling/reference in our framework.

4. `learned_program` (IID-trained type predictor + typed executor)
- Lineage: program-of-thought / neuro-symbolic decomposition.
- Status: moderate novelty in this repo context (learned dispatch under IID->OOD protocol), but conceptually adjacent to existing methods.

## What Is Still Missing To Realize Original Thesis

The strongest original part of our thesis is not yet fully implemented:
- explicit competing hypotheses with persistence over steps
- belief calibration dynamics under contradictory evidence
- active probe selection optimized for information gain
- non-monotonic revision (recovering from early wrong commits)

Current stack is a practical stepping stone, but still mostly "routing + tools + typed execution."

## Scientific Claim Boundaries (Current)

Supported now:
- Learned method beats our SOTA-style non-tool baseline on v2/v3/v4/v5 under controlled IID->OOD evaluation.
- Broadening to v5 exposed and then fixed a concrete distractor-robustness bug in operand parsing.

Not supported yet:
- Claiming overall SOTA against literature benchmarks.
- Claiming our gains come from full structured-belief dynamics as originally hypothesized.

## Next Priority For True Differentiation

1. Add explicit belief-state module with multiple candidate latent programs.
2. Train a probe/action policy to reduce belief entropy, not only answer loss.
3. Evaluate belief-calibration metrics (entropy trajectory, recovery after contradiction).
4. Test on external benchmarks with fixed-budget baselines and seed-robust statistics.
