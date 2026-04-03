# OOD Solver Prototype

Phase-1 PyTorch project for testing a structured OOD problem-solving loop.

The working hypothesis is that robust OOD solving benefits from:

- multiple competing **approaches**
- a revisable pool of probabilistic, testable **rules**
- targeted **probes**
- learned **belief updates**
- **surprise** and **stagnation** signals
- an **archive** for recovery and backtracking

The current benchmark is synthetic by design, but the architecture is meant to stay modular so the encoder can later be replaced with text, vision, or multimodal front ends.

## Repository intent

This repo is aimed at a narrow first question:

> Does an explicit, revisable multi-hypothesis controller improve recovery from wrong framings compared with simpler baselines?

The broader thesis behind that question is:

- OOD failure is often a belief-management failure, not just a capacity failure
- standard models tend to commit too early to one latent interpretation
- robust solving may require explicit competing hypotheses, testable rules, active probing, and recovery

## Canonical workflow

Use these as the primary entrypoints:

```bash
python scripts/overfit_tiny.py --config configs/debug.yaml
python scripts/train_structured.py --config configs/small.yaml
python scripts/train_recurrent.py --config configs/small.yaml
python scripts/eval_baselines.py --config configs/small.yaml \
  --structured-ckpt artifacts/structured_last.pt \
  --recurrent-ckpt artifacts/recurrent_last.pt
```

Ablation entrypoints:

```bash
python scripts/train_random_probe.py --config configs/small.yaml
python scripts/train_no_rules.py --config configs/small.yaml
```

Multi-seed suite (recommended for robust comparison):

```bash
python scripts/run_ablation_suite.py \
  --config configs/small_tuned.yaml \
  --seeds 0,1,2 \
  --tag small_tuned_suite_v1 \
  --eval-batches 30
```

Fast iteration loop (single seed):

```bash
python scripts/run_ablation_suite.py \
  --config configs/core_stage1_v3.yaml \
  --seeds 0 \
  --tag core_stage1_v3_seed0_iter \
  --eval-batches 20
```

Current best stage-1 sweep (as of April 2, 2026):

```bash
python scripts/run_ablation_suite.py \
  --config configs/core_stage1_v3.yaml \
  --seeds 0,1,2 \
  --tag core_stage1_v3_seed012 \
  --eval-batches 30
```

Bridge benchmark pack (multiple learnable OOD scenarios):

```bash
python scripts/run_bridge_benchmark.py \
  --benchmark-config configs/bridges/benchmark.yaml \
  --seeds 0 \
  --eval-batches 20 \
  --tag-prefix bridge_quick \
  --with-oracle
```

This runs staged scenarios and (optionally) oracle controls trained directly on each scenario's OOD regime.

OOD split eval (IID vs shifted distribution):

```bash
python scripts/eval_ood_baselines.py \
  --config configs/small_ood_paramshift.yaml \
  --structured-ckpt artifacts/structured_last.pt \
  --random-probe-ckpt artifacts/random_probe_last.pt \
  --no-rules-ckpt artifacts/no_rules_last.pt \
  --recurrent-ckpt artifacts/recurrent_last.pt \
  --num-batches 40
```

OOD output now includes:

- `seq_acc_gap = ood_seq_acc - iid_seq_acc`
- `seq_acc_retention = ood_seq_acc / iid_seq_acc`
- `loss_gap`, `loss_ratio`
- per-batch arrays (`batch_overall.seq_acc`, `batch_overall.loss`) for uncertainty estimation

Bridge benchmarking now compares oracle on the *same target OOD split* as base runs (apples-to-apples),
reported as:

- `oracle_structured_target_ood_seq_acc`
- `structured_ood_oracle_gap`
- `structured_adaptation_efficiency = structured_ood_seq_acc / oracle_structured_target_ood_seq_acc`

Compatibility scripts are also available:

- `python scripts/train.py --config configs/debug.yaml`
- `python scripts/eval.py --config configs/debug.yaml --ckpt artifacts/last.pt`
- `python scripts/inspect_episode.py --config configs/debug.yaml --ckpt artifacts/last.pt`

## Quick start

```bash
conda activate orpheus
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/overfit_tiny.py --config configs/debug.yaml --epochs 1 --num-fixed-batches 1
```

This repo is expected to run from the `orpheus` conda environment, which contains the working custom macOS PyTorch build. The `base` conda environment may have a different Torch install that crashes on import.

If `pytest` or `torch` is missing in `orpheus`, the environment is not bootstrapped yet and the training scripts and tests will not run.

When running from automation/sandboxed environments, Torch may crash with OpenMP shared-memory errors (`OMP: Error #179`). In that case, run commands through `conda run -n orpheus ...` so the known-good environment is used.

## Learning-Curve Stability Protocol

Training scripts support a fixed-eval monitor that is less noisy than epoch train accuracy:

- monitor metric can be `eval_seq_acc_lcb` (recommended), the lower 95% confidence bound of fixed eval accuracy
- scripts log `eval_seq_acc`, `eval_seq_acc_ci95`, and `eval_seq_acc_lcb` per epoch
- early stopping and best-metric tracking can use the lower bound directly to avoid overreacting to random wobble

Recommended monitor config:

```yaml
train:
  monitor:
    metric: eval_seq_acc_lcb
    metric_ema_beta: 0.6
    eval_batches: 8
    mode: max
```

## Current scope

Implemented:

- synthetic hidden-mechanism sequence environment
- structured solver with encoder, approach proposer, rule proposer, probe policy, belief updater, and solver head
- recurrent and trivial baselines
- training, evaluation, and inspection scripts
- basic tests and experiment configs

Current main known weakness:

- premature belief collapse, especially early entropy collapse over approaches and rules

## Loss design notes

Current training now uses explicit, mathematically separated objectives:

- `task_loss`: final target cross-entropy
- `probe_loss`: diagnostic probe supervision (when available)
- `rule_consistency_loss`: MSE between rule-predicted probe outcomes and observed probe outcomes in latent space
- `probe_ig_loss`: KL from policy distribution to detached approach-disagreement target distribution
- `approach_diversity_loss`: squared off-diagonal cosine similarity over approach slots (orthogonality pressure)
- `rule_diversity_loss`: squared off-diagonal cosine similarity over rule slots (orthogonality pressure)

Auxiliary losses (`rule_consistency_loss`, `probe_ig_loss`) are balanced by:

- scalar weights in config (`train.rule_consistency_weight`, `train.probe_ig_weight`)
- EMA normalization in trainer (`train.aux_loss_ema_decay`, `train.aux_loss_ema_eps`)
- optional warmup/ramp (`train.aligned_warmup_steps`, `train.aligned_ramp_steps`)

This keeps auxiliary terms comparable and prevents one auxiliary loss scale from overwhelming task learning.

Additional controllable balancing terms:

- `train.approach_diversity_weight`
- `train.rule_diversity_weight`

Entropy control supports two modes when target schedules are set:

- `train.entropy_target_mode: floor`
- keeps entropy above a floor target; avoids premature collapse but does not force commitment
- `train.entropy_target_mode: track`
- penalizes distance to the target; supports high uncertainty early and lower uncertainty later

For stage-1 curriculum runs we currently recommend `track`.

`run_ablation_suite.py` now reports `structured_vs_baselines` with:

- seed-level deltas (mean/std) for structured vs each baseline
- bootstrap 95% CIs over per-batch deltas
- `p_gt_zero` and `p_lt_zero` from bootstrap samples (directional confidence, not a formal hypothesis test)

New learnability-alignment objective:

- `train.demo_recon_weight`: auxiliary CE on reconstructing demo outputs from demo inputs
- Optional schedule:
- `train.demo_recon_weight_start`
- `train.demo_recon_weight_end`
- `train.demo_recon_anneal_steps`

This is intended to force latent beliefs to encode mechanism information from demonstrations directly.

## Task-shape controls

`HiddenMechanismSequenceEnv` now supports:

- `env.local_use_left_context` (default: `true`)

When set to `false`, `LOCAL` mechanism becomes per-token shift-only (`y_t = x_t + shift` mod vocab),
which is useful for a learnable bridge sanity stage before reintroducing harder local context coupling.

## Recommended first experiments

1. Overfit on `debug.yaml` just to validate the loop.
2. Train the recurrent baseline on `small.yaml`.
3. Train the structured solver on `small.yaml`.
4. Compare sequence accuracy, entropy behavior, probe usefulness, and per-mechanism performance.

## Project docs

- `RESEARCH_SPEC.md`: experiment rationale and success criteria
- `COPILOT_INSTRUCTIONS.md`: architecture and implementation guidance
- `AGENTS.md`: expected workflow and experiment commands
