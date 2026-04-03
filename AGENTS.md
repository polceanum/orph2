# AGENTS.md

See `RESEARCH_SPEC.md` for rationale and `COPILOT_INSTRUCTIONS.md` for implementation constraints.

This file defines how to run and evaluate experiments in a way that stays faithful to the research thesis.

## Mission

Build a structured reasoning system for OOD generalization.

The target is not just better accuracy. The target is better reasoning under uncertainty.

---

## Success Criteria

### Short term

- structured > recurrent on `small.yaml`
- gains visible per mechanism
- entropy not collapsing immediately

### Medium term

- recovery after surprise
- archive usefulness
- random-probe and no-rules ablations degrade behavior in expected ways

---

## Canonical Workflow

1. Debug with `debug.yaml`
2. Overfit sanity check
3. Train on `small.yaml`
4. Compare structured vs recurrent
5. Add ablations
6. Only then scale toward `train.yaml`

---

## Commands

Debug overfit:

```bash
python scripts/overfit_tiny.py --config configs/debug.yaml
```

Small structured:

```bash
python scripts/train_structured.py --config configs/small.yaml
```

Small recurrent:

```bash
python scripts/train_recurrent.py --config configs/small.yaml
```

Small eval:

```bash
python scripts/eval_baselines.py --config configs/small.yaml \
  --structured-ckpt artifacts/structured_last.pt \
  --recurrent-ckpt artifacts/recurrent_last.pt
```

Small ablations:

```bash
python scripts/train_random_probe.py --config configs/small.yaml
python scripts/train_no_rules.py --config configs/small.yaml
```

Multi-seed ablation suite:

```bash
python scripts/run_ablation_suite.py \
  --config configs/small_tuned.yaml \
  --seeds 0,1,2 \
  --tag small_tuned_suite_v1 \
  --eval-batches 30
```

Stage-1 controlled OOD curriculum (recommended first robust run):

```bash
python scripts/run_ablation_suite.py \
  --config configs/core_stage1_v3.yaml \
  --seeds 0,1,2 \
  --tag core_stage1_v3_seed012 \
  --eval-batches 30
```

Bridge benchmark (staged OOD scenarios + optional oracle references):

```bash
python scripts/run_bridge_benchmark.py \
  --benchmark-config configs/bridges/benchmark.yaml \
  --seeds 0 \
  --eval-batches 20 \
  --tag-prefix bridge_quick \
  --with-oracle
```

Fast iteration variant (one seed):

```bash
python scripts/run_ablation_suite.py \
  --config configs/core_stage1_v3.yaml \
  --seeds 0 \
  --tag core_stage1_v3_seed0_iter \
  --eval-batches 20
```

OOD split evaluation (IID vs OOD shift):

```bash
python scripts/eval_ood_baselines.py \
  --config configs/small_ood_paramshift.yaml \
  --structured-ckpt artifacts/structured_last.pt \
  --random-probe-ckpt artifacts/random_probe_last.pt \
  --no-rules-ckpt artifacts/no_rules_last.pt \
  --recurrent-ckpt artifacts/recurrent_last.pt \
  --num-batches 40
```

Larger train:

```bash
python scripts/train_structured.py --config configs/train.yaml
python scripts/train_recurrent.py --config configs/train.yaml
```

---

## Experiment Rules

- run from the `orpheus` conda environment (base may crash on Torch import)
- do not collapse the architecture into a single latent except in baselines
- do not trust debug results as research evidence
- always compare against baselines
- keep experiments controlled
- inspect entropy, not just accuracy
- prefer per-mechanism interpretation over aggregate-only reporting

---

## What To Watch During Runs

Especially monitor:

- sequence accuracy
- per-mechanism accuracy
- approach entropy
- rule entropy
- surprise behavior
- whether the structured model behaves differently from recurrent

Warning signs:

- entropy collapses immediately
- structured barely differs from recurrent internally
- probe-related metrics do not move
- gains vanish when evaluation is held out
- IID and OOD results are nearly identical because no actual distribution shift was configured

Interpretation helper:

- use one seed for fast iteration
- use 3 seeds before deciding a change is real
- for OOD claims, prefer `structured_vs_baselines.*.ood_seq_acc_delta.batch_bootstrap_95ci`
  and `structured_vs_baselines.*.ood_gap_seq_acc_delta.batch_bootstrap_95ci`

---

## Current Priority Questions

1. Does structure help on `small.yaml`?
2. Do learned probes matter?
3. Do rules matter?
4. Does archive or recovery state matter?
5. Can the system remain uncertain long enough to disambiguate hidden mechanisms?

---

## Interpretation Rules

Promising:

- structured beats recurrent with visible mechanism-level gains
- entropy remains meaningfully non-collapsed early in the episode
- ablations hurt in interpretable ways

Not sufficient:

- better debug performance
- tiny overfit success
- small aggregate gains without evidence that the structured components matter
