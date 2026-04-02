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

## Current scope

Implemented:

- synthetic hidden-mechanism sequence environment
- structured solver with encoder, approach proposer, rule proposer, probe policy, belief updater, and solver head
- recurrent and trivial baselines
- training, evaluation, and inspection scripts
- basic tests and experiment configs

Current main known weakness:

- premature belief collapse, especially early entropy collapse over approaches and rules

## Recommended first experiments

1. Overfit on `debug.yaml` just to validate the loop.
2. Train the recurrent baseline on `small.yaml`.
3. Train the structured solver on `small.yaml`.
4. Compare sequence accuracy, entropy behavior, probe usefulness, and per-mechanism performance.

## Project docs

- `RESEARCH_SPEC.md`: experiment rationale and success criteria
- `COPILOT_INSTRUCTIONS.md`: architecture and implementation guidance
- `AGENTS.md`: expected workflow and experiment commands
