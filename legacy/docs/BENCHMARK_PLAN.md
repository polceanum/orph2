# Benchmark Plan (Mac-First, Literature-Aligned)

## Goals

- Preserve current bridge experiments for continuity.
- Add at least one benchmark family used in published OOD/generalization RL work.
- Keep comparisons scientifically controlled and reproducible on a Mac.

## Sources (Primary)

- Procgen benchmark overview: https://openai.com/index/procgen-benchmark/
- Procgen package and paper citation: https://pypi.org/project/procgen/0.10.6/
- Meta-World docs: https://metaworld.farama.org/
- Minigrid docs (package naming/install context): https://minigrid.farama.org/v2.2.0/release_notes/
- Statistical evaluation guidance (RLiable paper): https://openreview.net/forum?id=uqv8-U4lKBe

## Candidate Selection (as of 2026-04-04)

1. Minigrid (recommended first)
- Why now:
  - Lightweight and practical on Mac CPU.
  - Widely used in generalization/meta-RL toy-to-mid-scale experiments.
  - Easy to define explicit IID/OOD splits by environment size/layout family.
- Risk:
  - Less standardized "single leaderboard" than Procgen.

2. Procgen (recommended second)
- Why:
  - Strong literature visibility for generalization.
  - Explicit train/test level split protocol.
- Risk on Mac:
  - Dependency friction is higher than Minigrid; may require stack isolation.

3. Meta-World (recommended third)
- Why:
  - Standard benchmark for multi-task/meta-RL.
- Risk on Mac:
  - MuJoCo dependency + robotics setup increases setup/runtime complexity.

## Decision

- Start integration with Minigrid now.
- Keep Procgen as next standard benchmark once Minigrid protocol is stable.
- Add Meta-World only after we have stable benchmark automation.

## Required Comparators (Every Benchmark)

- `structured` (our method)
- `recurrent_rl` baseline
- `standard_actor_critic` baseline
- `random` reference
- `oracle` reference (same class trained directly on target OOD split)

## Required Metrics

- IID performance
- OOD performance
- `structured - recurrent_rl` delta
- `structured - standard_actor_critic` delta
- `structured - random` gap
- Oracle gap and transfer ratio (`structured / oracle`)
- Efficiency: performance per million trainable parameters

## Seed Policy

- Fast iteration: 1 seed
- Candidate acceptance: 3 seeds
- Main claims: 5 seeds

## Fairness Rules

- Same env split and seed set across methods.
- Same training budget unless explicitly declared otherwise.
- No OOD/test-informed tuning.
- Report confidence intervals for final comparisons.

## Current Integration Status

- Added Minigrid smoke config:
  - `configs/benchmarks/minigrid_door_key_smoke.yaml`
- Added Minigrid smoke evaluator (random-policy baseline):
  - `scripts/benchmarks/minigrid_smoke_eval.py`
- Added optional benchmark deps:
  - `requirements-benchmarks.txt`
- Added SB3 PPO benchmark trainer/evaluator:
  - `scripts/benchmarks/minigrid_sb3_ppo.py`

## Next Steps

1. Install benchmark deps in `orpheus`:
   - `conda run -n orpheus python -m pip install -r requirements-benchmarks.txt`
2. Run smoke benchmark:
   - `conda run -n orpheus python scripts/benchmarks/minigrid_smoke_eval.py --config configs/benchmarks/minigrid_door_key_smoke.yaml --out artifacts/benchmarks/minigrid_smoke_random_s0.json`
3. Use `Empty-Random` as the first non-trivial Minigrid OOD benchmark tier:
   - train: `MiniGrid-Empty-Random-5x5-v0`
   - test/OOD: `MiniGrid-Empty-Random-6x6-v0`
4. Add train/eval harness for `structured` and in-repo baselines on the same Minigrid split.
5. Keep DoorKey/Crossing as higher-difficulty benchmarks with larger budgets.
