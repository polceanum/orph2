# Copilot Instructions

This repository is intentionally narrowed to an RL-first OOD adaptation track.

## Preserve

- Explicit structured reasoning state (approaches/rules/belief updates).
- Comparable recurrent RL baseline.
- Explicit IID/OOD split evaluation.

## Prioritize

- OOD accuracy delta: `structured - recurrent_rl`.
- Stable multi-seed evidence.
- Fast failure detection via live JSONL logs.

## Avoid

- Re-introducing broad legacy experiment paths.
- Implicit OOD claims without explicit `eval.ood` shift.
