# Experiment Log

## Iteration 1A

Question:
- Does increasing probe-policy entropy regularization (`rl.entropy_weight`) improve OOD adaptation?

Hypothesis:
- Raising `entropy_weight` from `0.01` to `0.03` will improve OOD by keeping probe exploration less collapsed early.

Controls:
- Same architecture, optimizer, seeds (`0,1,2`), steps, and evaluation.
- Single changed variable: `rl.entropy_weight`.

Runs:
- Baseline (`entropy_weight=0.01`):
  - config: `configs/bridges/bridge_10_mechanism_addition_rl_quick.yaml`
  - output: `artifacts/rl/iter1_baseline_quick_s012.json`
- Variant (`entropy_weight=0.03`):
  - config: `configs/bridges/bridge_10_mechanism_addition_rl_quick_entropy003.yaml`
  - output: `artifacts/rl/iter1_entropy003_quick_s012.json`

Result:
- Baseline (`entropy=0.01`):
  - structured OOD: `0.1963`
  - recurrent OOD: `0.1183`
  - OOD delta (structured - recurrent): `+0.0780`
- Variant (`entropy=0.03`):
  - structured OOD: `0.1950`
  - recurrent OOD: `0.1148`
  - OOD delta (structured - recurrent): `+0.0802`

Interpretation:
- OOD delta improved slightly, but structured OOD itself did not improve.
- Most of the gain came from lower recurrent baseline performance.

Decision:
- Keep baseline entropy (`0.01`) as default.
- Do not adopt `0.03` based on this evidence.

## Iteration 1B

Question:
- Does increasing stochastic probe sampling temperature (`rl.sample_probe_temp`) improve OOD adaptation?

Hypothesis:
- Raising probe temperature from `1.0` to `1.2` may improve exploration and OOD adaptation.

Controls:
- Same architecture, optimizer, seeds (`0,1,2`), steps, and evaluation.
- Single changed variable: `rl.sample_probe_temp`.

Runs:
- Baseline:
  - config: `configs/bridges/bridge_10_mechanism_addition_rl_quick.yaml`
  - output: `artifacts/rl/iter1_baseline_quick_s012.json`
- Variant (`temp=1.2`):
  - config: `configs/bridges/bridge_10_mechanism_addition_rl_quick_temp12.yaml`
  - output: `artifacts/rl/iter2_temp12_quick_s012.json`

Result:
- Baseline:
  - structured OOD: `0.1963`
  - OOD delta: `+0.0780`
- Variant (`temp=1.2`):
  - structured OOD: `0.1923`
  - OOD delta: `+0.0773`

Decision:
- Reject `temp=1.2` for now.
- Keep `sample_probe_temp=1.0`.
