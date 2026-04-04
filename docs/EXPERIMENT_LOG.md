# Experiment Log

## Research Log Protocol (Required)

For every meaningful change or run, append one entry with:
- Question
- Hypothesis
- Controls (what stayed fixed)
- Runs (config + output artifact paths)
- Result (IID/OOD and key deltas)
- Interpretation (why it likely happened)
- Decision (`keep`, `reject`, or `needs more evidence`)
- Next step

Scientific rules:
- Log negative results and failed ideas (not only wins).
- Mark 1-seed results as exploratory only.
- Require 3 seeds for candidate acceptance and 5 seeds for stronger claims.
- Include random-chance and oracle context when available.
- If oracle quality gate fails, explicitly mark oracle as underfit and avoid transfer-ratio claims.

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

## Iteration 1C

Question:
- Does increasing policy-gradient weight (`rl.pg_weight`) improve OOD adaptation?

Hypothesis:
- Raising `pg_weight` from `1.0` to `1.5` may help probe-policy learning and improve OOD.

Controls:
- Same architecture, optimizer, seeds (`0,1,2`), steps, and evaluation.
- Single changed variable: `rl.pg_weight`.

Runs:
- Baseline:
  - config: `configs/bridges/bridge_10_mechanism_addition_rl_quick.yaml`
  - output: `artifacts/rl/iter1_baseline_quick_s012.json`
- Variant (`pg_weight=1.5`):
  - config: `configs/bridges/bridge_10_mechanism_addition_rl_quick_pg15.yaml`
  - output: `artifacts/rl/iter3_pg15_quick_s012.json`

Result:
- Baseline:
  - structured OOD: `0.1963`
  - recurrent OOD: `0.1183`
  - OOD delta: `+0.0780`
- Variant (`pg=1.5`):
  - structured OOD: `0.1974` (slightly higher)
  - recurrent OOD: `0.1215` (higher)
  - OOD delta: `+0.0758` (lower)

Decision:
- Reject `pg_weight=1.5` for now.
- Keep `pg_weight=1.0`.

## Iteration 1D

Question:
- Does increasing structured rule-slot capacity (`model.num_rules`) improve OOD adaptation?

Hypothesis:
- Increasing `num_rules` from `8` to `12` might improve structured OOD by supporting richer latent rule decomposition.

Controls:
- Same architecture/training/eval/seeds except `model.num_rules`.

Runs:
- Baseline:
  - config: `configs/bridges/bridge_10_mechanism_addition_rl_quick.yaml`
  - output: `artifacts/rl/iter1_baseline_quick_s012.json`
- Variant (`num_rules=12`):
  - config: `configs/bridges/bridge_10_mechanism_addition_rl_quick_rules12.yaml`
  - output: `artifacts/rl/iter4_rules12_quick_s012.json`

Result:
- Baseline:
  - structured OOD: `0.1963`
  - OOD delta: `+0.0780`
- Variant (`rules=12`):
  - structured OOD: `0.1691`
  - OOD delta: `+0.0513`
  - much higher seed variance

Decision:
- Reject `num_rules=12`.
- Keep baseline `num_rules=8` and test smaller capacity next (`num_rules=6`).

## Iteration 1E

Question:
- Does reducing rule-slot capacity (`num_rules=6`) improve OOD adaptation via regularization?

Hypothesis:
- Fewer rules may reduce overfitting and improve OOD.

Controls:
- Same setup as baseline, only `model.num_rules: 8 -> 6`.

Runs:
- Baseline: `artifacts/rl/iter1_baseline_quick_s012.json`
- Variant: `artifacts/rl/iter5_rules6_quick_s012.json`

Result:
- Baseline OOD delta: `+0.0780`
- `rules6` OOD delta: `+0.0744`
- Structured OOD dropped: `0.1963 -> 0.1893`

Decision:
- Reject `num_rules=6`.
- Keep `num_rules=8`.

## Iteration 1F

Question:
- Does increasing approach-slot capacity (`num_approaches=6`) improve OOD adaptation?

Hypothesis:
- More approach slots may help retain competing hypotheses and improve OOD.

Controls:
- Same setup as baseline, only `model.num_approaches: 4 -> 6`.

Runs:
- Baseline: `artifacts/rl/iter1_baseline_quick_s012.json`
- Variant: `artifacts/rl/iter6_approaches6_quick_s012.json`

Result:
- Baseline:
  - structured OOD: `0.1963`
  - OOD delta: `+0.0780`
- `approaches6`:
  - structured OOD: `0.1883`
  - OOD delta: `+0.0705`
  - higher variance than baseline

Decision:
- Reject `num_approaches=6`.
- Keep `num_approaches=4`.

## Iteration 1G

Question:
- Does increasing router expert count (`num_mechanism_experts=4`) improve OOD adaptation?

Hypothesis:
- More mechanism experts may improve structured specialization and OOD transfer.

Controls:
- Same setup as baseline, only `model.num_mechanism_experts: 3 -> 4`.

Runs:
- Baseline: `artifacts/rl/iter1_baseline_quick_s012.json`
- Variant: `artifacts/rl/iter8_experts4_quick_s012.json`

Result:
- Baseline:
  - structured OOD: `0.1963`
  - OOD delta: `+0.0780`
- `experts4`:
  - structured OOD: `0.2058`
  - OOD delta: `+0.0799`

Decision:
- Keep as promising candidate.
- Next step: verify on full-budget run (not only quick setting).

## Iteration 1H

Question:
- Does `experts=4` hold under full-budget training?

Runs:
- Baseline full: `artifacts/rl/bridge10_rl_full_s012.json`
- Variant full (`experts=4`): `artifacts/rl/iter8_experts4_full_s012.json`

Result:
- Baseline full:
  - structured OOD: `0.4672`
  - OOD delta: `+0.2814`
- `experts=4` full:
  - structured OOD: `0.4761` (`+0.0089` absolute)
  - OOD delta: `+0.2799` (`-0.0014`)

Decision:
- Keep as *borderline/promising* (absolute structured OOD improved).
- Continue local router search with `experts=4` and sharper temperature.

## Iteration 1I

Question:
- Does actor-critic training (learned value baseline) improve OOD adaptation over plain REINFORCE baseline?

Field prior:
- Actor-critic is a standard variance-reduction upgrade in policy-gradient RL and often improves sample efficiency/stability.

Controls:
- Same quick setup/seeds as baseline, with one algorithmic change:
  - `rl.use_critic: true`
  - `rl.value_loss_weight: 0.5`

Runs:
- Baseline quick: `artifacts/rl/iter1_baseline_quick_s012.json`
- Actor-critic quick: `artifacts/rl/iter10_actorcritic_quick_s012.json`

Result:
- Baseline quick:
  - structured OOD: `0.1963`
  - recurrent OOD: `0.1183`
  - OOD delta: `+0.0780`
- Actor-critic quick:
  - structured OOD: `0.1986`
  - recurrent OOD: `0.1143`
  - OOD delta: `+0.0843`

Decision:
- Keep as promising candidate.
- Next step: full-budget confirmation.

## Iteration 1J

Question:
- Does combining actor-critic with `experts=4` improve over baseline quick?

Runs:
- Baseline quick: `artifacts/rl/iter11b_baseline_quick_s012.json`
- Actor-critic + experts=4 quick: `artifacts/rl/iter12_actorcritic_experts4_quick_s012.json`

Result:
- Baseline quick:
  - structured OOD: `0.1963`
  - OOD delta: `+0.0780`
- Actor-critic + experts=4 quick:
  - structured OOD: `0.1875`
  - OOD delta: `+0.0569`

Decision:
- Reject the combination for now.
- Keep `experts=4` and actor-critic as separate hypotheses (do not combine yet).

## Iteration 1K

Question:
- Does plain actor-critic remain beneficial on the current code after stability fixes?

Runs:
- Baseline quick: `artifacts/rl/iter11b_baseline_quick_s012.json`
- Actor-critic quick: `artifacts/rl/iter12b_actorcritic_quick_s012.json`

Result:
- Baseline quick:
  - structured OOD: `0.1963`
  - recurrent OOD: `0.1183`
  - OOD delta: `+0.0780`
- Actor-critic quick:
  - structured OOD: `0.2011`
  - recurrent OOD: `0.1153`
  - OOD delta: `+0.0857`

Decision:
- Keep actor-critic as current leading candidate in quick setting.
- Next step: confirm with full-budget actor-critic rerun.

## Iteration 1L

Question:
- Does actor-critic improve at full budget on the updated/stable code path?

Runs:
- Baseline full: `artifacts/rl/bridge10_rl_full_s012.json`
- Actor-critic full (rerun): `artifacts/rl/iter13_actorcritic_full_s012.json`

Result:
- Baseline full:
  - structured OOD: `0.4672`
  - recurrent OOD: `0.1858`
  - OOD delta: `+0.2814`
- Actor-critic full:
  - structured OOD: `0.4596`
  - recurrent OOD: `0.1607`
  - OOD delta: `+0.2988`

Interpretation:
- Actor-critic improved relative OOD margin (`+0.0174` delta), mainly by reducing recurrent baseline performance.
- Absolute structured OOD remained below baseline full (`-0.0077`).

Decision:
- Keep actor-critic as useful for stronger separation/margin analysis.
- Do not call it an absolute structured-OOD win at full budget yet.

## Iteration 1M

Question:
- Does adding a rule-critic auxiliary loss (rules as latent value estimators) improve OOD?

Runs:
- Actor-critic quick: `artifacts/rl/iter12b_actorcritic_quick_s012.json`
- Actor-critic + rule-critic quick: `artifacts/rl/iter14_actorcritic_rulecritic_quick_s012.json`

Result:
- Actor-critic quick:
  - structured OOD: `0.2011`
  - OOD delta: `+0.0857`
- Actor-critic + rule-critic quick:
  - structured OOD: `0.1820`
  - OOD delta: `+0.0561`

Decision:
- Reject rule-critic in current form.
- Discard code path to keep framework clean.

## Iteration 1N

Question:
- Does PPO-style clipped policy gradient (`pg_clip_eps=0.2`) improve actor-critic stability/OOD?

Runs:
- Actor-critic quick: `artifacts/rl/iter12b_actorcritic_quick_s012.json`
- Actor-critic + clip quick: `artifacts/rl/iter15_actorcritic_pgclip02_quick_s012.json`

Result:
- Actor-critic quick:
  - structured OOD: `0.2011`
  - OOD delta: `+0.0857`
- +clip quick:
  - structured OOD: `0.1969`
  - OOD delta: `+0.0729`

Decision:
- Reject clipped PG in this setup.

## Iteration 1O

Question:
- Does entropy annealing (high to low entropy weight) improve actor-critic OOD?

Runs:
- Actor-critic quick: `artifacts/rl/iter12b_actorcritic_quick_s012.json`
- Actor-critic + entropy anneal quick: `artifacts/rl/iter16_actorcritic_entropyanneal_quick_s012.json`

Result:
- Actor-critic quick:
  - structured OOD: `0.2011`
  - OOD delta: `+0.0857`
- +entropy anneal quick:
  - structured OOD: `0.1956`
  - OOD delta: `+0.0767`

Decision:
- Reject entropy annealing in this form.

## Iteration 1P

Question:
- Does reducing critic loss weight (`value_loss_weight: 0.5 -> 0.2`) improve actor-critic OOD?

Runs:
- Actor-critic quick (`vloss=0.5`): `artifacts/rl/iter12b_actorcritic_quick_s012.json`
- Actor-critic quick (`vloss=0.2`): `artifacts/rl/iter17_actorcritic_vloss02_quick_s012.json`

Result:
- `vloss=0.5`:
  - structured OOD: `0.2011`
  - OOD delta: `+0.0857`
- `vloss=0.2`:
  - structured OOD: `0.2029`
  - OOD delta: `+0.0881`

Decision:
- Keep `value_loss_weight=0.2` as current best quick setting.

## Iteration 1Q

Question:
- Does combining `vloss=0.2` with `experts=4` help?

Runs:
- `vloss=0.2` quick: `artifacts/rl/iter17_actorcritic_vloss02_quick_s012.json`
- `vloss=0.2` + experts4 quick: `artifacts/rl/iter18_actorcritic_vloss02_experts4_quick_s012.json`

Result:
- `vloss=0.2` quick:
  - structured OOD: `0.2029`
  - OOD delta: `+0.0881`
- +experts4:
  - structured OOD: `0.1775`
  - OOD delta: `+0.0547`

Decision:
- Reject experts4 combination.

## Iteration 1R

Question:
- Does `vloss=0.2` improve full-budget results?

Runs:
- Baseline full: `artifacts/rl/bridge10_rl_full_s012.json`
- Actor-critic full (`vloss=0.5`): `artifacts/rl/iter13_actorcritic_full_s012.json`
- Actor-critic full (`vloss=0.2`): `artifacts/rl/iter19_actorcritic_vloss02_full_s012.json`

Result:
- Baseline full:
  - structured OOD: `0.4672`
  - OOD delta: `+0.2814`
- Actor-critic `vloss=0.5`:
  - structured OOD: `0.4596`
  - OOD delta: `+0.2988`
- Actor-critic `vloss=0.2`:
  - structured OOD: `0.4767` (best absolute among these)
  - OOD delta: `+0.2776`

Decision:
- Keep `vloss=0.2` as current best absolute structured-OOD full setting.

## Iteration 1S

Question:
- Does disabling advantage normalization help `vloss=0.2`?

Runs:
- Quick (`nonorm`): `artifacts/rl/iter20_actorcritic_vloss02_nonorm_quick_s012.json`
- Full (`nonorm`): `artifacts/rl/iter21_actorcritic_vloss02_nonorm_full_s012.json`
- Comparator (`norm`, full): `artifacts/rl/iter19_actorcritic_vloss02_full_s012.json`

Result:
- Quick: slight OOD delta gain (`+0.0898`).
- Full:
  - `norm` structured OOD: `0.4767`
  - `nonorm` structured OOD: `0.4619`
  - `norm` OOD delta: `+0.2776`
  - `nonorm` OOD delta: `+0.2734`

Decision:
- Keep normalization enabled for full-budget runs.

## Iteration 1T

Question:
- Does more training budget (with early stop) improve the current best `vloss=0.2` setting?

Runs:
- Full (`12x120`): `artifacts/rl/iter19_actorcritic_vloss02_full_s012.json`
- Long (`16x120`, patience=3): `artifacts/rl/iter22_actorcritic_vloss02_long_s012.json`

Result:
- `iter19`:
  - structured OOD: `0.4767`
  - OOD delta: `+0.2776`
- `iter22`:
  - structured OOD: `0.5208`
  - OOD delta: `+0.3083`

Decision:
- Keep long-budget `vloss=0.2` as new best overall.

## Iteration 1U

Question:
- Can we compare against an explicit mainstream baseline (`standard_actor_critic`) in the same training/eval loop?

Implementation:
- Added optional `baselines.standard_actor_critic` path in `scripts/train_rl_compare.py`.
- Uses a non-structured recurrent actor-critic with shared policy/value training objective and dedicated reporting fields.

Quick run:
- Config: `configs/bridges/bridge_10_mechanism_addition_rl_quick_with_standard_ac.yaml`
- Output: `artifacts/rl/iter27_quick_with_standard_ac_s012.json`

Result (quick):
- structured OOD: `0.1950`
- recurrent OOD: `0.1136`
- standard actor-critic OOD: `0.1353`
- structured - standard AC OOD delta: `+0.0597`

Decision:
- Keep standard actor-critic baseline as default reporting comparator.

## Iteration 1V

Question:
- Does structured still beat standard actor-critic on a harder OOD split (unseen stochastic mechanism only)?

Quick run:
- Config: `configs/bridges/bridge_10_mechanism_stochastic_ood_quick_with_standard_ac.yaml`
- Output: `artifacts/rl/iter29_stochasticood_quick_with_standard_ac_s012.json`

Result (hard OOD, quick):
- structured OOD: `0.3158`
- recurrent OOD: `0.1646`
- standard actor-critic OOD: `0.2658`
- structured - standard AC OOD delta: `+0.0499`

Interpretation:
- Gap narrows on harder OOD, but structured still leads both baselines.

Decision:
- Proceed to full-budget hard-OOD run with standard baseline included.

## Iteration 2A

Question:
- Does the stronger long-budget setup with explicit standard actor-critic baseline yield robust 5-seed OOD gains?

Runs:
- Config: `configs/bridges/bridge_10_mechanism_addition_rl_long20_with_standard_ac.yaml`
- Output: `artifacts/rl/iter41_long20_with_standard_ac_s0to4_combined.json`

Result (5 seeds):
- structured OOD: `0.5836`
- recurrent OOD: `0.2090`
- standard actor-critic OOD: `0.2106`
- structured - recurrent OOD: `+0.3745`
- structured - standard AC OOD: `+0.3730`

Interpretation:
- Structured is clearly above both baselines at this budget.

Decision:
- Keep as strong baseline checkpoint for subsequent iterations.

## Iteration 2B

Question:
- Is oracle training sufficiently strong to serve as a valid transfer-ratio reference?

Runs:
- Config: `configs/bridges/bridge_10_mechanism_addition_rl_long20_oracle_generated_strong_with_standard_ac.yaml`
- Output: `artifacts/rl/iter42_oracle_long20_strong_s012.json`
- Gate summary: `artifacts/rl/summary_iter46_s012_vs_iter42_oracle.json`

Result:
- Oracle structured OOD: `0.4675` (below primary structured OOD checkpoints)
- Oracle recurrent OOD: `0.2033`
- Oracle standard AC OOD: `0.2001`

Interpretation:
- Oracle appears underfit in this configuration, so it is not a reliable upper-bound reference.

Decision:
- Mark oracle as underfit; do not rely on transfer-ratio claims from this oracle.

## Iteration 2C

Question:
- Does longer training (`long24`) further improve OOD beyond `long20`?

Runs:
- Seed 0: `artifacts/rl/iter43_long24_with_standard_ac_s0.json`
- Seeds 1,2: `artifacts/rl/iter44_long24_with_standard_ac_s12.json`
- Seeds 3,4: `artifacts/rl/iter47_long24_with_standard_ac_s34.json`
- Combined 0..4: `artifacts/rl/iter48_long24_with_standard_ac_s0to4_combined.json`
- Summary: `artifacts/rl/summary_iter48_s0to4_vs_iter42_oracle.json`

Result (5 seeds, long24):
- structured OOD: `0.6268`
- recurrent OOD: `0.2136`
- standard actor-critic OOD: `0.2078`
- structured - recurrent OOD: `+0.4132`
- structured - standard AC OOD: `+0.4191`
- vs long20 (`iter41`): structured OOD `+0.0433`

Interpretation:
- Longer budget improved structured OOD materially while baseline OOD remained near `~0.21`.
- Relative structured advantage increased vs both baselines.

Decision:
- Keep `long24` as the current primary setting.

## Iteration 2D

Question:
- Can a stronger oracle on the same long24 regime restore a valid oracle upper bound?

Runs:
- Generated config: `configs/bridges/bridge_10_mechanism_addition_rl_long24_oracle_generated_strong_with_standard_ac.yaml`
- In-progress run: `artifacts/rl/iter49_oracle_long24_strong_s012.json`
- Live log: `artifacts/rl/iter49_oracle_long24_strong_s012.log.jsonl`

Current status:
- Run started and logging normally; not yet finalized in this log entry.

Decision:
- Pending completion; update this entry with final metrics and oracle-gate conclusion.
