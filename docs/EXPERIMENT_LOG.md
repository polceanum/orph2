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
- Oracle run: `artifacts/rl/iter49_oracle_long24_strong_s012.json`
- Live log: `artifacts/rl/iter49_oracle_long24_strong_s012.log.jsonl`
- Validation command:
  - `scripts/validate_oracle_quality.py --primary artifacts/rl/iter48_long24_with_standard_ac_s0to4_combined.json --oracle artifacts/rl/iter49_oracle_long24_strong_s012.json --min-margin 0.01`
- Summary:
  - `artifacts/rl/summary_iter48_s0to4_vs_iter49_oracle_long24.json`
  - `artifacts/rl/summary_iter48_s0to4_vs_iter49_oracle_long24.md`

Result:
- Oracle (long24 strong, 3 seeds):
  - structured OOD: `0.5213`
  - recurrent OOD: `0.2141`
  - standard actor-critic OOD: `0.2103`
- Primary (long24, 5 seeds):
  - structured OOD: `0.6268`
  - recurrent OOD: `0.2136`
  - standard actor-critic OOD: `0.2078`
- Oracle quality check (min margin `0.01`) failed:
  - structured margin: `-0.1055` (fail)
  - recurrent margin: `+0.0005` (fail)
  - standard actor-critic margin: `+0.0025` (fail)

Interpretation:
- Even with stronger oracle budget, oracle remains underfit relative to primary on structured OOD.
- This indicates the current oracle protocol is not a reliable upper bound and transfer-ratio claims remain invalid.

Decision:
- Mark long24 oracle as underfit.
- Keep primary long24 result as current best empirical checkpoint.
- Next iteration should focus on fixing oracle protocol/training reliability (not just adding budget).

## Iteration 2E

Question:
- Is oracle underfit caused by the prior stronger-budget recipe itself (`epochs/steps` multipliers), or does it persist with equal budget?

Runs:
- Generated equal-budget oracle config:
  - `configs/bridges/bridge_10_mechanism_addition_rl_long24_oracle_equalbudget_with_standard_ac.yaml`
- Diagnostic run (seed 0):
  - `artifacts/rl/iter50_oracle_long24_equalbudget_s0.json`
  - `artifacts/rl/iter50_oracle_long24_equalbudget_s0.log.jsonl`

Current observations (in progress):
- Early and mid-epoch curves remain much lower than primary long24 behavior.
- Structured IID is still in the low range during early training windows (roughly `~0.18` to `~0.30` in the first third of run), similar underfit pattern to prior oracle attempts.

Provisional interpretation:
- Oracle weakness likely does not come only from the stronger-budget oracle recipe.
- More targeted oracle fixes are likely needed (e.g., optimization/capacity/training dynamics), not just budget scaling.

Decision:
- Keep this as an active diagnostic run.
- If final metrics remain underfit, reject equal-budget oracle as a fix and pivot to targeted oracle optimization changes.

## Iteration 2F

Question:
- Is the observed underfit mainly due to short-budget diagnostics and distribution effects, rather than a hard architecture failure?

Runs:
- Primary short diagnostic (mechanism breakdown):
  - `artifacts/rl/iter51_diag_mechbreak_primary_s0.json`
- Oracle short diagnostic (matched capacity/budget, mechanism breakdown):
  - `artifacts/rl/iter52_diag_mechbreak_oracle_s0.json`

Result:
- Primary short diagnostic:
  - structured IID: `0.3963`, OOD: `0.3086`
- Oracle short diagnostic:
  - structured IID: `0.2075`, OOD: `0.1858`
- Per-mechanism OOD (structured):
  - Primary: mech0 `0.6402`, mech1 `0.0799`, mech2 `0.1697`
  - Oracle: mech0 `0.3676`, mech1 `0.0777`, mech2 `0.1350`

Interpretation:
- Short-budget configs are genuinely underfit.
- The hard unseen mechanism (mech1) is near-random for all variants in both settings.
- The large primary advantage in short diagnostics is driven by much stronger fit on seen/easier mechanisms.

Decision:
- Do not use short-budget diagnostics for final quality claims.
- Test a matched wider-capacity long-budget setting to isolate capacity/optimization effects.

## Iteration 2G

Question:
- Does increasing model capacity (while keeping variant comparability) remove the underfit signature and improve OOD?

Runs:
- New config:
  - `configs/bridges/bridge_10_mechanism_addition_rl_long24_with_standard_ac_width96_s0.yaml`
- Run:
  - `artifacts/rl/iter53_long24_width96_s0.json`
  - `artifacts/rl/iter53_long24_width96_s0.log.jsonl`

Result (seed 0):
- structured IID: `0.9776`
- structured OOD: `0.6890`
- recurrent IID/OOD: `0.2886 / 0.2159`
- standard AC IID/OOD: `0.2884 / 0.2378`
- structured OOD per mechanism:
  - mech0 `0.9926`, mech1 `0.0809`, mech2 `0.9460`

Interpretation:
- Underfit in prior diagnostics was not a hard task limit; this setup fits strongly.
- Capacity/training budget are first-order drivers of the underfit behavior.
- OOD remains bottlenecked by mech1 (still near-random), indicating a mechanism-specific generalization gap rather than global optimization failure.

Decision:
- Keep width-96 long24 as the new working config for iteration.
- Next: targeted improvements for mechanism-1 transfer (probing objective and train distribution), while preserving capacity parity across variants.

## Iteration 2H

Question:
- Can reducing probe entropy pressure recover oracle-width training without hurting variant comparability?

Runs:
- Config:
  - `configs/bridges/bridge_10_mechanism_addition_rl_long24_oracle_equalbudget_width96_lowent_s0.yaml`
- Run:
  - `artifacts/rl/iter55_oracle_long24_width96_lowent_s0.json`
  - `artifacts/rl/iter55_oracle_long24_width96_lowent_s0.log.jsonl`

Result (seed 0):
- vs prior oracle-width (`iter54`):
  - structured IID: `0.4429 -> 0.5263` (`+0.0833`)
  - structured OOD: `0.4733 -> 0.5241` (`+0.0508`)
- OOD per mechanism (structured):
  - mech0: `0.9334 -> 0.9952`
  - mech1: `0.0853 -> 0.0907`
  - mech2: `0.3548 -> 0.4618`

Interpretation:
- Lower entropy regularization materially improved oracle-width optimization and OOD.
- Mechanism 1 remains near-random despite overall gains, so the main bottleneck persists.

Decision:
- Keep low-entropy oracle-width setting as better than iter54.
- Continue targeted attempts for mechanism-1 improvement.

## Iteration 2I

Question:
- Does oversampling mechanism 1 during training improve mechanism-1 OOD without breaking overall performance?

Runs:
- Added optional env knob:
  - `mechanism_sampling_weights` in `ood_solver/envs/hidden_mechanism_seq.py`
  - wired in `ood_solver/envs/config.py`
- Config:
  - `configs/bridges/bridge_10_mechanism_addition_rl_long24_oracle_equalbudget_width96_lowent_mech1focus_s0.yaml`
- Run:
  - `artifacts/rl/iter56_oracle_long24_width96_lowent_mech1focus_s0.json`
  - `artifacts/rl/iter56_oracle_long24_width96_lowent_mech1focus_s0.log.jsonl`

Result (seed 0):
- structured IID/OOD: `0.4325 / 0.4510` (worse than iter55 `0.5263 / 0.5241`)
- OOD per mechanism (structured):
  - mech0: `0.9657`
  - mech1: `0.0827` (no improvement vs iter55 `0.0907`)
  - mech2: `0.2582` (large regression vs iter55 `0.4618`)

Interpretation:
- Mech1 oversampling did not improve the hard mechanism and harmed other mechanisms.
- This is likely the wrong intervention under the current objective/representation setup.

Decision:
- Discard mech1-focused sampling as a default strategy.
- Next: probe-budget/solvability diagnostic for mechanism 1 (increase probing capacity and re-measure per-mechanism behavior).

## Iteration 2J

Question:
- Is mechanism 1 currently a true bottleneck even after increasing probe budget and context?

Runs:
- Probe-budget diagnostic:
  - `artifacts/rl/iter57_probebudget_diag_width96_lowent_s0.json`
- Entropy-floor variant:
  - `artifacts/rl/iter58_probebudget_diag_width96_lowent_floor_s0.json`
- More demos:
  - `artifacts/rl/iter59_probebudget_diag_width96_lowent_demos5_s0.json`
- Mech1-only diagnostics:
  - `artifacts/rl/iter60_mech1_only_diag_width96_lowent_s0.json`
  - `artifacts/rl/iter62_mech1_anchor_valuefeat_diag_s0.json`

Result:
- Probe-budget alone (`iter57`) improved aggregate OOD vs older oracle diagnostics, but mechanism 1 remained near chance (`~0.078`).
- Entropy-floor (`iter58`) regressed vs `iter57` and was discarded.
- More demos (`iter59`) did not improve mechanism 1 (still `~0.086`, near chance).
- Mech1-only training stayed at chance:
  - `iter60` structured OOD `~0.0879`
  - `iter62` structured OOD `~0.0833`

Interpretation:
- Mechanism 1 is not currently learnable under this task/formulation and objective stack.
- This is a task-solvability bottleneck, not an OOD-only failure.

Decision:
- Quarantine mechanism 1 for now as pathological.
- Shift primary iteration to a trainable OOD split.

## Iteration 2K

Question:
- Can we establish a trainable, still-OOD benchmark where structured clearly beats strong RL baselines?

Runs:
- Trainable OOD split config:
  - `configs/bridges/bridge_10_trainable_ood_m023_width96_lowent_s0.yaml`
- Seed 0:
  - `artifacts/rl/iter63_trainable_ood_m023_width96_lowent_s0.json`

Result (seed 0):
- structured IID/OOD: `0.7146 / 0.6252`
- recurrent IID/OOD: `0.3409 / 0.3082`
- standard AC IID/OOD: `0.3613 / 0.3366`

Interpretation:
- Structured strongly outperforms both baselines on this trainable OOD split.
- Unseen mechanism 3 is still the weakest component but above baseline in aggregate.

Decision:
- Keep this as a valid working benchmark.
- Continue improving transfer with controlled changes.

## Iteration 2L

Question:
- Does increasing demonstrations (`num_demos=4`) on the trainable OOD split improve robustness and OOD margins?

Runs:
- New config:
  - `configs/bridges/bridge_10_trainable_ood_m023_width96_lowent_demos4_s0.yaml`
- Seed 0:
  - `artifacts/rl/iter64_trainable_ood_m023_width96_lowent_demos4_s0.json`
- Seeds 1,2:
  - `artifacts/rl/iter65_trainable_ood_m023_width96_lowent_demos4_s12.json`
- Combined seeds 0,1,2:
  - `artifacts/rl/iter66_trainable_ood_m023_width96_lowent_demos4_s0to2_combined.json`

Result:
- 3-seed aggregate (`iter66`):
  - structured IID/OOD: `0.9202 ± 0.0373 / 0.7186 ± 0.0243`
  - recurrent IID/OOD: `0.3368 ± 0.0160 / 0.3176 ± 0.0063`
  - standard AC IID/OOD: `0.3310 ± 0.0052 / 0.3325 ± 0.0054`
  - structured minus recurrent OOD: `+0.4010` (mean)
  - structured minus standard AC OOD: `+0.3861` (mean)

Interpretation:
- `num_demos=4` produces a robust multi-seed improvement and large OOD margins over both baselines.
- Current best trainable benchmark checkpoint.

Decision:
- Promote `trainable_ood_m023_width96_lowent_demos4` as the primary iteration target.
- Next iteration: tighten unseen mechanism-3 transfer while preserving these margins.

## Iteration 2M

Question:
- Does a non-zero entropy floor stabilize late training and improve OOD on the best trainable split?

Runs:
- Entropy-floor variant:
  - `configs/bridges/bridge_10_trainable_ood_m023_width96_lowent_demos4_floor_s0.yaml`
- Seed 0:
  - `artifacts/rl/iter67_trainable_ood_m023_width96_lowent_demos4_floor_s0.json`

Result (seed 0):
- structured IID/OOD: `0.9427 / 0.7385`
- recurrent IID/OOD: `0.3340 / 0.3040`
- standard AC IID/OOD: `0.3320 / 0.3346`
- OOD per mechanism (structured):
  - mech0: `0.9859`
  - mech2: `0.8967`
  - mech3 (unseen): `0.3589`

Interpretation:
- Entropy-floor improved over prior best seed-0 OOD (`0.7209 -> 0.7385`) while keeping large baseline gaps.
- Unseen mechanism-3 transfer remains the limiting factor, but improved vs prior seed-0 run.

Decision:
- Keep entropy-floor as a promising default for this benchmark.
- Validate with multi-seed follow-up before making it canonical.

## Iteration 2N

Question:
- Are baseline comparisons fair in capacity/efficiency terms, and can recurrent strength close the gap?

Code changes:
- `scripts/train_rl_compare.py`:
  - Added configurable recurrent baseline width: `baselines.recurrent.hidden_dim`.
  - Added `param_counts` and `efficiency` reporting (OOD/IID seq_acc per million params).
  - Added aggregate efficiency deltas in summary output.
  - Fixed two trainer issues surfaced by fairness work:
    - Safe parameter counting with lazy modules (skip uninitialized params).
    - Recurrent critic head input dimension now matches `baselines.recurrent.hidden_dim`.
- New fairness config:
  - `configs/bridges/bridge_10_trainable_ood_m023_width96_lowent_demos4_recur192_s0.yaml`

Status:
- Fairness run complete:
  - `artifacts/rl/iter68_trainable_ood_m023_width96_lowent_demos4_recur192_s0.json`
  - seed-0 structured IID/OOD: `0.9403 / 0.7405`
  - seed-0 recurrent(192) IID/OOD: `0.3481 / 0.3025`
  - seed-0 standard AC(192) IID/OOD: `0.3468 / 0.3372`
  - OOD deltas:
    - structured minus recurrent(192): `+0.4379`
    - structured minus standard AC(192): `+0.4032`
  - parameter counts (trainable):
    - structured: `1,351,273`
    - recurrent(192): `337,262`
    - standard AC(192): `337,262`

Decision:
- Keep these instrumentation and fairness fixes; they improve scientific validity and prevent silent shape errors during baseline scaling.
- Capacity-matched recurrent still does not close the gap; structured advantage appears architectural on this benchmark.
