# Fair Ridge versus LightGBM comparison protocol

## Status and research question

**PREPARATION VALIDATED; HISTORICAL TUNING AND PRIMARY EVALUATION NOT EXECUTED.**
This protocol asks whether model error differences persist under a common
information set, preprocessing, chronological selection schedule and paired
statistical procedure. It does not predict which model will win. A null,
negative, seasonal or horizon-dependent finding is a valid outcome.

The initial local/remote commit was
`51553444d4ae01ae433b6d9ac2fd7e80eb1c9bed`, branch `main`, with no divergence
or uncommitted work after a safe fetch. No equivalent framework existed.
Priority 1 remains completed exploratory research. Only a deterministic
synthetic fixture and software checks are executed in this preparation phase.
No original dataset, result, model API or published numerical table is changed.

## Original benchmark audit

Evidence: `results/forecasting_baselines/20261012T093203Z/run.json`, the saved
parquets, `models/tabular.py`, `features/build.py`, `models/base.py`,
`evaluation/backtest.py`, `experiments/zoo.py` and the forecasting report.
The snapshot is retrospective EIA-930 TEX demand, version
`eia930-TEX+openmeteo@2026-10-10`, 97,914 hourly observations. It includes 72
causally repaired demand values. Evaluation excludes their targets.

| Item | Original Ridge | Original LightGBM |
|---|---|---|
| Origin/target | UTC hour ending; demand through origin; target = origin + h | Same |
| Horizons | 1/24/48/168 hours | Same |
| Initial fitting | 2015-08-01 to fold `val_start`, target strictly before cutoff | 2015-08-01 to fold `cal_start`, target strictly before cutoff |
| Selection | Five penalties .1/1/10/100/1000, earlier temporal validation | Fixed settings, no point-model selection |
| Selection objective | **MAE in normalized target units**, not MW | No validation objective; training uses normalized-target L2 |
| Refitting | Refit estimator and scaler on fitting + validation masks, before `cal_start` | Fit once on fitting + validation before `cal_start` |
| Features | Demand lags through 336-hour lookback, trailing statistics/differences, target calendar, 168-hour normalization | Same feature builder, different representation |
| Representation | Fixed hour/dow/offday interactions; drops explicit month; retains annual sine/cosine | Raw calendar with hour/dow categorical; retains explicit month |
| Missing features | Drop nonfinite fitting/validation rows; fitting-period median fallback at inference | Native LightGBM missing-value handling; no explicit common finite-feature mask |
| Scaling | Fitting-only StandardScaler, refitted at final cutoff | No StandardScaler; same causal target/input normalization |
| Imputed fitting targets | Allowed if otherwise finite | Allowed if otherwise finite |
| Point settings | Ridge defaults, selected alpha | L2, 600 trees, 63 leaves, learning rate .05, minimum child 40, L1=0/L2=1 |
| Other tree settings | Not applicable | .8 row sampling each iteration and .8 feature sampling; seed 0 |
| Early stopping | None | None |
| Quantiles | None | Nine independent quantile fits per horizon, 250 trees/31 leaves, learning rate .08 |

Exact original half-open UTC boundaries:

| Fold | Validation | Calibration | Test origins |
|---|---|---|---|
| test1 | 2024-01-01 to 2024-04-01 | 2024-04-01 to 2024-07-01 | 2024-07-01 to 2025-01-01 |
| test2 | 2024-07-03 to 2024-10-02 | 2024-10-02 to 2025-01-01 | 2025-01-01 to 2025-07-01 |
| test3 | 2024-12-31 to 2025-04-01 | 2025-04-01 to 2025-07-01 | 2025-07-01 to 2026-01-01 |
| test4 | 2025-07-03 to 2025-10-02 | 2025-10-02 to 2026-01-01 | 2026-01-01 to 2026-07-01 |

Use the authoritative recorded fold dates in `run.json` and `config.TEST_FOLDS`;
the exact third-fold dates are generated from the existing 91-day definitions.
Fitting starts 2015-08-01 in all folds. The full original fitting span ends at
`cal_start`, not at test start. Forecast origins and targets must both stay
inside their test interval. Confirmed matched counts per model/horizon in
the original report are 17,492/17,400/17,304/16,824.

Original fit counts: Ridge 5 candidate fits plus 1 refit × 4 horizons × 4 folds
= **96 fits**. LightGBM 1 point + 9 quantile fits × 4 horizons × 4 folds
= **160 fits**, of which only 16 are point fits. Quantile training is not a
point-tuning opportunity. Recorded mean per-fold Ridge fit/prediction time is
3.2903/1.1028 s; LightGBM 57.7171/2.2280 s includes all quantile work. Total
baseline wall time is 273.62 s. Point-only LightGBM cost is not separately
identified by that timing record. Ridge has no stochastic sampling; LightGBM
records seed 0 and the executed benchmark requested two CPU threads.

**Confirmed fairness limitations:** unequal tuning opportunities, normalized
versus new MW selection objectives, different supplied feature representations,
and different missing-feature treatment. These prevent interpreting the
original table as a balanced hyperparameter search. Explicit month is already
related to other known calendar signals, so its omission is a representation
difference, not proof of privileged unavailable data or a causal performance
explanation. No target leakage or mismatched final keys was established in
the validated no-weather benchmark. Scalers and supervised target cutoffs were
already causal. Possible unseen-period instability and publication delays are
limitations, not confirmed faults invented to explain a particular winner.

## New chronological design

Three predeclared development validation blocks, half-open UTC:

| Block | Candidate fitting start | Candidate target cutoff | Validation origins and target end |
|---|---|---|---|
| dev1 | 2015-08-01 | 2023-07-01 | 2023-07-01 to 2023-10-01 |
| dev2 | 2015-08-01 | 2023-10-01 | 2023-10-01 to 2024-01-01 |
| dev3 | 2015-08-01 | 2024-01-01 | 2024-01-01 to 2024-04-01 |

For each candidate/horizon/block, labels must have target timestamps strictly
before the candidate fitting cutoff. Origins alone never establish eligibility.
All labels are assumed available at their hour-ending timestamps; actual
telemetry/revision receipt timestamps are not available. This is the existing
zero-delay retrospective assumption, not a validated operational data service.
Both validation origin and target must lie in that block. Earlier validation
labels may enter the next expanding fitting block once available; no later
validation label enters an earlier fit. Search input is physically truncated
at development end, including when a separately authorized full mode runs.

Pool eligible validation absolute errors across all three blocks in **MW**,
weighting by observations. RMSE is secondary and cannot change selection.
Select separately per horizon and model. Exact MAE ties choose the earlier
stable candidate ID. Parameters and the selection procedure are frozen in
`selection.json` with configuration and prediction hashes before test fitting.
No horizon, block or candidate may be dropped because it performs poorly.

Selected parameters are used unchanged in all four existing primary folds.
Final refits use the same training start and all eligible targets before each
fold's original `cal_start`. Calibration observations and current-fold test
labels never enter refitting. Prior-fold observations that precede a later
fit cutoff may enter later expanding refits. No per-fold test-based retuning
occurs. The original four test folds were inspected in previous experiments;
future results on them are **retrospective exploratory evidence**. Development
months also appeared in Priority 1; they are not a newly untouched holdout.
An independent later period requires its own locked, audited protocol.

## Equal information and preprocessing

Reuse `make_features` with no-weather `FeatureSpec`, causal demand lags and
rolling normalization. Both models receive the **same numerical column matrix**:
fixed one-hot hour, dow and offday interactions from the existing design,
plus a fixed 12-month one-hot basis, and all remaining demand/calendar columns.
There is no learned category discovery from future observations. LightGBM
uses that shared dense representation rather than privileged native-calendar
features. This deliberately changes the original representation for a
controlled comparison; it is not a reproduction of original benchmark scores.

Both models use identical fitting rows: target before cutoff, target finite
and nonimputed, and positive finite origin normalization. Repaired demand
history remains available as in the original problem, but supervised imputed
labels are excluded equally in this protocol. A fitting-only median imputer
and fitting-only StandardScaler are applied to both models. Entirely missing
training features cause explicit failure. No backward fill, future weather,
global median, test normalization fit or feature selection is allowed.
Scaling is not necessary for trees but applying the identical transform makes
preprocessing availability equal; it does not make the architectures equal.

For each candidate/block, save fitting target cutoff, latest actual fitting
target, fitting sample count, feature names, preprocessing count, iterations,
validation count, wall time, cumulative process peak memory and MAE/RMSE.
Models and preprocessing are never fit on validation/test values. Future
perturbation tests check demand features and fitting-only imputation/scaling.

## Predetermined candidate grids and search resources

Configuration: `configs/fair_model_comparison.json`.

Ridge: alpha **.1, 1, 10, 100, 1000**, in that order, IDs `r00` to `r04`.
This retains the original penalty grid as reference rather than expanding it
based on which horizons previously favored Ridge.

LightGBM: twelve candidates, generated in the fixed order profile → objective
→ leaves. Objectives **L2 and L1**; leaves **15, 31, 63**; two paired profiles:

| Profile | Learning rate | Trees | Minimum child | L1 regularization | L2 regularization |
|---|---:|---:|---:|---:|---:|
| A | .05 | 600 | 40 | 0 | 1 |
| B | .08 | 300 | 80 | 1 | 10 |

All use seed 0, CPU, at most two threads, deterministic/column-wise settings,
.8 row and feature sampling, and no validation early stopping. Profile A/L2/
63 leaves retains original fixed **point** parameters as candidate `l02`.
Both profiles apply to both objectives/leaves. This small paired design is
not a factorial identification of every parameter effect; profile components
are confounded. It supersedes the roadmap's provisional fixed-rate grid to
address the requested rate/iteration/minimum-child/regularization design,
without reference to primary-test model winners. Quantile fitting, interval
calibration, multiple seeds and more expansive search are separate experiments.

Expected counts, assuming no failures:

| Model | Candidates | Validation blocks | Horizons | Search fits | Primary refits | Total fits |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 5 | 3 | 4 | 60 | 16 | 76 |
| LightGBM | 12 | 3 | 4 | 144 | 16 | 160 |

Equal-information validation is the central fairness control. A common
**1,800 s per-model search cap** is imposed, with a 3,600 s campaign ceiling
and 2 GiB process-memory limit. This is not computational equality: model
complexity, candidate counts, actual fit time and numerical implementation
differ. If either model hits its cap, mark the campaign failed/incomplete,
retain all results and stop for approval. Do not promote a truncated search as
fairly completed or quietly give the slower model extra time. Record completed
fits and failed attempts separately; interrupted/failed attempts are not
reported as successful fits. Memory is cumulative peak RSS, not incremental
per-fit memory. One-second signal checks may wait for native calls to return;
they are monitoring guards, not a hard operating-system memory reservation.

## Key completeness and failure handling

Canonical key, **without model identity**:
`(fold, regime='none', horizon, issue UTC timestamp, target UTC timestamp)`.
Model identity is used only to reject duplicates within an estimator and to
pair the same task. All candidates use the same block/horizon schedule,
observations and imputation flags. A candidate with a missing/nonfinite
forecast, different keys or inconsistent observations is disqualified across
the complete three-block selection; its errors, partial scores and failures
remain saved. If no complete candidate exists for a model/horizon, stop. No
favorable subset can silently decide selection.

Final paired scores use the intersection of finite nonimputed keys, with
per-model, per-key accounting for missing rows, imputed targets and nonfinite
observations/predictions. Canonical-schedule values are checked even when the
other model lacks a row. Do not call common-key metrics all-schedule metrics.
The authorized runner currently requires selected models to produce complete
finite outputs; an unforeseen final fit/prediction failure stops the run.
The reusable evaluator can explicitly report partial paired schedules.

Original Ridge/LightGBM saved predictions are retained as historical reference
comparators in full mode, with separate reference metrics/accounting. Their
original APIs/results remain unchanged. They cannot be treated as a second
fair-search replication because their tuning/preprocessing differed.

## Statistical analysis plan

Report MAE, RMSE, signed mean error (`prediction - observed`), median absolute
error, 90/95/99th absolute-error quantiles; slice by horizon, fold, target
season and extreme demand. Extreme means observed target demand at or above
that fold's **95th percentile of earlier fitting-period nonimputed demand**.
This threshold is fixed before current-fold evaluation and never tuned from
test outcomes. Realized-demand slices are retrospective diagnostics, not
future-known forecast inputs.

Compute paired Ridge-minus-LightGBM absolute-error differences on common keys.
Use 500 moving-block replicates, seed 0, at **7/14/28 elapsed UTC-day** lengths,
within each fold. Daily bins retain zero-count days for missing-key gaps, so
blocks preserve elapsed calendar time rather than extending a row-count window.
Sampling strata never cross folds; pool sampled error sums/counts with valid-
observation weighting. If a fold spans fewer than a requested block length,
mark that interval unavailable instead of silently shortening the block or
omitting the fold. Report paired counts and approximate fold-day/block counts,
which are not independently established effective sample sizes.

Publish 95% percentile intervals with block-length sensitivity. Do not use
IID hourly resampling or decide the block length from a favorable interval.
Long-range dependence and overlapping paths, especially at 168 hours, may
remain inadequately captured even by 28-day blocks. Intervals condition on
fitted models and chosen parameters; they exclude tuning/revision uncertainty.
No multiplicity-adjusted or universal significance claim is predeclared.

## Preparation evidence and resource estimates

Initial disk availability was about 25 GiB; memory-pressure output reported
45% free. No packages were installed or upgraded; the project uv environment
and lockfile were reused. Four small synthetic smoke executions were retained
in append-only directories. The latest measured run is
`results/fair_model_comparison/20261013T201914026515Z/`.

It used 10,000 synthetic hours, three validation blocks, four horizons, one
Ridge alpha and one deliberately small 20-tree LightGBM configuration; 24
search fits and eight synthetic holdout refits completed in **4.080 s** with
**433,258,496 bytes peak RSS (413.19 MiB)**. Archive size is 3,396,002 bytes
(3.24 MiB), excluded from Git. There are zero missing, nonfinite or imputed
holdout keys and all twelve bootstrap horizon/block combinations completed.

| Horizon | Paired synthetic keys | Ridge MAE | LightGBM MAE |
|---|---:|---:|---:|
| 1 | 2799 | 88.7738 | 1201.8532 |
| 24 | 2776 | 88.8485 | 1204.2553 |
| 48 | 2752 | 89.2376 | 1202.7594 |
| 168 | 2632 | 85.9085 | 1200.1361 |

These fixture units imitate MW but are **not real ERCOT findings**. The
synthetic process is sinusoidal with noise, and the intentionally small tree
model is not the proposed historical search. This ranking does not support
choosing Ridge, changing the grid or estimating real accuracy. Smoke testing
validates plumbing/causality/reproducibility, not benchmark superiority.

Full-run planning estimate: **10–45 CPU wall-clock minutes**, **0.7–2 GiB**
peak RAM, **50–300 MiB** artifacts. These are uncertain assumptions, not
measurements or a guarantee of completing within the 60-minute campaign cap.
The original Priority 1 point-only LightGBM development fit took about 14 s
for four horizons (~3.5 s per fit), with original representation and 600 trees.
144 analogous candidate fits yield a roughly 504 s starting estimate. Allow
roughly 1–4× variation for dataset span, shared dense representation, candidate
complexity and system load, plus final refits/preprocessing/statistical work.
Ridge is relatively cheaper in earlier records but its sixty fits must still
be measured. Synthetic 20-tree timing is not linearly extrapolated to ten-year
600-tree training. RAM/storage estimates assume sequential model fits, one
shared-feature matrix at a time, compressed prediction tables and 32 fitted
pipelines. Actual coefficient/tree serialization sizes and resource usage
could exceed estimates; stop at limits and seek approval rather than continue.
No tiny real-data timing sample or historical tuning was executed here.

## Implementation, verification and readiness

- `evaluation/fair_comparison.py`: schema/key checks, intersection accounting,
  error metrics, fitting-only extreme thresholds and fold/day-aware bootstrap.
- `experiments/fair_model_comparison.py`: default dry run; synthetic smoke;
  separately gated development/full modes; fixed enumeration, chronological
  fitting, selection, frozen provenance, new output directories and monitoring.
- `configs/fair_model_comparison.json`: reviewed deterministic protocol.
- `tests/test_fair_model_comparison.py`: synthetic causal/preprocessing/key/
  statistics/failure/resource/authorization tests.

One initial negative-case test exposed an all-invalid-candidate concatenation
error. The runner now checks complete-candidate selection before concatenating
nonexistent predictions, retains failure tables and raises the intended
no-complete-candidate error. A later test fixture was corrected to carry the
same imputation flag as its canonical schedule; the production consistency
check was retained. A final unit test exposed equivalent hourly timestamps rejected solely because
of storage precision; UTC grid comparison now normalizes precision before
checking equality. No scientific numerical result was altered to pass tests.

Final checks: **155 tests passed** (150 non-neural in 22.27 s and five CPU
neural fixtures in 14.64 s, separate processes), including 31 new synthetic
tests. Targeted tests, Ruff, changed-file formatting, compilation, offline
lock validation, four evidence-package checks and the completed interval
evidence checker passed. SHA-256 checks preserve all 124 original data/raw/
processed, benchmark, weather/uncertainty/interval archive and published-report
files. Independent direct-formula smoke checks reproduced every metric/slice
row, confirmed all fitting-target cutoffs and preprocessing sample counts,
and verified byte-identical repeated metrics/bootstrap tables. Existing
pandas warnings remain; no dependency change or new historical run occurred.

The fit-event ledger reports 32 actual starts/completions in the final smoke
run; candidate-score tables separately preserve failures/partial attempts.
The common search cap is also checked after the last candidate, so it cannot
be exceeded unnoticed merely because no next candidate remains.

Ready means: schema/grid validation, all targeted and feasible full tests,
lint/changed-file formatting, compilation, offline lock verification, existing
research checks, preserved input/archive fingerprints, and clean reviewed
Git synchronization pass. These verify execution machinery, not real-data
outcomes or measured historical compute feasibility. Historical tuning/full
primary evaluation still require explicit human authorization. CPU-only
synthetic execution remains under five minutes and 2 GiB; no cloud, GPU/MPS,
paid API, large download, deployment or other research priority is started.

## Commands and approval boundaries

Safe preparation commands:

```sh
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m gridcast.experiments.fair_model_comparison --mode dry-run
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m gridcast.experiments.fair_model_comparison --mode smoke
```

**After separate approval only**, next historical execution command:

```sh
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m gridcast.experiments.fair_model_comparison \
  --config configs/fair_model_comparison.json --mode full \
  --authorize-historical-search --authorize-primary-evaluation
```

Do not run that command as part of preparation. The flags are explicit local
execution gates, not proof that permission was granted. Development-only
historical search also requires approval: `--mode development
--authorize-historical-search`. Importing the runner or invoking its default
mode never starts a model fit. Every execution uses a new timestamped directory
under `results/fair_model_comparison/`; existing directories are rejected.
Outputs, failures, predictions and models stay ignored. Git publication of
reviewed source/protocol is authorized; it does not authorize the experiment.
