# GridCast: controlled next-stage experiment plan

Priority 1 is **COMPLETED**, with development coverage failures and limited
exploratory benefits documented in [the interval validation report](INTERVAL_VALIDATION_REPORT.md).
Priority 2 **PREPARATION IS COMPLETED**; historical tuning and primary evaluation
are **NOT EXECUTED**. Priorities 3–5 remain **PENDING**. No GPU,
cloud job, paid API call or archive download was launched while preparing this
plan. Runtime/storage estimates are planning ranges, not measurements.

## Shared protocol and acceptance gates

Use the unchanged EIA-930 TEX snapshot and fingerprints, UTC hour-ending
index, four expanding test folds and horizons 1/24/48/168. Exclude imputed
targets; require identical `(fold, issue time, target time, horizon)` keys.
Fit transformations and choose parameters only using eligible earlier data.
Retain the no-weather feature design as the primary information regime.

Every run must save configurations, seeds, source/lock/data hashes, fitted
artifacts, convergence failures, all predictions and actual runtime. Evaluate
MAE, RMSE, stated sMAPE, fold-specific seasonal MASE, daily/weekly-naive skill,
and season/month/demand/day-night/week-part slices. Probabilistic runs add
nine-level pinball, 80/90/95% coverage, width, Winkler score, crossing and
conditional coverage. Use paired block resampling on matched keys, within
folds, with 7/14/28-day sensitivity and effective sample-count reporting.
State overlapping-target and long-range-dependence limitations.

These four historical test folds have already been inspected. New comparisons
on them are exploratory, not untouched confirmatory evidence. Freeze the
next protocol before scoring an independent later period. July–August 2026
is present in the snapshot but was not part of the primary benchmark; verify
its non-use and revisions before designating it as a secondary locked holdout.
Do not describe revised retrospective EIA records as a point-in-time telemetry
archive. No state-of-the-art claim is warranted without a directly comparable
published dataset, horizon, information set and evaluation protocol.

## Priority 1: validation-only interval tuning

**Question:** Can a predetermined rolling-window policy approach nominal
coverage without excessive width or seasonal undercoverage?

**Dataset/features:** Saved no-weather forecasts for development periods,
with demand labels and feedback timestamps. New development predictions are
needed: the completed primary test predictions must not select the window.
Use a rolling-origin development-validation interval before 2024-04-01,
for example July 2023–March 2024, with model fitting and a disjoint warm-up
calibration block entirely earlier. This precedes the first primary fold's
calibration and test periods. Record new development split boundaries explicitly.

**Training/tuning:** Keep point models fixed. Pre-register eligible-score
windows 7/14/28/60 days and minimum 168 scores, using target-time feedback.
Require warm-up long enough for the longest window plus 168-hour feedback
delay. Select the smallest mean Winkler score among candidates within a
predeclared empirical coverage tolerance (e.g. ±2 percentage points at 90%);
if none qualify, report the failure rather than relaxing the tolerance.
Freeze the selected policy before the four-fold test evaluation. Keep final
fold calibration labels out of hyperparameter selection.

**Leakage risks:** Row-count shifts; labels before target arrival; fitting a
point model on the same validation labels used to form residuals; selecting
windows from the already inspected winter test scores; confusing one nominal
coverage level with a coherent predictive distribution.

**Metrics/acceptance:** Shared probabilistic metrics, fixed matched keys and
paired Winkler/coverage comparisons. Accept implementation only after causal
feedback tests pass. Recommend a policy only if development criteria hold and
test tradeoffs are disclosed. The existing adaptive failure is a negative
result; retuning it must be a separately registered ablation.

**Compute:** Post-processing on CPU, approximately 5–20 minutes and under
1 GiB working memory for a compact development study, to be measured by a
pilot. No GPU is needed. Any new long point-model fit requires approval.

## Priority 2: fair Ridge versus LightGBM tuning

**Preparation completed; full experiment pending separate approval.**
The [October 14 local preflight](FAIR_MODEL_COMPARISON_RESOURCE_PILOT.md)
was blocked by host memory-pressure warning (dispatch level 2), with zero fits.
On October 22, 2026 the authorized Colab CPU resource pilot completed all six
fits: 68.66 s internal wall time and 657.18 MiB process high-water RSS.
Preflight was CONDITIONAL because cgroup memory was explicitly unlimited;
all hard checks passed and the process supervisor remained active.
**Six-fit resource pilot COMPLETE; full tuning and primary refits PENDING.**
The [Colab guide](COLAB_FAIR_COMPARISON_GUIDE.md) and
`notebooks/gridcast_fair_comparison_colab.ipynb` add a Linux resource-admission
gate (`/proc/meminfo`, cgroup limit/usage, pressure stall, disk, CPU-only,
two threads), committed dataset fingerprints, a supervising parent that kills
the pilot at 2,048 MiB/330 s, run inspection and lightweight provenance export.
The six fits, grid, cutoffs and caps are unchanged. The user has explicitly
authorized the six-fit CPU resource pilot, conditional on successful preflight.
The larger 204-fit search and 32 primary refits still require separate approval.
See [the Colab pilot report](GRIDCAST_COLAB_SIX_FIT_PILOT_REPORT.md) for execution status.
The reviewed [fair model comparison protocol](FAIR_MODEL_COMPARISON_PROTOCOL.md)
and `configs/fair_model_comparison.json` supersede the provisional grid below
this plan's earlier version. The framework provides a default dry run,
synthetic smoke mode and separately authorized development/full execution.
The six historical resource fits are complete; no historical hyperparameter
search or new primary-test model evaluation has run.

**Question/design:** Compare no-weather Ridge and LightGBM with the identical
fixed feature matrix, fitting-only median imputation/scaling, chronological
fitting and three July-2023–March-2024 validation blocks. Select per horizon
by pooled validation MAE in MW, then freeze before exploratory primary refits.
Keep the unchanged four test folds and imputed-target exclusions. The original
normalized-MAE Ridge search versus fixed LightGBM, preprocessing and explicit
calendar-representation differences are audited in the protocol.

**Search:** Five original Ridge penalties; twelve predefined LightGBM
candidates combining three leaf sizes, L1/L2 objectives and two bounded
rate/iteration/minimum-child/regularization profiles. No early stopping or
quantile training. Record completed fits, failed attempts, observations,
wall time and peak memory separately. Equal information and a common
per-model time cap do not establish equal model complexity or actual cost.

**Validation/resources:** Synthetic smoke passed in 4.080 s, peak 413.19 MiB.
The earlier 10–45 minute, 0.7–2 GiB, 50–300 MiB full-run planning range
remains unverified; the pilot does not measure full-search retention or
larger final fitting spans. See the measured report before considering approval.
The configured campaign
ceiling is 60 minutes, each model search cap 30 minutes, CPU threads at most
two, memory at most 2 GiB. Hitting a cap fails the campaign and requires
approval to continue. The completed six-fit pilot does not authorize the
204-fit search or 32 primary refits.
Priorities 3–5 and their approval boundaries remain unchanged.

## Priority 3: classical SARIMAX and seasonal smoothing

**Question:** Are recurrent statistical states competitive once their fitting
span and seasonal specification are tuned fairly?

**Dataset/features:** Same no-weather demand history and deterministic
calendar. The existing SARIMAX implementation uses log-demand, daily
seasonality and weekly Fourier/calendar regressors. The existing double
seasonal Holt–Winters implementation uses periods 24 and 168. Start there,
then add ordinary additive/damped seasonal exponential smoothing with period
24 or 168 as a clearly separate comparator; ordinary Holt–Winters is not a
two-season model. [Statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
documents seasonal and regression specifications.

**Training:** At most six predefined SARIMAX combinations: orders (1,0,1)
or (2,0,1), daily seasonal differencing/MA terms, and fitting span 180 or
365 days. Reduce the Cartesian product to six before running. Compare
fitting spans for existing double-season smoothing; choose smoothing/trend
parameters only in earlier training/validation blocks. Use a pilot with a
fixed optimizer iteration/time cap, recording optimizer failures rather than
silently accepting unconverged fits. Update filtering states only through
each issue time. Match output keys even when training spans differ, and
document span differences explicitly.

**Leakage risks:** Refiltering across future observations with smoothed
states; initializing seasons from a future period; filling missing weather
backward; silently dropping difficult nonconverged folds. No future-known
weather exogenous values are included in the initial study.

**Metrics/acceptance:** Shared point/probabilistic metrics, all-fold completion,
convergence rates, finite predictions, state-filter versus library-forecast
agreement and paired seasonal-naive comparisons. Improvement is not required
for completion; scientifically useful negative results are acceptable.

**Compute:** CPU only; one pilot configuration capped at 10 minutes. A six-
candidate, four-fold study could take 30–120 minutes or longer, particularly
with large seasonal states. Stop and request approval rather than exhausting
the shared machine. Estimated working memory 1–3 GiB; measure first.

## Priority 4: authentic issue-available weather

**Question:** Does an audited forecast archive improve demand forecasts beyond
the no-weather model without privileged future observations?

**Dataset/features:** Same demand snapshot plus a provider-specific long-form
archive with initialization, **actual publication/availability**, valid time,
source/version, units and geographic weighting provenance. The new strict
feature schema is already implemented. Temperature/humidity/wind and causal
observation/forecast features must be identical across comparison models.

**Acquisition/training:** First audit a small date/site sample, archive licensing,
timestamps, horizon coverage and revisions. No full download is authorized by
this plan. Open-Meteo's [Single Runs API](https://open-meteo.com/en/docs/single-runs-api)
explicitly distinguishes initialization from public availability. Its listed
coverage differs by model; most model runs start in April 2026, and historical
IFS entries can include hindcasts. That does not establish authentic original
operational forecasts for all four folds. Verify the provider's actual records,
not merely current release schedules or reconstructed timestamps.

If authentic coverage is partial, define a separate common-key subset and
evaluate all comparators there. Do not silently replace unavailable forecasts
with reanalysis. Keep the original full no-weather benchmark. Fit weather
corrections only on labels already available at issue time. Train/validate
using issued forecasts when enough archive data exist; otherwise document
any observation-to-forecast distribution mismatch.

**Leakage risks:** Initialization mistaken for availability; hindcasts mistaken
for issued forecasts; seven-day buckets used despite publication delay;
geographic weights chosen from test outcomes; future observations in bias
correction; missing-forecast imputation presented as genuine weather data.

**Metrics/acceptance:** Shared point/interval metrics, actual availability
coverage by fold/horizon, missingness, issue-time causality and latency
sensitivity. Operational wording requires externally supported publication
evidence and complete metadata. A failed source audit is a valid outcome.

**Compute/storage:** Sample audit under 5 MiB first. A multi-year, multi-run,
seven-site archive may occupy roughly 0.2–1 GiB or more depending on format
and forecast horizon; establish a measured estimate before acquisition.
Any download over 500 MB, paid API or cloud provisioning needs approval.
Do not duplicate shared immutable archives across projects unnecessarily.

## Priority 5: TCN and N-BEATS-style forecasting

**Question:** Do existing neural models offer useful accuracy/uncertainty gains
under matched information and a controlled training budget?

**Dataset/features:** Same snapshot, four folds and horizons. Start no-weather,
with causal demand windows and known calendar covariates. Use identical
lookback (initially 168 hours), normalization and target masks. The repository
has an N-BEATS-style generic-basis implementation, not a claim of reproducing
all published [N-BEATS](https://arxiv.org/abs/1905.10437) benchmark results.

**Training:** Use `scripts/neural_benchmark_pending.py`, first one small
development CPU smoke fit, then approved TCN/N-BEATS-style fits. Width 32,
two layers/blocks and at most 20 epochs are a pilot budget, not final optimized
architectures. Early stopping uses only permitted validation data; parameters,
lookback and learning rate are frozen before primary test evaluation. Add
seeds 0/1/2 for an approved final run. Keep point-objective comparisons
separate from the existing pinball-trained neural models; identify median
versus conditional-mean forecast differences.

**Leakage risks:** Windows ending after origin; future weather; validation
targets crossing cutoffs; test-driven architecture choice; claiming GPU
execution from hardware availability. Correct temporal convolution padding
and identical inference keys must be tested.

**Metrics/acceptance:** Shared metrics plus training curves, deterministic
seed checks where supported, calibration and compute cost. Require finite
complete predictions and proper provenance. Publish unsuccessful models
without promoting a selected lucky seed; no SOTA/deployment claim.

**Compute:** Long CPU/MPS jobs or Kaggle/Colab GPU runs require explicit
approval. Planning estimate: 0.5–4 hours per architecture/seed on CPU and
roughly 2–6 GiB working memory, with GPU times unknown until measured. A CUDA
environment can require multi-gigabyte dependencies. The safe runner installs
nothing; the legacy Kaggle wrapper now prints a plan by default and requires
explicit execution plus an already prepared project environment.

## Approval boundaries

The synchronization task completed code review, tests, row-level export and
metric reproduction. Priority 1 subsequently completed short CPU development
fits and interval evaluation, with explicit approval for a 2 GiB evaluation
cap after the initial 1 GiB failure. The separately approved six-fit Colab CPU
resource pilot subsequently completed
on October 22, 2026. No larger proposed training experiment or acquisition
was started. Approval is required before long shared-machine
training, GPU/MPS jobs, remote execution, large downloads, paid APIs or cloud
changes. Preserve all prior evidence and record every revised protocol as a
new run rather than overwriting measured results.
