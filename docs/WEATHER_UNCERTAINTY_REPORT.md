# Weather availability, residual drift and delayed uncertainty calibration

## Research questions and hypotheses

This extension asks whether historical weather adds useful information beyond
demand lags, how error distributions change across the existing expanding
folds, and whether timestamp-aware rolling or adaptive intervals reduce the
long-horizon undercoverage seen in the validated GridCast benchmark. Hypotheses
are investigated with a fixed configuration and matched forecast keys; test
results are not used to select features, penalties, window size or step size.
The no-weather comparison remains primary. There is no claim of a universal
forecasting improvement or causal explanation of distribution shift.

## Audit and preserved baseline

The baseline is commit `e6a1a6d` and the completed evidence directory
`results/forecasting_baselines/20261012T093203Z/`. See
`WEATHER_UNCERTAINTY_AUDIT_2026-10.md` for the architecture and availability
audit. Before modifying code, all 62 existing tests passed in the established
separate CPU processes. The existing reports, dataset, predictions, synthetic
network study and uncommitted work are preserved.

The source snapshot contains 97,914 EIA-930 TEX hourly demand records and
historical Open-Meteo reanalysis/forecast proxy columns. The processed SHA-256
is checked against the validated baseline before running this extension.
The experiment does not rebuild the data or download additional observations.

## Information availability and weather methodology

`features.available_weather` adds independently configured weather features
to the existing no-weather demand/calendar design. Variables, observation
delay, nonnegative lags, rolling windows and bounded past-only filling are
validated. Missing indicators accompany weather inputs; no backward fill is
used. Remaining nonfinite training feature rows are excluded by the existing
Ridge trainer. At inference, that trainer's fitting-period medians supply its
existing fallback. The fallback must not be interpreted as an observed or
available forecast. Calendar and trailing demand normalization are unchanged.

Three explicitly labelled modes are supported in addition to `none`:

* `history_proxy`: temperature, humidity and wind at least one hour before
  origin, lags 0/1/24 relative to that delayed series, and trailing 24/72-hour
  means. A maximum three-hour causal fill is allowed. This is retrospective:
  reanalysis does not have a verified point-in-time publication record.
* `oracle`: the same historical features plus future reanalysis at target time
  and target-minus-1/23-hour points. This measures a deliberately privileged
  information control, not operational weather forecasting.
* `issued`: accepts a long-form forecast archive with timezone-aware
  `issue_time`, `available_at`, `valid_time`, nonempty `source` and variable
  values. Each feature selects the latest matching valid-time row published
  by the forecast origin. Publication before issue, issue after valid time,
  ambiguous revisions and missing metadata are rejected. Schema validation
  cannot establish that a provider's timestamps are truthful; provenance must
  be audited externally. No real-data `issued` experiment is executed here.

The existing [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)
archive retains fixed lead-day offsets, not verified publication timestamps.
Its download time is not original forecast availability. No issue timestamp
is invented from those buckets. The strict pipeline rejects bucket-only
input, including the 168-hour case. The legacy pipeline remains available
for reproducing old measurements, with its availability claims corrected.

## Forecasting protocol and model comparison

The five validated no-weather models are read from immutable prediction
files. Only two new Ridge controls are trained, using the existing estimator,
penalty candidates and validation/refitting procedure. Every horizon uses the
same four folds, forecast-origin convention and test target boundaries as the
validated report. The exact `(fold, origin, horizon)` sets are asserted equal
across all models. Targets repaired during data cleaning are excluded.

Scalers and medians fit on fitting-period data only. Supervised training labels
must precede the relevant validation/calibration cutoff. Hyperparameters
remain selected solely by the permitted validation block. Coefficients,
intercepts, scaler arrays, feature names and medians are saved for all 32 new
fold/horizon fits. There is no new LightGBM, SARIMAX or neural training run.

| Ridge information regime | 1h MAE (MW) | 24h MAE (MW) | 48h MAE (MW) | 168h MAE (MW) |
|---|---:|---:|---:|---:|
| Validated no-weather | 313.17 | 2234.26 | 3194.42 | 4136.95 |
| Delayed historical proxy | 311.96 | 2237.86 | 3205.46 | 4192.61 |
| Retrospective oracle | 311.96 | 2221.88 | 3098.17 | 3998.47 |

Counts per model are 17,492/17,400/17,304/16,824. MAE, RMSE, sMAPE and
fold-specific 168-hour seasonal MASE are saved with the original definitions.
Skill is `1 − model MAE / baseline MAE`, separately against daily and weekly
seasonal naive on the identical keys. Raw LightGBM quantile pinball and
coverage measurements remain part of the unchanged no-weather baseline.

Paired differences below use 500 seven-day moving-block bootstrap replicates
within each fold, seed 0, weighting by valid hourly observations:

| Comparison to no-weather Ridge | Horizon | MAE difference (MW) | 95% interval (MW) |
|---|---:|---:|---|
| History proxy | 1 | −1.21 | [−1.78, −0.73] |
| History proxy | 24 | +3.60 | [−10.54, +19.43] |
| History proxy | 48 | +11.04 | [−22.93, +41.86] |
| History proxy | 168 | +55.66 | [−7.17, +115.72] |
| Oracle | 1 | −1.21 | [−2.19, −0.40] |
| Oracle | 24 | −12.38 | [−53.62, +30.44] |
| Oracle | 48 | −96.25 | [−179.76, −8.57] |
| Oracle | 168 | −138.49 | [−266.14, +23.15] |

The history-proxy gain at one hour is very small. Its longer-horizon intervals
do not establish an improvement. Oracle results are explanatory controls;
they cannot support deployment claims. These are exploratory intervals across
multiple horizons without a multiple-comparison adjustment. They condition
on fitted models and cannot capture parameter or archive uncertainty.

## Distribution-shift analysis

Diagnostics include target season, month, year-month, fold, horizon, day/night
(06:00–18:00 local versus the remainder), and weekday/weekend. Chicago local
**interval start** handles the hour-ending convention and DST. Demand-regime
thresholds are computed separately from each fold's fitting period rather
than chosen from test values; realized demand is used only to assign diagnostic
slices, never as a future-known model input.

Adjacent-fold diagnostics report signed-residual Wasserstein distance,
Wasserstein distance normalized by the fitting-period seasonal scale, change
in MAE/bias, and a descriptive two-sample KS statistic. No IID KS p-value is
reported. For no-weather Ridge at 24 hours, raw residual Wasserstein distances
are 787.63, 840.01 and 560.20 MW for successive fold pairs, with descriptive
scaled KS statistics 0.0915, 0.0893 and 0.0700. Fold-to-fold MAE changes are
+733.17, −837.05 and +555.99 MW. Changing season mix, demand and refitted models
can all contribute; these diagnostics do not isolate a cause.

## Uncertainty methodology and feedback availability

The existing static absolute-residual conformal result is preserved as the
baseline. New methods use unchanged no-weather Ridge and LightGBM point
predictions. Each `(model, information regime, fold, horizon)` has independent
calibration state. A score becomes eligible only at `target_time + delay`,
not a number of rows after issue. The experiment assumes demand at the hour
ending at the origin is known, so demand feedback delay is zero, consistent
with the validated origin definition.

Rolling calibration uses the most recent 1,440 **finite eligible residuals**,
with at least 168 required. Imputed labels are excluded. The finite-sample
order-statistic rank uses the actual eligible count. Gaps and irregular
forecast origins therefore cannot release feedback early. Static and online
methods differ at boundaries because online feedback can include a previously
labelled gap row once its target has actually arrived.

The adaptive method is a bounded delayed-feedback variant inspired by
[Gibbs and Candès (2021)](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).
For each nominal miscoverage level, it stores the interval actually issued and
updates its effective level only after that interval's target is observed:

\[
\alpha_{next}=\operatorname{clip}\left(\alpha_{current}
+0.01(\alpha_{nominal}-\mathbb{1}\{y\notin C_{issued}\}),\;0.001,\;0.5\right).
\]

At long horizons, many intervals may be issued before any feedback about them
arrives. This delay is explicitly respected. Bounded levels, finite rolling
score pools, overlapping targets and delayed updates differ from the original
theory; no distribution-free or conditional coverage guarantee is asserted.
Independent levels may produce nonnested intervals. Nine-level pinball loss
is reported together with quantile-crossing diagnostics without rearranging
the intervals after issuance, which would invalidate their feedback record.

Empirical 80/90/95% coverage, width and mean Winkler score are reported overall
and by season, demand regime, month, fold, day/night and weekday/weekend.
For bounds \(l,u\) and miscoverage \(\alpha\), Winkler score is

\[
(u-l)+\frac{2}{\alpha}(l-y)\mathbb{1}(y<l)
+\frac{2}{\alpha}(y-u)\mathbb{1}(y>u).
\]

## Completed interval results and failure cases

Executed evidence is `results/weather_uncertainty/20261012T150556Z/`, with
total runtime 343.32 s on CPU. Historical-proxy fits average 4.93 s per fold
(prediction 1.31 s); oracle fits average 7.62 s (prediction 4.10 s). Most
additional runtime is post-hoc calibration and diagnostics, not model training.
The run records commit `22b59b2` plus dirty source hashes and 32 saved model
artifacts. No MPS, CUDA or remote GPU execution occurred.

The following is measured nominal 90% performance on identical target keys:

| Model | Horizon | Static coverage | Rolling coverage | Adaptive coverage |
|---|---:|---:|---:|---:|
| Ridge | 1 | 90.33% | 89.94% | 90.05% |
| Ridge | 24 | 87.64% | 89.41% | 88.67% |
| Ridge | 48 | 86.26% | 89.31% | 87.38% |
| Ridge | 168 | 82.81% | 88.62% | 84.61% |
| LightGBM | 1 | 89.93% | 90.03% | 90.07% |
| LightGBM | 24 | 88.22% | 88.90% | 88.53% |
| LightGBM | 48 | 86.73% | 89.01% | 87.14% |
| LightGBM | 168 | 86.41% | 88.93% | 83.99% |

| Model, 168h | Method | Mean 90% width (MW) | Winkler (MW) | Nine-level pinball (MW) |
|---|---|---:|---:|---:|
| Ridge | Static | 14351.87 | 26788.53 | 1083.29 |
| Ridge | Rolling | 17823.16 | 26024.25 | 1073.97 |
| Ridge | Adaptive | 18553.89 | 28218.80 | 1188.50 |
| LightGBM | Static | 15586.84 | 25329.11 | 1049.10 |
| LightGBM | Rolling | 17647.90 | 25411.50 | 1050.27 |
| LightGBM | Adaptive | 18438.91 | 28793.83 | 1186.21 |

Rolling coverage is closer to nominal at 24/48/168 hours but requires wider
intervals. Ridge's 168-hour mean Winkler score improves descriptively; the
LightGBM score slightly worsens despite better coverage. No significance
claim is made for interval-score differences. Coverage alone is insufficient
to rank uncertainty methods.

Conditional diagnostics reveal remaining failures. Ridge's 168-hour rolling
90% coverage is 89.61% in autumn, 88.45% in spring, 93.48% in summer and
82.73% in winter. Static winter coverage was 73.46%. These observed seasonal
differences are not conditional guarantees. All 80/90/95% results and counts
are in `interval_overall.csv` and conditional tables.

At 168h, independent adaptive levels yield quantile crossing on 37.42% of
Ridge forecasts and 31.99% of LightGBM forecasts. Long feedback delay and the
bounded update/pool choices are plausible contributors but have not been
isolated causally. This variant is an executed experimental failure case,
not a recommended calibrated distribution or an improvement over rolling.
There was no test-driven retuning after seeing this outcome.

Figures `horizon_error`, `seasonal_error`, `calibration_coverage` and
`interval_width` are saved in PNG and SVG with captions. Primary no-weather
point metrics exactly reproduce the earlier validated tables. Future weather
claims, source revision history, actual reanalysis availability, dependent
coverage guarantees and robustness to unseen grid conditions remain unproven.

## Software validation

All 100 tests passed: 95 non-neural tests in 33.34 s and 5 neural fixture tests
in 21.72 s, using separate CPU processes. The 38 added tests check weather
timing/gaps, legacy archive rejection, model/config validation, irregular
origins, feedback delays at all four horizons, finite-score counts, imputed
label exclusion, reproducible online results, delayed adaptive updates,
shift diagnostics, local-time slices and known Winkler values. Neural fixture
tests are not evidence of an ERCOT neural benchmark. Existing deprecation
warnings remain; no configured static type checker exists.

Ruff, bytecode compilation, lock consistency and whitespace checks passed.
The completed package also passed feedback-causality and matched-interval-key
checks, archived-model replay, exact preservation of baseline MAE values and
all baseline prediction hashes. All 31 pre-existing modified/untracked file
hashes remained unchanged. Four figure types were visually inspected. The
new output directory occupies approximately 52 MiB and is excluded from Git.

## Reproducibility and pending work

Configuration: `configs/weather_uncertainty.json`. Run from the repository root:

```sh
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MPLCONFIGDIR=results/scratch/matplotlib \
  .venv/bin/python -m gridcast.experiments.weather_uncertainty
```

The runner accepts `--analysis-only` to skip Ridge fitting. It writes a new
timestamped output directory, with all baseline prediction hashes checked
again after execution. Metadata includes seeds, versions, data/source/config
fingerprints, fold boundaries, device, training/inference timings and model
artifacts. No dependency versions are changed. Figure captions are saved next
to PNG/SVG horizon, seasonal, coverage and width plots.

Full neural training is pending explicit approval, not executed. A runner
using existing TCN/GRU/N-BEATS-style modules supports CPU, MPS or CUDA and
preserves the four-fold, four-horizon, no-weather protocol:

```sh
# Only after approving the compute job and checking competing jobs/memory.
PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/neural_benchmark_pending.py --arch tcn --device cpu
# On an approved Kaggle/Colab host with the locked environment and dataset:
# use the same command with --device cuda.
```

The script installs/downloads nothing. Provisioning a CUDA environment may
need multi-gigabyte dependencies and must be approved separately. There is no
Kaggle upload, GPU run or N-HiTS implementation claim. Useful next steps are
verified forecast-run availability, validation-only calibration tuning,
longer bootstrap blocks, independent replications, and fresh statistical or
neural comparisons after compute approval.
