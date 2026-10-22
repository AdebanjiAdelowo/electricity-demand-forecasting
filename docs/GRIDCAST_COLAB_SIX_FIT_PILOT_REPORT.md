# GridCast Colab six-fit CPU resource pilot report

## Outcome

**COMPLETE — six of six fits, 22 October 2026.** The user authorized this
resource-only pilot subject to preflight. The full 204-fit development search
and 32 primary-fold refits remain pending separate approval. Baseline
forecasting and Priority 1 calibration were already complete; no 168-hour
calibration candidate passed its development coverage gate.

Run: `resource_pilot_colab_20261022T212520455907Z`.
Internal wall time: **68.6553 s**; supervised child lifetime: **71.7648 s**.
Process high-water RSS: **657.18 MiB**; sampled process-tree peak:
**654.77 MiB**. Exit code 0, no supervisor kill or interruption.
All values here are measured on this runtime unless explicitly called estimates.

## Environment and limits

Free Google Colab CPU, Linux 6.6.122+ x86_64, Intel Xeon @ 2.20 GHz,
2 logical and affinity CPUs. Kernel Python 3.13.16; isolated project Python
**3.12.3**, installed through `uv==0.11.16` and frozen core dependencies.
No torch in the project environment; no accelerator attached; CUDA disabled.
All six computational thread variables were set to 2.

Locked versions: numpy 2.4.6, pandas 2.3.3, pyarrow 25.0.1, scipy 1.18.1,
scikit-learn 1.9.1, lightgbm 4.7.0, threadpoolctl 3.7.0, statsmodels 0.15.0.

Limits remained 2,048 MiB process-tree RSS / 330 s supervisor lifetime,
1,843 MiB internal RSS guard / 300 s internal wall time. Admission was
recorded initially and before every fit. No retry or additional fit ran.

## Source and dataset provenance

Execution commit: `32be70b38d2f75745d4fb69a7cc8e4083ec08d3b`, clean checkout.
Repository HEAD before documentation: `ac362c1c978467f72abeef9f18141dd860f87dd3`.

| Input | SHA-256 |
|---|---|
| Original scientific configuration | `5418e81fa00a9930248d2f1a25166eed826fc139bad06f36fc54a4e3fa805d09` |
| uv.lock | `cdb4de095c4cf2f0de947f0fe6258f49b96d4d1fdff082566fca2d6af07ab2c8` |
| Processed dataset | `1e3acfa6c4070d0dc69d6a2d854964f09afc42b3291221fda646c606249ca5ab` |
| Dataset manifest | `b881107165298776a96218f88ce60123340f049b960efcd03ebc8c555f1e34de` |
| Git bundle | `f2912d64504b4fd65a65247bc9dc16bedde7017889061994ec856df908c18b53` |

The run's serialized `configuration.json` has SHA-256
`5a05214eed03b7fc9be2f01f89192439c2915233707c3641830f2fd8744bac0f`;
its whitespace differs from the original JSON, while its parsed settings
are the unchanged approved configuration. Preflight verified the original byte hash.

Snapshot `eia930-TEX+openmeteo@2026-10-10`: 97,914 hourly UTC rows,
2015-07-01 06:00 through 2026-08-31 23:00, approved demand/imputation schema,
72 causally repaired observations, provenance and licenses in the manifest.
The pilot decoded only the demand/imputation prefix ending 2023-07-14 23:00.

## Preflight

**CONDITIONAL**, no blocking failures. The only warning was
“no cgroup memory limit found; host MemAvailable used as the binding sensor”.
The runtime exposes cgroup v2 with `memory.max = max`, rather than an
unreadable finite quota. This documented condition was checked before proceeding.
The wrapper represents unlimited memory using a sentinel; it is not a real
exabyte allocation. The independent 2 GiB process supervisor remained active.

Fresh pilot preflight: 2026-10-22 21:25:20 UTC, available RAM about 11.33 GiB,
RSS about 219.68 MiB, 86.49 GiB free disk, no swap, memory pressure-stall
some/full avg10 both 0.0. Git, package versions, configuration, dataset hash,
schema, coverage, provenance, CPU and thread checks passed.

## Six-fit resource measurements

| Candidate | Horizon h | Training rows | Validation rows | Fit s | Predict s | Peak RSS MiB | Model bytes | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| r01 | 1 | 69,335 | 335 | 1.920 | 0.0238 | 642.12 | 6,868 | Complete |
| l02 | 1 | 69,335 | 335 | 18.127 | 0.0289 | 642.12 | 3,719,781 | Complete |
| l11 | 1 | 69,335 | 335 | 12.732 | 0.0169 | 650.98 | 1,803,037 | Complete |
| r01 | 168 | 69,168 | 168 | 1.194 | 0.0067 | 655.05 | 6,868 | Complete |
| l02 | 168 | 69,168 | 168 | 15.457 | 0.0153 | 655.05 | 3,603,809 | Complete |
| l11 | 168 | 69,168 | 168 | 12.167 | 0.0107 | 657.18 | 1,783,741 | Complete |

All fits used 108 shared no-weather features and seed 0. `r01`: Ridge alpha
1.0. `l02`: LightGBM L2, 600 trees, 63 leaves, rate .05, child minimum 40,
L1=0/L2=1. `l11`: LightGBM L1, 300 trees, 63 leaves, rate .08, child minimum
80, L1=1/L2=10. No early stopping. Training-only preprocessing row counts
match the training counts. Each fit completed its configured iterations.

The per-fit RSS column is cumulative process high-water RSS; later fits can
inherit earlier peaks. It is not isolated model-attributable memory.
The supervisor's sampled peak can be below process high-water RSS because
polling can miss short spikes. Captured pilot stderr was empty; no convergence
or numerical warning was observed. LightGBM verbosity is suppressed by the
existing protocol, so empty logs do not establish absence of all internal notices.

## Validation and evidence

Fitting began 2015-08-01 UTC. Every measurement records latest supervised
target **2023-06-30 23:00 UTC**, strictly before the July 1 cutoff.
All six saved prediction files have unique, UTC, horizon-aligned keys,
finite predictions, origins in the prescribed validation interval, and
targets strictly before July 15. The pilot checked complete expected keys
and identical observed/imputation values against its schedule. No primary
test fold was evaluated. No accuracy scores or winning-model selection were produced.

Both ZIPs passed CRC checks and every exported member's SHA-256 matched its
export manifest. The uploaded bundle's SHA-256 also matched from Colab.
The separate executed notebook was read back from Drive: steps 2–11 ran
in order with execution counts 1–10, followed by a no-fit artifact audit.
A successful COMPLETE result and the outputs are present.

Persistent private Drive evidence:

- [Verified source bundle](https://drive.google.com/file/d/1Hmmnz36eO-fzUQGItSqmMbUT6dfMRfS7/view).
- [Configured notebook](https://drive.google.com/file/d/1B4f_XAee4U3tA88zvDG1PANHsRmjKfKc/view).
- [Separate executed notebook with outputs](https://drive.google.com/file/d/1uSgJnqLGu8yTl1sm8pVjCNjSgTbeDZXE/view).
- [Preflight provenance ZIP](https://drive.google.com/file/d/14E9L67VJJG6Q0MxQN3iSkstcoi4IqWGK/view),
  SHA-256 `018946aeed34bdf14421cba423003946b03d98c5dff5c15559fc44e40062298d`.
- [Pilot provenance ZIP](https://drive.google.com/file/d/1SMLKw6NNj7bMW4a6rN9rBSpVVX1qLynq/view),
  SHA-256 `d06562b0a3e44a1d6787cc9e4dc77d23df84019e14f182e952fa2113cc528ed3`.
- [Detailed audit JSON](https://drive.google.com/file/d/1qn88v51qoC_-24bBA1IGZa3GsepjsU6m/view).

ZIPs and audit are under `gridcast/exports`; the executed notebook is under
`gridcast/notebooks`. Lightweight exports exclude raw demand, fitted models
and prediction tables by default. Model and prediction files remain on the
Colab VM and are not guaranteed to survive runtime deletion. Original local
research archives were preserved. The added audit performs no fitting/scoring.

## Bottlenecks, limitations and recommendation

Fitting consumed 61.60 s, feature construction 5.08 s, and prediction 0.102 s.
The four LightGBM fits account for about 58.48 s of fitting. The observed
pilot stayed comfortably below both memory and time guards. These limits
were sufficient for this six-fit run; no cap change is indicated.

The larger experiment is **reasonable to consider for separate approval,
with conditions**, but its completion within the existing campaign limits
has not been demonstrated. This pilot omits the 24/48-hour horizons, later
fitting spans, 600-tree L1 candidates, full-grid prediction retention and
statistical evaluation. Larger searches can accumulate memory that this
sequential short validation pilot does not reproduce. Colab hardware and
session availability vary. No bitwise Mac/Linux reproducibility claim is made.
The snapshot is retrospective, and already-inspected primary folds support
exploratory comparisons rather than independent confirmation.

Before authorizing the larger run, review a conservative component budget
against each 30-minute model cap, the 60-minute campaign cap and 2 GiB memory
ceiling, and review full-run enforcement/persistent evidence handling: the
six-fit supervisor dispatches only the pilot and does not automatically
protect the full-search entry point. No scientific-protocol correction was
identified by the pilot. Keep the grid and training span frozen; do not
reduce them solely to improve runtime. Do not infer model superiority.

## Software and preservation checks

33 no-fit Colab/pilot tests passed; relevant Ruff checks and whitespace checks
passed. Four baseline/evidence packages and the completed interval-validation
package passed read-only integrity verification. Notebook code cells compile.
The approved source, configuration, lockfile and original data are unchanged.
Documentation records authorization and measured completion only.
