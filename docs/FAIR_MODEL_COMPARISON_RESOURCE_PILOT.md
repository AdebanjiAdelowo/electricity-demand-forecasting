# Fair comparison resource pilot and readiness gate

## Outcome: BLOCKED by system memory pressure

The real-data pilot was **not admitted**. Zero model fits were executed and
no historical search, selection, primary-fold refit or new primary-test
comparison occurred. Real-data fit times, training-row counts, fitted-feature
counts, model sizes and fit-related peak RAM are **unavailable**, not zero.
The current host is unsafe for the requested measurement; the existing caps
have not been demonstrated sufficient or insufficient by this attempt.

Initial local/remote SHA: `8cdf2689db6819ab14f76858853afd4d70ab5989`, `main`,
clean tree and 0/0 divergence after fetching. No newer commit or equivalent
real-data resource pilot existed. The reviewed scientific configuration and
original fair-comparison protocol remain byte-identical. The only existing
measurements were the previously documented synthetic smoke runs.

System checks on October 14, 2026 returned
`kern.memorystatus_vm_pressure_level = 2`, repeatedly. This is the **dispatch
warning flag**, not the differently numbered internal XNU pressure enum.
The [Apple kernel sysctl implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_memorystatus_notify.c)
converts internal pressure to notification flags; the
[Apple dispatch header](https://github.com/apple/swift-corelibs-libdispatch/blob/main/dispatch/source.h)
defines normal/warning/critical as 1/2/4. The pilot admits only normal (1).
Unknown values, unavailable sensors and unsupported platforms fail closed.
An apparently favorable free-memory percentage must not override a warning.

`memory_pressure` reported about 37% free; disk availability was about 21 GiB.
The read-only sysctl is blocked by the Codex filesystem sandbox, so the
preflight required permission for the system check. It was recorded outside
that sandbox without granting permission for any larger experiment.

Recorded no-fit preflight:
`results/fair_model_comparison/resource_pilot_20261014T113953837979Z/`.
At **2026-10-14 11:39:53.940677 UTC**, the gate again observed warning (2),
returned `blocked_memory_pressure` and recorded zero completed fits.
Preflight wall time was **0.102226 s** and process peak RSS **217,956,352 bytes
(207.86 MiB)**. This is import/preflight overhead, **not model-fit memory**.
No pilot observation prefix was loaded. The directory contains configuration,
fixed pilot plan, source/lock/version/Git hashes and pressure/failure metadata;
it is ignored by Git. The executed source was the initial SHA plus recorded
dirty source hashes, not a claim that the parent commit already contained this
new mode. Later guard/error-handling edits were checked by no-fit unit tests.

## Planned real-data measurements, unchanged candidate configurations

New entry point: `gridcast.experiments.fair_resource_pilot`. Default invocation
prints its plan. `--preflight-only` can only inspect and record system state;
`--execute-pilot` is the explicit six-fit gate. They are mutually exclusive.
Neither imports nor tests invoke historical search automatically.

The fixed plan is independent of prior Ridge/LightGBM accuracy:

| Candidate | Original grid configuration |
|---|---|
| r01 | Ridge alpha 1.0 |
| l02 | L2, 600 trees, 63 leaves, rate .05, child minimum 40, L1=0/L2=1 |
| l11 | L1, 300 trees, 63 leaves, rate .08, child minimum 80, L1=1/L2=10 |

Fit each at **1 and 168 hours**, six fits total, using existing shared
no-weather features, fitting-only median imputation/scaling, seed 0 and
at most two LightGBM/BLAS threads. No tree/regularization/grid setting is
changed. No MAE table, winner, selection file or bootstrap comparison is
produced by this pilot. These fits measure resource use and finite complete
forecast output only.

Historical fitting origin start: **2015-08-01 UTC**. Supervised target cutoff:
**2023-07-01 UTC**, strict inequality for every horizon. Data before the fitting
start supply the existing demand-history context. Validation origins lie in
**2023-07-01 to 2023-07-15 UTC**, with targets also strictly before July 15.
The eight-year fitting span is representative of development fitting, rather
than a shortened few-day toy training set. The final primary folds may have
larger fitting spans; that uncertainty must enter later extrapolation.

`load_prefix` projects only demand/imputation columns and applies a parquet
predicate `utc_end < 2023-07-15`, serial decoding, then verifies the returned
UTC prefix. Later primary labels cannot enter the returned model inputs.
Parquet may internally scan a mixed row group; this is a logical row-access
boundary, not a claim that no later compressed bytes were touched. Input
hashing reads the immutable file without decoding its labels. No primary
prediction archive is loaded by this pilot.

Each completed fit saves an append-only-run measurement with candidate/params,
horizon, row/feature counts, target cutoff/latest supervised target, validation
origin range, feature-construction time, fit time, prediction time, iterations,
process high-water RSS, serialized pipeline size, writing time, finite outputs
and exact key completeness. Learned imputation/scaling time is included in
pipeline fit time; it is not falsely separated from estimator fitting.
Feature construction is separately timed. Process RSS is cumulative within
the sequential pilot, so a later fit can inherit an earlier high-water mark.
It is not an isolated causal memory attribution to each model.

## Resource-control audit

| Control | Existing full framework | Pilot |
|---|---|---|
| Campaign wall limit | 3,600 s | 300 s including prefix loading/fitting/writing |
| Per-model search limit | 1,800 s, checked before fits and after final candidate | No search; fixed six-fit plan |
| Nominal process memory | 2,048 MiB | Same nominal ceiling |
| Early stopping for RAM | Failure when sampled peak exceeds cap | Stop at 1,843 MiB (~90% of nominal ceiling) |
| Monitoring | One-second SIGALRM checks and stage checks | Reuses those checks; pressure read before every fit |
| Training threads | Explicit CPU, max two threads, threadpoolctl | Same |
| Native execution | CPU LightGBM | Same |
| Search/evaluation authorization | Separate explicit historical/full flags | No route to either stage; pilot flag only |
| Selection freeze | Hashed before test evaluation | No selection performed |

The 90% threshold is a pilot early-stop margin, not a cap increase or a change
to the scientific/full-run configuration. No automatic retry, larger memory
allowance, shorter fitting period or smaller search grid is implemented.

**RSS monitoring is not a hard OS memory limit.** Python signal handlers may
wait until native LightGBM or BLAS operations return. Allocations can cross
both the early margin and nominal ceiling before Python responds. High-water
RSS will then expose the excess, but it cannot undo it. No OS RSS quota,
external supervising worker, cgroup or process-footprint enforcement exists.
Time interruption likewise may wait inside native calls. Two training threads
are bounded explicitly; serial parquet decode limits input-reader parallelism.

`fit_events.jsonl` records actual fit starts/completions/failures in the common
helper. Completed measurements and model/prediction artifacts are persisted
after each fit; `active_attempt.json` identifies an interrupted one. Resource
errors and Python keyboard interruptions write terminal metadata and preserve
partial results. A SIGKILL/system termination cannot execute Python `finally`;
only already-flushed artifacts/ledger entries can be guaranteed then. New
output directories reject reuse. No existing experiment directory is changed.

The full framework's default dry run and separate historical/primary flags
remain intact. `fit_predict` gained only feature-construction timing and feature
count fields; its fitting masks, transformations and model calculations did
not change. These fields do not alter previously published results.

## Estimates: not recalibrated because no real fit ran

No new measured full-run estimate is scientifically defensible. The prior
protocol's **10–45 minutes, 0.7–2 GiB, 50–300 MiB** remains an **unverified
planning range**, not a resource-pilot finding. Its assumptions came from
older point-only timings with a different representation. This preflight
cannot narrow that range or establish fit within the existing limits.

Once the six representative fits are admitted, estimate components separately:

| Component | Planned count | Required measured input and uncertainty |
|---|---:|---|
| Ridge development search | 60 fits | Ridge preparation/fit/prediction time, across horizons/blocks |
| LightGBM development search | 144 fits | 72 profile-A + 72 profile-B fits, objective/leaves effects |
| Ridge final refits | 16 fits | Larger final fitting spans and longer forecast schedules |
| LightGBM final refits | 16 fits | Same, plus unknown selected candidate; no winner inferred from pilot |
| Statistical evaluation | Four folds × four horizons | Paired metrics/day-block bootstrap; pilot does not benchmark this |
| Artifact writing | 236 fit events, candidate predictions, 32 final pipelines | Measured serialization sizes/time plus longer forecast tables |

Missing measurements should remain explicit. In particular profile-A 600-tree
L1 is not directly measured by this six-fit design and could cost more than
600-tree L2 or 300-tree L1. Do not equate lower tree count with proportional
runtime or extrapolate only the fastest candidate. The 24/48-hour horizons,
later development blocks, final fitting spans, allocator retention, all-candidate
prediction retention and concurrent system load add uncertainty. The old
search retains candidate prediction frames across fits; that memory growth is
not measured by isolated short pilot schedules. A successful six-fit pilot
alone would not turn RSS monitoring into a hard cap.

Compare the conservative upper range separately to each 30-minute model cap,
the 60-minute total cap, and 2 GiB ceiling. No cap was changed in this task.
**Current verdict is BLOCKED by host pressure; full-run feasibility remains
unmeasured.** This is not evidence favoring either model or changing protocol.

## Validation and preservation

No-fit tests cover the fixed grid, target cutoff, admission under normal/
warning/critical/unknown states, preflight isolation, six-fit-only dispatch,
resource-error partial results, keyboard interruption, no overwrite, original
caps and continued full-mode authorization. Their fitted objects are mocked;
they do not constitute historical or synthetic training evidence.

The current host warning made the full training-containing test suite
inappropriate. It was deferred rather than bypassing the pressure guard.
The prior 155-test suite at the starting commit remains historical validation,
not a fresh full-suite claim for this change. Current validation passed **41 no-fit tests**: 12 new pilot tests and 29
existing fair-comparison tests with the two model-fitting tests deselected.
The 40-test combined run preceded the final keyboard-interruption test; all
12 final pilot tests then passed. Ruff, changed-file formatting, compilation,
offline lock validation and whitespace checks passed. Four existing evidence
packages and the completed interval package passed read-only integrity checks.
These checks do not generate new primary-fold forecasts or retune any model.
No current executed test failed; the full training-containing suite remains
deferred because of host pressure.

SHA-256 checks confirm all **124 protected input/archive/published-report files**
from the prior preservation inventory remain unchanged. The scientific JSON
configuration and original fair-comparison protocol remain byte-identical to
`8cdf268`. Another 83 existing fair-preparation artifacts were inventoried;
none were modified. The new preflight and test artifacts are separate and
ignored. No dependency was installed/upgraded, data rebuilt or shared cache
removed. No GPU/MPS, cloud, paid service, large download or deployment occurred.

## Colab/Linux addendum (preparation only)

A Linux execution path is now prepared; see the
[Colab guide](COLAB_FAIR_COMPARISON_GUIDE.md). On Linux, `pressure()` returns a
`/proc`/cgroup snapshot and `admit()` applies the fail-closed Linux rules in
`gridcast.resources`. The macOS dispatch gate and its normal-only rule are
unchanged, and other platforms remain blocked. The expected dataset hash now
comes from the committed `configs/approved_dataset_snapshot.json`, which must
agree with the local benchmark record when that file exists. `--output` lets a
supervisor name the new (still non-existing) run directory. The scientific
plan, candidates, cutoffs and caps are unchanged. No real-data fit has run.

## Next stages and approval

First wait for verified **normal system pressure (1)**. Then resume the
already scoped six-fit pilot under the unchanged limits:

```sh
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m gridcast.experiments.fair_resource_pilot --execute-pilot
```

Do not substitute a warning override. The read-only sensor may require local
sandbox permission; unavailable pressure readings cause no fit.
Repeat the postponed full suite when safe, then use actual measurements to
write a READY/CONDITIONAL/BLOCKED feasibility decision. No cleanup of another
session or resource-cap increase is authorized.

**Only after separate explicit full-run approval**, the unchanged proposed
full-run command is:

```sh
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m gridcast.experiments.fair_model_comparison \
  --config configs/fair_model_comparison.json --mode full \
  --authorize-historical-search --authorize-primary-evaluation
```

That command was not executed. READY would still not constitute permission;
this attempt is BLOCKED and cannot support approval for the full benchmark.
