# Supervised fair-comparison campaign preparation

## Status and approval boundary

The October 23 infrastructure implements the input, enforcement and persistence
items from [the readiness review](FAIR_COMPARISON_FULL_RUN_READINESS.md).
Validation uses mocked fits and tiny sleeping/printing subprocesses only.
No new historical fit, search, primary refit or primary scoring has run.
The completed six-fit pilot must not be repeated as setup.

**The resource-budget decision remains pending.** The conservative proxy still
puts LightGBM search near 41 minutes against its unchanged 30-minute limit.
Preparing these controls does not establish that the larger workload fits.
The full experiment still requires explicit historical-search and primary
approval plus an explicit decision to accept the risk of hitting existing caps.
No thread, seed, grid, cutoff, feature, preprocessing or scientific limit changed.

## Environment and approved inputs

Use a fresh free Colab CPU runtime, the same `uv==0.11.16`, Python 3.12 and
`uv sync --frozen --python 3.12` core environment as the pilot. Do not run
Step 8 of the old pilot notebook again. Full campaigns have a separate CLI;
the pilot supervisor still rejects full-search commands.

Check out an explicitly reviewed commit containing this infrastructure,
not the older pilot execution commit. Keep the checkout clean and record its
full SHA as `PINNED_COMMIT`; a dirty or mismatched source blocks preflight.
The existing immutable dataset remains in the repository and must pass
`configs/approved_dataset_snapshot.json` verification before decoding values.

Full mode additionally needs the original `20261012T093203Z` reference inputs:

- `run.json` — 10,424 bytes.
- `ridge.parquet` — 1,953,098 bytes.
- `lightgbm.parquet` — 11,444,514 bytes.

They are ignored by Git. Preserve them privately in a Drive folder you select;
never regenerate them or commit their contents. The approximately 12.79 MiB
combined size excludes archive/container overhead. On October 23 these three
existing immutable files were staged in the selected private Drive reference
folder. No historical campaign was launched. Preserve the local originals;
do not upload the whole results tree.

`configs/approved_fair_reference.json` commits their approved SHA-256 values,
Parquet schemas and row counts. Preflight hashes the complete files, reads
only Parquet metadata, and checks benchmark dataset provenance. This also
anchors the already audited reference key/schema identity without decoding
primary demand labels during preparation. Missing or changed references
block before fitting. Full mode uses `--baseline-dir` (or
`GRIDCAST_BASELINE_DIR`) rather than assuming a local ignored archive exists.
Development-only mode uses the committed snapshot approval without requiring
reference predictions. End-of-run hashes cover dataset and reference inputs.

## Preparation-only preflight

After mounting Drive, configure paths explicitly. These are examples, not
private paths embedded in a notebook:

```sh
export GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=src

.venv/bin/python -m gridcast.experiments.fair_campaign preflight \
  --mode full --expected-commit "$PINNED_COMMIT" \
  --baseline-dir /content/drive/MyDrive/gridcast/reference_20261012
```

Preflight verifies source, frozen configuration, locked dependencies, approved
dataset/reference inputs, CPU/thread settings, `/proc` RAM, cgroups, pressure
and disk via the existing admission interface. It builds no model and fits
nothing. It returns BLOCKED if any hard check fails; otherwise CONDITIONAL
because the resource-budget risk remains explicit. On macOS the Linux
campaign command is BLOCKED and the original dispatch-pressure gate is not
bypassed. The 41-minute proxy is a risk assessment, not measured full-run time.

## Future execution command — requires separate approval

Create and select an existing private persistent output folder. On Colab it
must be under the mounted `/content/drive`; an ephemeral VM output path is
rejected. All outputs are written directly there rather than copied only at
campaign end. The following command is documented, **not executed**:

```sh
.venv/bin/python -m gridcast.experiments.fair_campaign execute \
  --mode full --expected-commit "$PINNED_COMMIT" \
  --baseline-dir /content/drive/MyDrive/gridcast/reference_20261012 \
  --persistent-root /content/drive/MyDrive/gridcast/campaigns \
  --authorize-historical-search --authorize-primary-evaluation \
  --accept-budget-risk
```

Development-only execution omits `--authorize-primary-evaluation` and selects
`--mode development`; it still requires separate historical-search approval
and the resource-budget decision. Neither path resumes an old run. The direct
historical Linux worker entry point rejects invocation outside this supervisor
using a parent-PID handshake; this is accidental-launch protection, not a
security boundary against a person deliberately changing environment/code.

## Enforcement and persistence

The parent watches child process-tree RSS and the stage journal every 0.25 s,
including native operations which may delay Python signal handlers. It kills
the process group at **2,048 MiB**, **3,600 s campaign time**, or **1,800 s for
either model's development search**. Model clocks span all horizons/candidates/
blocks for that model and cannot be reset by stage transitions. Journal
transitions shorter than one polling interval remain recorded for checking.
No 30-second extension is added to the campaign. Startup, loading, fitting and
worker finalization are inside the supervised lifetime; post-exit integrity
inspection/export is separately outside the fitting campaign budget.

The worker retains its internal guards and checks system admission before
historical fitting blocks and primary fits. CPU and BLAS threads remain two.
Missing RSS or unreadable deadline evidence fails safely. Polling is not an
OS memory reservation: sub-interval spikes may be missed, and container OOM
can act first. Parent termination also cannot run cleanup if Colab kills the
entire VM; already-published persistent artifacts are the recovery evidence.
Drive/FUSE flush semantics are provider-dependent. File fsync is required;
directory fsync is attempted where the filesystem supports it. The October 23 no-fit Colab preflight passed atomic file publication, file
fsync and readback in the selected Drive export directory. Linux CI passed
the launch guard. These checks do not prove survival of provider termination
or completion of the full training campaign.

Each fresh `fair_campaign_<UTC>/` contains:

- Parent preflight, supervisor record, unbuffered worker stdout/stderr logs.
- `experiment/fit_artifacts/search/<identity>/`: per-fit model, predictions,
  measurement and final SHA-256 completion manifest.
- `experiment/fit_artifacts/primary/<identity>/`: per-fit predictions and
  measurement referencing the existing top-level final-model file/hash.
- The original combined candidate predictions, scores, frozen selection,
  primary predictions, paired metrics/bootstrap and original-reference tables.
- Flushed fit/stage ledgers and terminal run status when the worker can write it.
- `complete.json` published last, only after exact planned artifact identities,
  fit ledger, final files and measurements pass inspection.

Per-fit bundles are immutable; a missing/invalid completion manifest means an
incomplete bundle. Atomic same-directory publication avoids exposing partially
written checkpoints as completed files. Persistence failures abort instead of
disqualifying a candidate and silently selecting another model. No automatic
retry, smaller grid or larger limit is available. Search-model checkpoints add
storage and serialization time relative to the old runner; the old output-size
planning estimate is not a measured bound for these expanded artifacts.

## Inspection, interruption and export

```sh
.venv/bin/python -m gridcast.experiments.fair_campaign inspect --run-dir "$RUN_DIR"
.venv/bin/python -m gridcast.experiments.fair_campaign export \
  --run-dir "$RUN_DIR" --destination "$EXPORT_DIR"
```

Inspection reports saved fits and active interrupted attempts even when terminal
metadata is missing. COMPLETE requires the exact 204 search fits (and 32 primary
fits for full mode), valid bundles/primary models, confirmed fit ledger,
required final evidence, limits and the campaign completion marker. Failed or
skipped candidates remain visible but never produce a COMPLETE campaign label.
The runner's original candidate-failure scoring policy is retained; stricter
completion reporting does not change selection or model calculations.

The append-only ZIP includes lightweight provenance, selection, measurements,
metrics, ledgers, warnings and hashes. Model binaries and prediction tables are
excluded by default; they are already retained privately in the persistent
campaign directory. Keep that directory, not only the ZIP. A disconnection is
not permission to restart or reuse its path. Review partial status and request
a new execution decision if a separate attempt is necessary.

## Validation

Focused tests use synthetic data, mocked fit outputs, temporary files and tiny
sleeping/printing workers. They exercise authorization, missing/hash/schema
inputs, atomic publication, persistence failures, each external stop condition,
stage-clock reset rejection, interruption reporting, exact 204-fit identities,
completion-marker corruption, append-only export and preservation of pilot
restrictions. No historical fitting test is included. See the final preparation
record for test counts and platform-specific skips.

Local preparation checks on 23 October 2026: **82 passed, 1 Linux-only test
skipped on macOS, 2 numerical model-fitting fixtures deselected**. The tiny
subprocess tests and 204-identity artifact fixture perform no model fits.
Repository-wide Ruff, changed-file formatting, compilation and offline uv
lock checking passed. Four saved evidence packages and the completed interval
package passed read-only integrity checks. The nine core scientific functions
(grid/validation, shared design, features, fitting mask, schedule, estimator,
fit/predict and selection) are AST-identical to the reviewed parent version.
Approved dataset and all reference-file hashes/metadata verify locally.
The wider training-containing suite was not rerun on the shared Mac.

Linux GitHub CI at source `3c0ea3dd4eb73b7df976b65255f837294b7912c4`
passed: **204 tests plus 5 deeper integrity tests**. The Linux launch guard
passed there. [CI evidence](https://github.com/AdebanjiAdelowo/electricity-demand-forecasting/actions/runs/39947981124).

## Live Colab preparation evidence — 23 October 2026

A clean checkout of source `3c0ea3dd4eb73b7df976b65255f837294b7912c4`
completed the no-fit full-mode preflight on free Colab CPU: **CONDITIONAL**,
no hard failures and **zero additional fits**. Python 3.12.3, core frozen
dependencies, dataset hash/provenance, all three reference hashes and metadata,
CPU-only settings and two threads passed. Available host RAM was 11.30 GiB,
preflight process RSS 242.5 MiB and free VM disk 86.48 GiB; these are preflight
measurements, not training peaks or guaranteed container capacity.

The cgroup limit was effectively unlimited; no usable container ceiling was
exposed. This caveat and the unchanged LightGBM 30-minute budget risk remain
explicit. Atomic Drive publication, file fsync and readback passed. The JSON
report and write probe were verified present privately in Drive.

[Preflight JSON](https://drive.google.com/file/d/1LOH7ny3k0_Mf2b0PYHB6RZhzlwdlaLFs/view)
is saved as `gridcast/exports/campaign_preflight_20261023T164856224063Z.json`.
The six-fit pilot was not repeated, and no historical search or primary
evaluation was launched. The next step remains a separate execution and
resource-budget decision; preparation is not authorization.
