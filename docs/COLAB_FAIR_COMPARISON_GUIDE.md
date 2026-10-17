# Colab CPU guide: fair-comparison resource pilot

## Status

**COLAB PREPARATION COMPLETE; HISTORICAL PILOT NOT EXECUTED.** The notebook
`notebooks/gridcast_fair_comparison_colab.ipynb` prepares and inspects a free
Colab CPU runtime for the six-fit resource pilot described in
[the resource-pilot record](FAIR_MODEL_COMPARISON_RESOURCE_PILOT.md). No real-data
fit, historical search, selection or primary-test evaluation has run on Colab
or anywhere else as part of this preparation. Running the pilot needs separate
explicit approval. The full 204-fit search and 32 primary refits have no
route through this notebook at all.

The scientific protocol is unchanged: same candidates (`r01`, `l02`, `l11`),
horizons 1 and 168 h, fitting start 2015-08-01, target cutoff 2023-07-01,
validation origins 2023-07-01 to 2023-07-15, no-weather shared feature matrix,
fitting-only imputation/scaling, imputed-target exclusion, seed 0, at most two
threads, 2,048 MiB memory ceiling and 300 s pilot wall time.
`configs/fair_model_comparison.json` is byte-identical to the reviewed version
(SHA-256 `5418e81f…5d09`). Preflight blocks if it differs.

## 1. Open the notebook in Colab

The repository is private. Either:

* In Colab choose **File → Open notebook → GitHub**, authorize GitHub access,
  pick `AdebanjiAdelowo/electricity-demand-forecasting` and open
  `notebooks/gridcast_fair_comparison_colab.ipynb`; or
* download the notebook file and use **File → Upload notebook**.

## 2. Select a free CPU runtime

**Runtime → Change runtime type → Hardware accelerator: CPU** (the default
free tier). Do not choose a GPU or TPU. Step 2 stops with BLOCKED if
`/dev/nvidia*`, `/dev/accel0`, `COLAB_GPU` or a TPU variable is present. No paid
add-on or background execution is used. The free CPU runtime usually has two
vCPUs, which matches the two-thread cap, but this is checked rather than assumed.

## 3. Get the code and dataset

Step 3 checks out an exact commit (`PINNED_COMMIT`) and blocks unless `HEAD`
equals it and the working tree is clean. The approved processed snapshot
(`data/processed/ercot_hourly.parquet`, 15 MB, plus `manifest.json`) is tracked
in Git, so the checkout contains the dataset. No public download URL is
assumed and no large transfer occurs.

**Option A: Git bundle (recommended).** On your machine:

```sh
cd electricity-demand-forecasting
git bundle create gridcast.bundle main
```

Upload `gridcast.bundle` (about 13 MB at this commit) to a Drive folder you choose. In Step 3
set `REPO_SOURCE = "bundle"`, `MOUNT_DRIVE = True` and `REPO_BUNDLE_PATH` to its
path. The notebook ships with empty paths; nothing private is embedded.

**Option B: GitHub with a Colab Secret.** Create a read-only, single-repository
fine-grained token, add it in Colab's **Secrets** panel (key icon) as
`GITHUB_TOKEN`, grant the notebook access and set `REPO_SOURCE = "github"`. The
token is sent as a transient Git header via environment variables. It is never
printed, stored in `.git/config` or written to the notebook.

**Option C: user-provided dataset folder.** Use this only if your checkout lacks
`data/processed`. Set `DATASET_SOURCE_DIR` in Step 5 to a folder that holds
exactly `ercot_hourly.parquet` and `manifest.json`. The files are symbolic-linked,
not copied, and only after their SHA-256 matches the approval. Existing
different files are never replaced.

### Verification (Step 5), fail closed

`configs/approved_dataset_snapshot.json` holds the approved fingerprints
(processed `1e3acfa6…a5ab`, manifest `b8811071…34de`, dataset version
`eia930-TEX+openmeteo@2026-10-10`). When the ignored local benchmark record
exists, the two must agree. Checks:

* SHA-256 of both files;
* Parquet schema: `demand: double`, `is_imputed: bool`, index
  `utc_end: timestamp[us, tz=UTC]`, 97,914 rows;
* coverage 2015-07-01 06:00 to 2026-08-31 23:00 UTC from Parquet statistics,
  and no demand nulls;
* manifest version/rows/span/raw-EIA hash, UTC hour-ending index, licences and
  72 imputed values.

Schema checks read metadata only. No demand values are decoded before an
authorized pilot. Any mismatch stops the notebook before training.

## 4. Dependencies

Step 4 installs `uv==0.11.16` into the disposable Colab VM. It then runs
`uv sync --frozen --python 3.12` to create the project `.venv` from the
committed `uv.lock`, with **core dependencies only** (no torch/CUDA, serving or
copilot extras). Preflight blocks if installed numpy, pandas, pyarrow, scipy,
scikit-learn, LightGBM, threadpoolctl or statsmodels versions are not in the
lockfile. If Colab's interpreter is outside 3.11–3.12, uv fetches a managed
Python 3.12. Expect roughly 0.5–1 GB of VM disk and a few minutes.

## 5. Preflight (action A)

Step 6 runs `python -m gridcast.experiments.colab_pilot preflight
--expected-commit <sha>`. It writes a new
`results/fair_model_comparison/colab_preflight_<UTC stamp>/preflight.json`
recording:

* Git commit and dirty state;
* Python and package versions and the lockfile hash;
* dataset fingerprints and problems;
* runtime type, CPU model, logical CPUs, affinity and cgroup CPU quota;
* MemTotal, MemAvailable, the cgroup memory limit and usage, process RSS and
  pressure-stall averages;
* disk free, thread variables, the fixed pilot plan and the hard limits.

It never decodes labels or builds a model. Step 7 prints the fixed plan.

### Linux resource admission

Implemented in `src/gridcast/resources.py`. psutil is not used: it is only in
the lock through the optional `foundation` extra.

| Check | Rule |
|---|---|
| `/proc/meminfo` `MemAvailable` | Required; missing → BLOCKED |
| cgroup v2 `memory.max`/`memory.current` (smallest along the cgroup ancestry) or v1 `memory.limit_in_bytes`/`usage_in_bytes` | Required; unreadable → BLOCKED. No limit → CONDITIONAL, host value binds |
| Effective available = min(MemAvailable, limit − usage) | Must be ≥ (2,048 MiB − current RSS) + 1,024 MiB margin |
| `/proc/pressure/memory` `some avg10` | ≥ 10 → BLOCKED; unavailable → CONDITIONAL |
| Process RSS (`/proc/self/status`) | Required |
| Disk free at results | ≥ 1 GiB |
| Accelerators | Any → BLOCKED |
| Usable CPUs (min of count, affinity, quota) | < 1 → BLOCKED; < 2 → CONDITIONAL |
| `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS`, `GRIDCAST_JOBS` | Must be set and ≤ 2 |

A large reported host RAM never admits a run against a smaller container limit.
The 2,048 MiB ceiling is not raised because Colab has more memory. The pilot
repeats this admission before **every** fit. The margins are conservative
engineering choices, not measured requirements.

On macOS the existing `kern.memorystatus_vm_pressure_level` dispatch gate is
unchanged: only normal (1) admits. Other platforms are blocked.

## 6. Authorize the six-fit pilot later (action B)

Only after written approval, set `AUTHORIZE_SIX_FIT_PILOT = True` in Step 8 and
run it. Equivalent command inside the checkout:

```sh
.venv/bin/python -m gridcast.experiments.colab_pilot pilot \
  --authorize-six-fit-pilot --expected-commit <PINNED_COMMIT>
```

It first repeats preflight; if BLOCKED, nothing is loaded or fitted. It then
launches only `gridcast.experiments.fair_resource_pilot --execute-pilot --output
<new dir>` as a child with two threads and `CUDA_VISIBLE_DEVICES=""`. Any
command containing `--mode`, `full`, `development` or the historical/primary
authorization flags is refused, and the CLI has no full-search action.

**Enforcement.** A supervising parent polls the child process-tree RSS from
`/proc` every 0.25 s. It sends SIGKILL to the process group at **2,048 MiB** or
**330 s** (the 300 s pilot cap plus 30 s to write terminal metadata). The
pilot's own guards remain: 1,843 MiB early stop and 300 s, checked once per
second and between stages. The child's `oom_score_adj` is raised so the
container OOM killer chooses it before the notebook kernel. Limitations:

* a spike shorter than one poll can exceed the ceiling briefly;
* Colab's own container OOM killer may act first;
* no cgroup or systemd reservation is available inside Colab;
* `RLIMIT_AS` is not used, because allocator and thread address-space
  reservations make it an unreliable RSS proxy.

A run that exceeds 300 s or 2,048 MiB is reported INCOMPLETE even if it
finishes.

## 7. Inspect measurements

Step 9 runs `colab_pilot inspect <run dir>` for each
`resource_pilot_colab_*` directory and shows, per fit: training rows, feature
count, feature-preparation, fit and prediction seconds, cumulative process
high-water RSS, serialized model bytes and total attempt time. The run is
**COMPLETE** only if:

* `run.json` status is `completed`;
* all six `(candidate, horizon)` fits are measured;
* the supervisor exited 0 without a kill or user interrupt;
* wall time ≤ 300 s and peak RSS ≤ 2,048 MiB.

Anything else is **INCOMPLETE: not a completed pilot**, listing measured,
failed and started-but-unfinished fits. No MAE, selection or winner is
produced.

## 8. Export results (action C)

Set `EXPORT_DIR` in Step 10 to an existing persistent folder you choose (for
example in mounted Drive), or run:

```sh
.venv/bin/python -m gridcast.experiments.colab_pilot export <run dir> --destination <folder>
```

Each export is a new `<run>_provenance.zip`. It contains:

* `preflight.json`, `run.json`, `pilot_plan.json` and `configuration.json`;
* `measurements.csv` and `fit_events.jsonl`;
* `active_attempt.json`, `supervisor.json` and `inspection.json`;
* an `export_manifest.json` with SHA-256 for every member.

Fitted models (`*.pkl`) and prediction tables (`*.parquet`, which contain
observed demand) are **excluded**. An existing archive is never overwritten,
and writing goes through a `.partial` file that is renamed at the end.

## 9. Colab disconnections

* All run directories are new and append-only (`exist_ok=False`). There is no
  resume option, so a reconnect cannot silently continue with another
  configuration.
* Measurements, models, predictions and the `fit_events.jsonl` ledger are
  written after each fit, and `active_attempt.json` names the fit in progress.
* If the VM survives, rerun Steps 3–9. Step 3 refuses a checkout at a
  different commit or with changes, and Step 9 lists earlier runs with their
  labels. A killed run has no terminal `run.json` and is reported INCOMPLETE,
  with the unfinished fit.
* If the VM is recycled, files not exported are lost. Export promptly. A new
  attempt is a new run under the same approval; never merge partial runs into
  a "completed" pilot.
* A notebook interrupt is forwarded to the child as SIGINT, so the pilot
  records `interrupted`. If the child does not stop within 30 s, it is killed.

## 10. READY, CONDITIONAL and BLOCKED

* **READY**: every hard check passed (commit, clean tree, lockfile versions,
  reviewed configuration, dataset fingerprint/schema/provenance, memory,
  disk, CPU-only, threads).
* **CONDITIONAL**: hard checks passed with recorded caveats, such as no
  pressure-stall file, no cgroup limit (host memory binds) or fewer than two
  usable CPUs. A pilot may be attempted under approval, and the caveats must be
  reported with its measurements.
* **BLOCKED**: at least one hard check failed. Do not run the pilot. Fix the
  cause (runtime type, dataset, commit) or wait. Never weaken a threshold.

None of these verdicts is authorization. After a completed pilot, the
feasibility decision for the full campaign is still a separate written
READY/CONDITIONAL/BLOCKED assessment.

## 11. How Colab differs from the local Mac

| Aspect | Local Mac | Colab CPU |
|---|---|---|
| Admission sensor | Dispatch memory pressure (normal only) | `/proc/meminfo`, cgroup limit and usage, pressure-stall data |
| Memory enforcement | In-process RSS checks only | Same, plus supervising-parent SIGKILL at 2,048 MiB / 330 s |
| `ru_maxrss` units | Bytes | KiB (converted by `peak_rss`) |
| CPU | Apple Silicon, arm64 | Shared x86-64 vCPUs, usually two |
| Python/BLAS/LightGBM wheels | macOS arm64 builds from the same lock | manylinux x86-64 builds from the same lock |
| Other load | Parallel research sessions | Shared cloud host, noisy neighbours |

Numerical results are **not** claimed bitwise identical across macOS and Linux.
BLAS kernels, compiler builds, LightGBM threading and floating-point reductions
differ. Timings are hardware-specific: Colab measurements describe Colab and
must not be presented as Mac measurements, or the reverse.

## Colab compatibility notes

* Colab's image may change its default Python. The project requires 3.11–3.12,
  and uv supplies 3.12 if needed.
* SIGALRM-based in-process checks work on Linux. `os.sched_getaffinity` and
  `/proc` exist on Colab but not on macOS; tests use fake `/proc` and `/sys`
  trees.
* The notebook kernel needs only the standard library (plus pandas, which
  Colab provides, to display Step 9). All GridCast code runs in the locked
  `.venv`.
* None of this has been executed on a live Colab runtime yet. Locally, the
  bundle checkout at the pinned commit was simulated and its dataset
  verified against the committed approval without the ignored benchmark
  record. The first real Colab preflight is the next verification step.

## Exact next action, after separate approval

1. Open the notebook on a free CPU runtime. `PINNED_COMMIT` defaults to
   `32be70b38d2f75745d4fb69a7cc8e4083ec08d3b`, the commit containing this
   preparation code. A later reviewed commit may be pinned instead.
2. Run Steps 2–7 and record the preflight verdict.
3. If READY or CONDITIONAL and pilot execution is approved in writing, set
   `AUTHORIZE_SIX_FIT_PILOT = True` and run Steps 8–11. Export to Drive.
4. Write the measured full-run feasibility assessment before any request to run
   the historical search.
