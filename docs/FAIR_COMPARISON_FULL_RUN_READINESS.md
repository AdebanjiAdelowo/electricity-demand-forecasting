# Full fair-comparison resource and execution review

Review date: 23 October 2026. Reviewed source: `64f53fbbe38d1f041024f24f400a63d4daf563c3`;
experiment implementation remains identical to pilot commit `32be70b`.

## Decision

**BLOCKED for full Colab execution in the current implementation.** The
completed six-fit pilot remains valid and successful. Its measured costs
justify preparing the larger experiment, but do not establish compliance
with the full experiment's 30-minute per-model search cap, 60-minute campaign
cap, or 2 GiB memory ceiling. No further fit or primary evaluation ran during
this review. The frozen scientific grid, cutoffs and limits remain unchanged.

Required work is execution infrastructure and evidence durability, rather
than a scientific redesign. Do not automatically enlarge limits or shorten
the training span. Historical search and primary refits remain separately
authorized stages; this review does not launch them.

## Measured basis and explicit assumptions

The [completed pilot report](GRIDCAST_COLAB_SIX_FIT_PILOT_REPORT.md) records
68.655 s internal wall time, 71.765 s supervised lifetime and 657.176 MiB
process high-water RSS, with all six fits complete on a two-thread Colab CPU.
These are the only new historical fitting measurements used here.

For a conservative, reproducible proxy, take the slower of the two measured
horizons for each pilot candidate:

| Component | Measured seconds used |
|---|---:|
| Ridge alpha 1 fit | 1.919670107 |
| Profile A, L2/600 trees/63 leaves fit | 18.127194754 |
| Profile B, L1/300 trees/63 leaves fit | 12.731908683 |
| Feature construction per fit, maximum of all six | 1.639761373 |

The frozen development grid requires 60 Ridge and 144 LightGBM fits, including
72 fits per LightGBM profile. Final refits require 16 per model. This proxy
uses the 63-leaf pilot timings for all leaf sizes and uses L2 profile-A timing
as a placeholder for its unmeasured L1 objective. It is a sensitivity model,
not a statistical upper bound: smaller leaf counts may be faster; 600-tree
L1, longer spans, other horizons or system load may be slower.

Let R=1.919670107, A=18.127194754, B=12.731908683, F=1.639761373 seconds,
and let k multiply only LightGBM fitting time:

- Ridge search: `60 * (R + F)`.
- LightGBM search: `72 * (A + B) * k + 144 * F`.
- Final refits proxy: `16 * (R + max(A, B) * k) + 32 * F`.
- Campaign subtotal: sum of those three components.

| LightGBM fit multiplier k | Ridge search min | LightGBM search min | Final refits min | Campaign subtotal min |
|---:|---:|---:|---:|---:|
| 0.50 | 3.56 | 22.45 | 3.80 | 29.81 |
| 0.75 | 3.56 | 31.71 | 5.01 | 40.28 |
| 1.00 | 3.56 | 40.97 | 6.22 | 50.75 |
| 1.50 | 3.56 | 59.48 | 8.64 | 71.68 |

Every row excludes prediction, metric/bootstrap computation, artifact writes,
transfer and any extra span effects. None is a measured full-run duration.
The k=1 proxy already exceeds the LightGBM 30-minute search cap; even k=.75
does so. This flags a credible admission risk, not proof that the actual
smaller-leaf grid will exceed its cap. An average of 12.5 seconds per
LightGBM fit, including construction/prediction and search overhead, exhausts
1,800 seconds across 144 fits. No cap increase is justified by this review.

## Memory and storage

The exact development schedule contains 436,509 candidate prediction rows
before imputed-target exclusions: `17 * sum(block_hours - h)` over block
hours 2,208/2,208/2,184 and horizons 1/24/48/168. These rows are retained in
`search()` across candidate fits, then concatenated for parquet output.

The pilot's short schedules do not measure this accumulated retention.
Object/string columns, concatenation copies, feature matrices, imputation
and scaling copies, native estimators, final predictions and bootstrap
arrays contribute separately. The observed 657 MiB is not a full-run bound;
compressed parquet size is not an in-memory allocation estimate.

The 32 model artifacts may also be larger than pilot artifacts because final
fitting spans and selected candidates vary. The older 50–300 MiB output
range remains a planning estimate. Original comparison inputs needed on
Colab are locally present but ignored by Git: `run.json` (10,424 bytes),
`ridge.parquet` (1,953,098 bytes), `lightgbm.parquet` (11,444,514 bytes), about
12.79 MiB combined. They must be transferred privately and fingerprinted
before launch; do not regenerate, omit or silently substitute them.

## Execution gaps found in source inspection

1. **Fresh-checkout input failure.** `fair_model_comparison.run()` still reads
   the approved processed hash from the ignored baseline `run.json`. The
   pilot uses the committed approval file, but the full runner does not.
   A fresh bundle checkout lacks the baseline directory.
2. **Late reference-input failure.** After final fitting and paired scoring,
   `evaluate()` reads original Ridge and LightGBM baseline parquets for the
   reference comparison. Their absence can fail a campaign after its 236
   fits. All required inputs need preflight verification before any fit.
3. **External enforcement is pilot-only.** `colab_pilot.assert_pilot_only()`
   deliberately rejects the full runner. Its process-tree supervisor cannot
   be reused by passing the full-search command. The full runner's Python
   SIGALRM checks may wait during native calls, so they do not independently
   enforce an immediate RSS or wall-time limit. Per-model checks also occur
   at stage boundaries rather than through an external model-stage watchdog.
4. **Incomplete campaign coverage.** The full runner loads and checks inputs
   before entering `budget()`. The timer starts when that context is entered,
   rather than at process launch. It therefore does not bound all startup or
   input-loading time, even though terminal wall time starts earlier.
5. **Durability gaps.** Development scores and fit records are saved after
   each candidate, but complete candidate predictions and `selection.json`
   are saved only at the end of search. Final models are saved individually,
   while final predictions and `evaluation_fits.csv` are written later.
   A killed VM can lose completed-fit evidence or the whole local run.
6. **Export coverage differs.** The pilot ZIP allowlist excludes full-run
   tables, selection, evaluation evidence and model artifacts. It cannot be
   treated as an adequate full-campaign preservation mechanism.

## Concrete preparation required before launch

- Preflight the committed dataset approval and privately supplied immutable
  baseline record/parquets, including hashes, schemas and expected keys.
  Run preflight before fitting and retain a failure record on rejection.
- Add a separate full-run supervisor with explicit stage authorization,
  CPU/thread checks, process-tree RSS enforcement at 2,048 MiB, and campaign
  and model-stage deadlines. Keep the pilot dispatch restriction intact.
  Define whether startup/finalization count within the existing campaign cap
  without silently adding time or relaxing the scientific limits.
- Persist per-fit measurements, predictions, status and selection provenance
  durably at completed-stage boundaries. Use fresh directories and an atomic
  completion manifest. A disconnect must leave incomplete status, not trigger
  automatic retraining or an altered grid.
- Preserve the required full evidence privately in Drive, with hashes and a
  separate lightweight report. Do not publish raw demand/prediction tables
  or model binaries into Git. Test missing inputs, native-worker timeout,
  RSS rejection, partial stages and export round trips without historical fits.
- Reassess the 30-minute LightGBM cap using existing measurements and any
  separately authorized additional profiling. If still incompatible, stop
  for an explicit protocol decision. Do not drop candidates or raise a limit
  merely to obtain completion.

Acceptance requires verified inputs, passing synthetic/no-fit safeguard tests,
external controls covering native execution, durable partial/completed
artifacts, an explicit resource-budget decision, and separate authorization
for historical search and primary evaluation. No scientific accuracy claim
or winning-model selection follows from resource feasibility.

## Preservation and next decision

This review reads the already exported audit and source code; it does not
fit models, decode primary labels, compute new accuracy metrics or change
frozen settings. The original pilot notebook and artifacts remain intact.
The next work item is the bounded full-run infrastructure preparation above.
The six-fit pilot should not be repeated. Full execution remains pending
until the concrete gaps and budget decision are resolved.
