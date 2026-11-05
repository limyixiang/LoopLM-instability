# Instability reproduction and scale profiling

This patch supplies a one-GPU-per-run harness for testing whether increasing training
depth from four to eight loops produces repeatable instability. It does not
establish that instability occurs or that a gate intervention repairs it.

**Current result and next experiment, 5 November 2026:** the initial 1.4B-size
T8/context-512/microbatch-1 profile completed on an H100 NVL MIG 3g.47gb allocation.
Peak allocated/reserved memory was 27.16/28.79 GiB; four measured updates gave
662 target tokens/s. This establishes fit for that short configuration only.
The next check profiles microbatches 2/4 at T4/T8 using [arrays capped at two tasks](#run-arrays-on-two-47-gb-allocations).
The four 118M screens had no predefined instability events. The later seed-101
screenshot looks stable through 3,072 updates, but its raw logs remain unaudited.

## Implementation and reference

`small_repro` has six shared layers, width 768, 12 heads, FFN width 2,048,
49,152 vocabulary entries, and 117,984,769 parameters. It uses an untied LM head
and normalizes each loop's final state before the gate, readout, and next loop.
The joint objective remains the weighted sum of per-exit cross-entropies minus
gate entropy; every loop receives full recurrent backpropagation.

The architecture reference is [ByteDance/Ouro-1.4B at revision
574fa66cb8bf5abdc979642d01cf2b79b16bfab1](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1).
The reduced dimensions, context, batch, and token horizon are explicit pilot
adaptations. Released code is an architecture reference, not certification of
the original large-scale pretraining recipe. The reference model's default
`labels` path is not used as our training objective.

The legacy `small` preset remains tied and leaves the recurrent boundary
unnormalized. Old checkpoints can still instantiate their architecture, but shared
FP32 norm/probability reductions can change older BF16 numerical trajectories.
Do not resume a legacy checkpoint into `small_repro` and label it an unchanged run.

## Prepare before reserving the GPU

Save the cluster settings once, from the repository root:

```bash
if [ ! -f cluster.env ]; then cp cluster.env.example cluster.env; fi
nano cluster.env  # or use your preferred editor
```

The defaults match your working cluster setup: partition `gpu-long`, one
`gpu:h100-47:1`, 72 requested hours for reproduction, 8 CPUs, and 96G of host RAM.
The separate profile job requests two hours and defaults to a one-hour total
measurement budget, with a 30-minute timeout per case. The batch jobs send
BEGIN/END/FAIL notifications to `e1121685@u.nus.edu`. The user reports a 72-hour
limit per SLURM request and access to two one-GPU allocations concurrently.
This is a per-task wall-time limit, not a statement of a total GPU-hour allowance.
The 96G request is CPU host RAM; the GPU allocation exposes approximately 47 GB.
Check `PILOT_STORAGE`; the interpreter, cache, data, and output paths derive
from it and the repository location. `PILOT_ACCOUNT` is optional and empty by
default: the wrapper omits `--account` and lets SLURM use the site's default.
Set an account only if required for your allocation. If an older `cluster.env`
still contains `YOUR_ACCOUNT` or `YOUR_H100_PARTITION`, replace the account with
an empty value and the partition with `gpu-long`. For a site or
Conda environment, set `PYTHON_BIN` to that environment's absolute Python path.
Keep `cluster.env` on shared storage with the repository. It is ignored by Git
so your local paths are not committed. The downloader, training launcher, and
submission wrapper load it automatically on each invocation, including after
reconnecting. Existing environment values and inline overrides take precedence
over the example's saved defaults. Use `CLUSTER_ENV_FILE=/absolute/path/to/file`
to select another file. You do not need to add project settings to `.bashrc`.

The cluster does not need `uv`. Use Python 3.10–3.12 (3.11 is a reasonable
starting choice), `venv`, and pip. Load your site's Python module first if
required; module names are cluster-specific. Reuse your existing `.venv` on a
shared filesystem visible at the same path on the login and compute nodes.
Create it only if this checkout does not already have one:

```bash
cd /path/to/ouro-instability
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip

# Example CUDA 12.6 wheel; use the build supported by your cluster's driver.
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-reproduction.txt
python -m pip check
python -m pip freeze > .venv/requirements-resolved.txt
python -m pytest tests/test_reproduction.py tests/test_corpus_preparation.py -q
python scripts/verify_reproduction_tokenizer.py

# cluster.env defaults to this environment's absolute interpreter path.
```

Select the PyTorch wheel using the cluster's instructions and the
[official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2100).
If the site already supplies a working CUDA-enabled PyTorch module or Conda
environment, use it instead and install the core requirements there. To expose
module-provided packages in a new venv, create it with `--system-site-packages`.
Do not replace a working site build solely to match the example. Keep the same
environment for the matched runs and record any differences from the CPU-checked
Torch 2.10.0 environment. `requirements-reproduction.txt` pins the core packages;
it does not replace the full transitive `uv.lock`. Keep the resolved pip freeze
with your results. Optional legacy wandb/evaluation paths additionally need
`python -m pip install wandb lm-eval`.

Within a GPU allocation, check CUDA before starting the screen:

```bash
source cluster.env  # only needed here for direct Python commands in this shell
"$PYTHON_BIN" -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_bf16_supported()'
```

The released-code reference check needs a **separate** environment with
Transformers 4.55.0. Keep the main training environment on 5.2.0. This small CPU
check can run on a preparation node:

```bash
python3 -m venv .uv/reference-venv
.uv/reference-venv/bin/python -m pip install --upgrade pip
.uv/reference-venv/bin/python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
.uv/reference-venv/bin/python -m pip install 'numpy==2.2.6' 'transformers==4.55.0'
.uv/reference-venv/bin/python scripts/verify_ouro_reference.py --loops 8
```

The reference check downloads only pinned model code/configuration, loads that
code explicitly, and compares a tiny copied-weight FP32 model's per-loop outputs
and joint-loss gradients. It writes `.uv/ouro-reference/verification.json` and can
be repeated with `--offline` once cached. This is a correctness check, not a GPU
training experiment. Exact CPU resume is tested; verify replay agreement again
on the actual GPU/software stack. `--deterministic` requests strict PyTorch
determinism and may reject a kernel that cannot provide it.

The published reference imports APIs from Transformers 4.55.0 that are absent
from the project's locked 5.2.0. The separate environment isolates this checker
dependency. The local reproduction trainer does not load the released model
implementation at runtime. The `.uv/` directory here is only a cache location;
no `uv` executable is involved.

The primary corpus is a fixed selection from FineWeb-Edu `sample-10BT` at revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. On a node with download access:

```bash
bash scripts/download_artifacts.sh
```

FineWeb-Edu has a training split; the preparation command constructs our own
train/validation/test sets. For each nonempty source document, it hashes the exact
UTF-8 text, removes repeated text and IDs, and assigns the document using
`SHA256(seed:split:text_hash)` before tokenization or packing. The default seed is
42 and held-out fractions are 1% validation and 1% test. Duplicate text cannot
cross splits even if it has a different source ID. Source-order traversal of the
pinned, already sampled config selects the first eligible documents until each
target is reached; this is a reproducible selection, not a claim of an unbiased
sample from the entire FineWeb-Edu corpus.

Targets default to 250M training and 2M tokens per held-out split, measured with
the selected tokenizer. Complete documents and their EOS separators are retained,
so a split can exceed its token target by its final document. The source traversal
is capped at two million records. An exhausted source or cap produces an
`incomplete` manifest and a nonzero exit; the loader rejects that preparation.
Do not silently reduce only one experiment's data or token horizon. The
`sample-10BT` name describes the upstream source pool; the command does not
download or tokenize all 10B tokens. Streaming can still fetch whole source
Parquet files as needed, so network bytes are not identical to selected tokens.

`manifest.json` records immutable source/tokenizer revisions, selection settings,
actual token and packed-target counts, shard hashes, and software versions. Each
split's `records.jsonl` records the selected source IDs, text hashes, source row
indices, and token offsets. Its checksum is validated at load time. Keep these
files with the `.npy` arrays when staging data. Run metadata includes this
preparation provenance; resume rejects a changed manifest as well as changed
token shards. Paired four/eight-loop batch order and replay are unchanged.

Preparation defaults to `ByteDance/Ouro-1.4B` at revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, using
`add_special_tokens=False` and appending one EOS token per document. No automatic
BOS is inserted. The tokenizer verifier compares the actual vocabularies,
backend behavior, special-token settings, files, and encoding probes against
pinned SmolLM2. They are **not equivalent**: special-token mappings differ despite
both having 49,152 entries. See [TOKENIZER_VERIFICATION.json](TOKENIZER_VERIFICATION.json).
Training checks the prepared tokenizer identity/revision, behavior fingerprint
across splits, and model vocabulary. An explicit alternative tokenizer is a
documented experimental change; equal vocabulary size is insufficient.

For an optional **pipeline smoke test only**, use the official raw WikiText rows:

```bash
REPRO_DATA_DIR="$PWD/data/smoke_wikitext_raw" \
  bash scripts/download_artifacts.sh --dataset Salesforce/wikitext
```

This selects `wikitext-103-raw-v1` at the pinned source revision and defaults to
1M/128K/128K token targets. It reserves validation/test rows before admitting
training rows and removes exact repeated rows across selected splits. It does
not reconstruct Wikipedia articles or guarantee article-level separation. The
manifest labels these source units `text_row`; smoke-test curves are not the
primary FineWeb-Edu reproduction result. Never change datasets between the
four-loop and eight-loop cells of a comparison.

The downloader wrapper now invokes this bounded preparation, caching tokenizer
assets and writing the prepared dataset at `$REPRO_DATA_DIR` or
`${SCRATCH}/looplm/data/reproduction_fineweb_edu`. It never preloads the full
FineWeb-Edu corpus. For the smoke option, pass `--dataset Salesforce/wikitext`
and set `REPRO_DATA_DIR` to a different destination.

FineWeb-Edu is also used for training/validation in
[Fully Looped Transformer, Section 5.1](https://arxiv.org/html/2605.18797v2#S5.SS1)
and for Parcae's Transformer comparisons in
[Section 5.1](https://arxiv.org/html/2604.12946v1#S5.SS1).
This supports the data choice; it does not reproduce Ouro's full mixture or
guarantee that its reported instability appears in this pilot.

Input directories contain sorted `.npy` integer arrays, either flat token streams
or `(N, sequence_length + 1)` packed sequences. Every shard is packed independently,
with discarded tails recorded. Batches are generated by an indexed single-process
stream whose ordering is independent of model RNG and validation. Each input and
target sequence has 512 positions in the normal pilot.

## Run arrays on two 47 GB allocations

`scripts/submit_array.sh` submits profile or training cells with **at most two
independent one-GPU tasks running concurrently**. The batch header uses the
supplied `gpu-long`, `gpu:h100-47:1`, one node/task, 8 CPUs, 96G host RAM, and
BEGIN/END/FAIL mail settings. Profile tasks request two hours; training tasks
request 72 hours by default. Use `--time=...` to request less where appropriate.
An array does not combine GPU memory or distribute one model across both GPUs.

Submission creates a persistent `manifest.json` with the exact task mapping and
frozen experiment settings. Each task executes exactly one cell. The normal
training launcher retains checkpoint discovery, skip-completed behavior and
existing output directories. Site modules load in the batch job; later edits to
`cluster.env` cannot change the selected experiment cell. Keep source code
unchanged while jobs are queued/running; each training invocation records its
source hash. Duplicate cells and concurrency above two are rejected.
`ARRAY_CONCURRENCY=1` is supported.

First inspect the next four short profiles without submitting:

```bash
PROFILE_CASES="512:2 512:4" PROFILE_DEPTHS="4 8" \
  bash scripts/submit_array.sh profile --dry-run
```

Submit after reviewing the map:

```bash
PROFILE_CASES="512:2 512:4" PROFILE_DEPTHS="4 8" \
  bash scripts/submit_array.sh profile
```

| Array index | Context | Microbatch | Loops |
| --- | --- | --- | --- |
| 0 | 512 | 2 | 4 |
| 1 | 512 | 2 | 8 |
| 2 | 512 | 4 | 4 |
| 3 | 512 | 4 | 8 |

The generated range is `0-3%2`. Each profile retains 16,384 targets/update, two
performance-warmup updates and four measured updates. The limit applies to this
array; avoid overlapping arrays if total concurrency must remain two. SLURM
decides task start order. `%2`, task-index selection and `%A_%a` log naming follow
the [official array documentation](https://slurm.schedmd.com/job_array.html).

Submission prints the manifest path and job ID. Profile files live under
`$PROFILE_ROOT/arrays/ID/task_INDEX/` (default: under `$OUTPUT_ROOT/profiles`).
Logs use `logs/%x_%A_%a.out` and `.err`, with distinct array job/task IDs.
Inspect scheduling using `squeue -r -j ARRAY_JOB_ID`.

After all tasks finish, combine profiles and project matched-depth costs:

```bash
source cluster.env
"$PYTHON_BIN" scripts/array_jobs.py collect --manifest /printed/array/directory/manifest.json
tar --exclude='*.pt' --exclude='*.tmp' -czf ouro-profile-array-review.tar.gz \
  -C /printed/array/directory .
```

The collector writes `array_profile_summary.json`, reports missing/failed tasks,
and computes paired costs only when both matched depths completed. Pair-cost
sums are aggregate GPU-hours. The separate parallel elapsed estimate takes the
slower task's duration, assuming both start together. Queue delays, node variation
and concurrent checkpoint I/O can increase it. Profile cells are not independent
stability replications.

After selecting a setting, use the same wrapper for full training. For example,
the **already fitted microbatch-1** 50M-token pair is:

```bash
MODE=reproduce MODEL_CONFIG=ouro_1_4b_repro SEEDS=101 LRS=3e-4 DEPTHS="4 8" \
  SEQ_LEN=512 MICRO_BATCH=1 GRAD_ACCUM=32 MAX_STEPS=3072 \
  bash scripts/submit_array.sh reproduce --dry-run
# Remove --dry-run after the profile/budget review selects this protocol.
```

Task 0 is T4 and task 1 is T8, each with its own 72-hour limit and one GPU.
Preserve the selected batch settings if they differ from this example. To extend
a finished 50M-token pair to 100M, repeat the selected command with
`MAX_STEPS=6144`; each cell automatically resumes its existing checkpoint. Let
the old array finish or cancel it before extending, so two tasks never write
the same run directory concurrently. `SEEDS="101 202"` creates four distinct
training cells but still caps concurrent tasks at two.

Two tasks can reduce pair elapsed time toward the slower task's duration. They
do not reduce aggregate GPU-hours: two allocations each running 22 hours consume
44 allocation-hours. A 72-hour request limit does not establish an aggregate
experimental budget. Keep comparisons bounded and account for replication,
replay, storage overhead and proposal finishing. Keep the planned training-pair
scratch allowance; populated checkpoint writes can overlap. No SLURM jobs were
submitted by local implementation/tests. Existing serial wrappers remain usable.

## Profile Ouro-1.4B-size training next

Use the new `ouro_1_4b_repro` preset, **not** the legacy `ouro_1_4b` preset.
The new preset has 24 shared layers, hidden width 2,048, 16 attention/KV heads,
FFN width 5,632, vocabulary 49,152, and **1,434,652,673 parameters**. It retains
the verified sandwich norms, loop-boundary RMSNorm (epsilon 1e-6), untied head,
FP32 norm/gate probability handling, matrix initialization N(0, 0.02), unit norm
scales, zero gate bias, BF16 autocast, clipping at 1.0, full recurrent backward,
and the original joint objective. No recurrent state is detached.

The tensor shapes follow the [pinned released configuration](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json).
The preset uses RoPE base **10,000** and a 4,096-position default, following the
early training settings in [Ouro v6, Table 1 and Section 4.3](https://arxiv.org/pdf/2510.25741v6).
The released checkpoint instead has a 65,536-position context and RoPE base
1,000,000 after later training. Profiling overrides the executed context,
starting at 512. This is from-scratch profiling; it downloads no model weights.
The source/code audit and tiny reference comparison check architectural behavior,
not equivalence to the unreleased original pretraining implementation.

FP32 parameters, gradients, and two Adam moments alone require about **21.38 GiB**
(16 bytes/parameter), before activations, autocast copies, logits, temporaries,
and CUDA/allocator overhead. A populated checkpoint's model plus Adam tensors
are about **17.2 GB / 16.03 GiB**. The initial 512/microbatch-1 case fitted with
27.16 GiB allocated and measured 662 targets/s; other settings remain unmeasured.
Have at least **25 GiB free scratch** for one default profile. The driver checks
for its estimated checkpoint size plus 2 GiB before each case; leave more space
for logs and filesystem variation. By default it deletes only the newly created
profile checkpoint after measuring its write; `--keep-checkpoint` retains it.
Small-run checkpoints and output paths are never used by this workflow.

Run these steps **in order**, after copying the updated repository to the cluster:

1. Reuse the existing environment, prepared FineWeb-Edu streams, and `cluster.env`.
   No new Python packages or `uv` installation are needed for profiling. From the
   repository root, inspect a plan without allocating a model or using a GPU:

   ```bash
   bash scripts/profile_reproduction.sh --dry-run
   ```

2. Submit only the initial eight-loop case:

   ```bash
   bash scripts/submit_profile.sh
   ```

   The default is context 512, microbatch 1, accumulation 32: **16,384 target
   tokens/update**. It runs two complete warmup updates and four measured updates
   (98,304 training targets total), an initial 8K-target validation, a final
   32K-target validation, and one final optimizer-populated checkpoint write.
   Validation counts round to full batches. These six updates are a performance
   measurement, not an independent instability experiment. The two performance
   warmup updates are separate from the unchanged 200-update LR warmup.
   The wrapper loads saved settings and omits `--account` when the saved account
   is empty. Resource overrides work as before, e.g. `--time=01:00:00`.

3. Review the finished profile before submitting anything larger. Locate its
   summary and package the small records for analysis:

   ```bash
   source cluster.env
   find "${PROFILE_ROOT:-$OUTPUT_ROOT/profiles}/ouro_1_4b_repro" -name profile_summary.json -print
   tar --exclude='*.pt' --exclude='*.tmp' -czf ouro-1.4b-profile-review.tar.gz \
     -C "${PROFILE_ROOT:-$OUTPUT_ROOT/profiles}" ouro_1_4b_repro
   ```

   Outputs are unique timestamp/ID directories under
   `$OUTPUT_ROOT/profiles/ouro_1_4b_repro/` unless `PROFILE_ROOT` is set. Each case
   has the usual `run.json`, `metrics.jsonl`, `result.json`, plus `profile.json`,
   `profile_measurements.jsonl`, `console.log`, and `case_status.json`.
   `profile_plan.json` records exact commands and `profile_summary.json` collects
   outcomes. OOM, timeout, and insufficient storage are distinct from numerical
   divergence. A construction-time failure may have only console/case records.

4. **Only after the first case fits**, investigate larger microbatches at the
   same context, with unchanged effective target tokens/update:

   ```bash
   PROFILE_CASES="512:2 512:4" bash scripts/submit_profile.sh
   ```

   Then investigate longer contexts within a separately bounded 90-minute budget:

   ```bash
   PROFILE_CASES="1024:1 2048:1 4096:1" PROFILE_BUDGET_SECONDS=5400 \
     bash scripts/submit_profile.sh
   ```

   The prepared `.npy` shards are 1-D streams. `--repack-stream` explicitly
   reblocks them at the requested context after checking their original hashes,
   manifest, and selected-document provenance. It preserves document splits and
   tokenizer IDs and records runtime packing, new sequence boundaries, discarded
   tails, and actual target positions. This is a changed context protocol, not
   identical example ordering across contexts. Two-dimensional prepacked data
   cannot be reblocked this way. Do not rerun corpus download/preparation merely
   to change the profile context.

5. Select one feasible context/microbatch from measurements, then profile **both
   depths at that same setting**. For example, if 512/microbatch 2 was selected:

   ```bash
   PROFILE_CASES="512:2" PROFILE_DEPTHS="4 8" bash scripts/submit_profile.sh
   ```

   No broad size sweep or long training starts automatically. Each profile case
   initializes afresh in a separate process; no profile resume is allowed.
   Use `PROFILE_WARMUP_UPDATES=3 PROFILE_MEASURE_UPDATES=10` with an adequate
   declared time budget if the short timing samples vary substantially.

Measurement details:

- Steady throughput excludes complete performance warmup updates. Adam moments
  must exist before any steady measurement counts. Timing includes data access,
  transfers, full forward/backward, the real diagnostics, clipping, finite checks,
  optimizer update, and CUDA synchronization. JSON output and process-memory
  sampling are outside these timed sections.
- Each phase resets allocator peak counters without flushing the CUDA cache.
  Records include current/peak allocated and reserved bytes, device used/free
  observations via `mem_get_info`, and optional process-memory observations from
  `nvidia-smi`. Device/process observations are snapshots, not measured peaks;
  reserved cache can carry over between phases. Ordinary training also logs peak
  reserved memory alongside the existing allocated-memory metric.
- Validation time, actual token count, and memory are recorded separately.
  Checkpoint measurement includes `torch.save` and atomic rename, records actual
  file bytes, and makes no `fsync` durability guarantee. Defaults retain one
  populated checkpoint only long enough to measure it; no initial checkpoint is
  written in profile mode. The initial/final validation and checkpoint costs are
  excluded from steady training throughput and remain visible in total elapsed.
- The parent enforces a total wall-time budget and a per-case timeout, kills and
  waits for an over-budget child, and records the outcome. It stops after timeout
  or a code/configuration error. Explicitly requested cases may continue after
  OOM in a fresh process. SLURM time must exceed the declared budget plus startup
  and cleanup time. Completed records survive a later case's timeout/OOM.

**Decide the training budget from measured speed.** `profile.json` gives
training-only projections for 50,331,648 and 100,663,296 tokens. A profile job
containing matching T4/T8 cases also writes pair and two-seed-pair projections,
adding scaled 128K/every-100 validation, a 2M endpoint, and every-100 checkpoint
cost. These are extrapolations, excluding model/data setup, logging/sampling
overhead, storage contention, and replay. Reserve time for at least one replay
and proposal finishing. The user clarified that 72 hours is the per-request
limit; aggregate GPU-hour availability must be budgeted separately.
For illustration only, T8=1,000 and T4=2,000 tokens/s implies about 21 training
hours per 50.3M-token pair; halving those rates gives about 42 hours, before
validation/checkpoint overhead. Memory headroom does not determine throughput.

After this review, a selected full-size run can use the existing launcher with
`MODEL_CONFIG=ouro_1_4b_repro`, `SEQ_LEN`, `MICRO_BATCH`, and `GRAD_ACCUM` chosen
from the profile. Non-small models default to one rolling and one event-parent
checkpoint; initialization is also retained. For example, **only if the budget
review supports it**, the conservative 512/microbatch-1 50M pair is:

```bash
MODEL_CONFIG=ouro_1_4b_repro SEQ_LEN=512 MICRO_BATCH=1 GRAD_ACCUM=32 \
  MODE=reproduce LRS=3e-4 SEEDS=101 MAX_STEPS=3072 \
  bash scripts/submit_reproduction.sh
```

At context 4K with microbatch 1, use `SEQ_LEN=4096 GRAD_ACCUM=4 REPACK_STREAM=1`
instead. All larger-model runs have paths under
`$OUTPUT_ROOT/models/ouro_1_4b_repro/seq_CONTEXT/`, preserving old small-run paths.
The previous small-model LR choice does not establish an optimal full-size LR;
3e-4 is an initial recipe setting requiring review at this adapted batch size.
Allow roughly 100 GiB or more free scratch for a retained two-depth training
pair, including temporary checkpoint writes and possible event parents.

If full size is too slow or does not fit, explicitly select **one** fallback:
`PROFILE_MODEL_CONFIG=depth24_repro bash scripts/submit_profile.sh`. This retains
24 layers but uses width 768/FFN 2,048 (245,442,049 parameters). It matches the
original block depth—192 layer applications at T8—while remaining a reduced-width
adaptation. Do not sweep sizes automatically. A short successful profile does not
establish instability reproduction: original Stage 1a used 4K context, 4M–8M-token
batches, a different corpus mixture, and a 3T-token horizon. Model size alone does
not match those conditions, and instability is not guaranteed.

## Screen the original objective

First profile a short four/eight-loop run on the actual H100, including validation,
checkpoint time, and peak memory. No throughput figure is assumed by the code.
The default microbatch is two sequences, accumulated 16 times: 16,384 target
tokens per optimizer update. If changing the microbatch, preserve the product
with accumulation steps for the comparison.

The launcher runs serially on one allocated GPU. By default it screens four/eight
loops at learning rates 1e-4 and 3e-4, seed 0, for 512 updates with a 200-update
warmup. Neither depth is truncated or automatically reduced after a spike.

```bash
MODE=screen bash scripts/reproduce.sh
```

For submission from the login node, use the wrapper to read the optional account,
partition, interpreter, and data paths without repeating exports:

```bash
MODE=screen bash scripts/submit_reproduction.sh
```

The wrapper forwards additional `sbatch` options and exports the resolved
settings to the job. It sets the working directory to the repository and creates
`logs/` before calling `sbatch`, because SLURM opens its log files before running
the script. Standard output goes to `logs/ouro_reproduce_JOBID.out` and errors
to `logs/ouro_reproduce_JOBID.err`. It does not depend on SLURM's temporary script
location. Override resources when needed, for example
`bash scripts/submit_reproduction.sh --time=06:00:00`; GPU, memory, partition,
account, and notification overrides can also be passed as `sbatch` options.
Without overrides, it requests the cluster settings above. Resubmit the same
experiment command to continue from the latest rolling checkpoint if a job
does not finish its horizon in one allocation. The absolute `PYTHON_BIN` does
not depend on interactive shell activation. Put any site-required module-load commands in
`cluster.env` so they run during submission and inside the batch job. The
launcher installs nothing, requests one GPU, and writes local logs. Complete
pip installation and data preparation before going offline.

The earlier environment-only commands still work if no `cluster.env` is
present. The wrappers default to `python3` on `PATH` if `PYTHON_BIN` is unset.

For compute nodes without internet, stage the data, interpreter and installed
packages on shared storage first. If the preparation node also lacks internet,
download wheels on a compatible Linux host using the same Python version:

```bash
# On the compatible connected host:
python -m pip download --dest wheelhouse 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu126
python -m pip download --dest wheelhouse -r requirements-reproduction.txt
# After transferring wheelhouse and the prepared corpus to the cluster:
python -m pip install --no-index --find-links=wheelhouse 'torch==2.10.0' -r requirements-reproduction.txt
```

Use the same selected CUDA build in the download command. Create the venv on
the cluster; do not copy a Windows environment or an entire venv from another
machine. Prepare the corpus on a connected host and transfer the manifest,
record files, and token shards together. No model or tokenizer download occurs
during reproduction training from these prepared token files.

## Confirm reproduction before testing repair

Choose a reasonable learning rate from the original-objective screen. Repeat the
matched depth pair with independent seeds; target 3,072 updates, 50,331,648 target
tokens per run. Start with one seed if profiling indicates a tight allocation:

```bash
MODE=reproduce LRS=3e-4 SEEDS=101 bash scripts/submit_reproduction.sh
# After reviewing costs and the first pair:
MODE=reproduce LRS=3e-4 SEEDS="202 303" bash scripts/submit_reproduction.sh
```

The learning rate above is an example selection, not a pre-established best value.
All runs use the original joint objective. The same seed pairs initialization and
data order across depths. Every depth/seed/LR cell has its own output directory.
The launcher resumes from its latest rolling checkpoint and skips an already
completed requested horizon. `MAX_STEPS=6144` extends a run without changing its
earlier warmup-then-constant schedule. A shorter screen cannot establish the
absence of delayed instability.

If eight-loop events recur in at least two seed pairs with healthier four-loop
references, check unchanged checkpoint replay and conventional optimizer controls.
Only then add gate-detachment experiments. If both depths remain healthy, extend
a matched original-objective pair before expanding into a repair grid. Failure
to observe an event bounds the tested configuration and horizon.

## Outputs and event definitions

Each run writes:

- `run.json`: exact configs, seeds, data hashes, parameter count, code commit/diff
  hash, dependency versions, reference revision, and device information.
- `metrics.jsonl`: every completed optimizer update, full-accumulation loss and
  per-exit CE, exit probabilities, normalized entropy, raw/normalized state and
  residual RMS, pre-clipping gradient norm, clipping indicator, LR, processed
  tokens, training-only and end-to-end throughput, and CUDA peak memory.
- Validation records every 100 updates on up to 128K target tokens, and an endpoint
  validation on up to 2M tokens. All forced exits are read from the same unroll.
- `checkpoints/initial.pt` and five rolling checkpoints, with optimizer, RNG,
  stream position, event-monitor state, and validation protocol. Checkpoints are
  written atomically at accumulation boundaries.
- Up to three `events/` parent checkpoints retained separately, so early
  recoverable events remain replayable after rolling retention removes old files.
- `result.json`: process outcome, completed horizon, event counts, and the latest
  healthy checkpoint. `completed` means the process reached its horizon; it does
  not mean the model had no spikes or collapse.
- On numerical failure, `failure_context.pt` stores the triggering batch and
  identifies the preceding healthy checkpoint. No invalid gradient update is
  deliberately applied; if parameters become nonfinite during an optimizer step,
  the last healthy checkpoint remains available. Exit code 42 marks a numerical
  event, distinct from CLI errors. The launcher continues other experiment cells.

Each invocation has an `attempt_id`, including resumes. If a job restarts from an
older checkpoint, retained JSONL can contain repeated step numbers from different
attempts. Follow the chosen attempt's lineage rather than pooling repeated steps
as independent observations. Preserve the metadata and corpus manifest with results.

Operational flags follow the pilot protocol: after warmup and 100 previous
updates, a loss spike exceeds the trailing median uniform CE by both 0.5 nats
and five median absolute deviations. Gate collapse is first-exit mass above 0.95
for 100 updates. Persistent optimization collapse is fixed-probe uniform CE more
than 0.5 nats above its best earlier value for three consecutive evaluations.
Flags are reported separately. Finite spikes and gate collapse do not automatically
terminate training. These thresholds are descriptive definitions, not a trained
predictor or a proof of mechanism.

## Review completed runs in TensorBoard

The reproduction trainer writes JSONL metrics. Convert those existing logs to
TensorBoard event files with the optional exporter; training does not need to be
rerun. Install the viewer dependency in the existing environment and export:

```bash
source cluster.env
"$PYTHON_BIN" -m pip install tensorboard
"$PYTHON_BIN" scripts/export_tensorboard.py \
  --input-dir "$OUTPUT_ROOT/screen" \
  --output-dir "$OUTPUT_ROOT/tensorboard/screen"
"$PYTHON_BIN" -m tensorboard.main \
  --logdir "$OUTPUT_ROOT/tensorboard/screen" --host 127.0.0.1 --port 6006
```

Run the viewer on a host allowed by your cluster's service policy, for example
the login host if permitted. When it runs on the login host, use another terminal
on your own computer for the SSH tunnel, using your normal cluster login:

```bash
ssh -N -L 6006:127.0.0.1:6006 USER@CLUSTER_LOGIN
```

Open `http://localhost:6006` on your computer. Alternatively, copy the exported
event directory to your own computer and run TensorBoard there. Neither export
nor viewing uses a GPU or loads model checkpoints. The exporter uses PyTorch's
[SummaryWriter](https://docs.pytorch.org/docs/2.10/tensorboard.html).

Select the four screening runs and use the **Step** axis, which represents
optimizer updates. Start with smoothing at zero so recoverable spikes remain
visible. Useful charts are:

| TensorBoard tag | Review question |
| --- | --- |
| `train/uniform_ce` | Does prediction loss generally decrease? Are there abrupt excursions? |
| `validation/uniform_ce` | Does the fixed development probe improve or deteriorate persistently? |
| `endpoint_validation/per_step_losses/exit_4` | How does the common fourth exit compare across trained four/eight-loop models? |
| `endpoint_validation/per_step_losses/exit_8` | In an eight-loop model, is its final exit useful relative to its fourth exit? |
| `train/grad_norm`, `train/clipped` | Do loss excursions coincide with gradient excursions? Clipping alone is not failure. |
| `train/exit_probs/exit_1`, `train/normalized_entropy` | Is the gate concentrating on the first exit? |
| `train/raw_state_rms/loop_8` | Are raw recurrent states growing abruptly in the eight-loop run? |
| `train/events/loss_spike`, `validation/events/persistent_collapse` | Where did the predefined flags fire? |
| `train/end_to_end_tokens_per_second`, `train/peak_memory_gib` | What are the observed runtime and memory costs? |

Routine probes and endpoint validation have separate tag prefixes because they
use different token budgets. Compare like evaluations. Total `train/loss`
includes exit weighting and entropy, so use prediction CE and per-exit CE to
judge learning quality. Better deeper-exit prediction alone is not evidence of
training instability or a reasoning improvement.

Each resumed attempt gets its own TensorBoard run, named with `attempt_ID`, so
rolled-back or repeated steps are never joined into one curve. `export.json`
records the source hashes, attempt IDs, step ranges, and which attempt matches
each run's final `result.json`. Use that result and the resume metadata to follow
the chosen lineage. These attempt curves are not independent training seeds.
Nonfinite metric values are omitted from numeric curves and explicitly flagged
in the export report and `export/nonfinite_metric` chart.

The export leaves JSONL, checkpoints, and result metadata unchanged. It refuses
to append into a nonempty export directory to avoid duplicate points. Choose a
new output directory when exporting again, such as `screen_review_2`, and point
TensorBoard at that export. This is a snapshot exporter, not live training
logging. Event-file wall time reflects conversion time; use **Step** for the
curves and the recorded throughput/elapsed metrics for runtime. Keep the raw
JSONL as the source of record. TensorBoard is the visual review tool; predefined
event flags and independent-seed reproduction still determine the research
interpretation.

## Replay and final evaluation

Use the same arguments and add `--resume-from path/to/step_0000100.pt` for exact
stream continuation. The runner rejects changed architecture, training settings,
corpus contents, data ordering, validation cadence, and numerical options. Changing
only the total horizon is allowed. Use a fresh output directory for a separately
labeled checkpoint branch. `--resume-lr` explicitly changes the peak LR after
optimizer restoration; record it as an intervention. There is no automatic LR
change hidden inside a nominally unchanged replay.

The latest checkpoint stores the next batch cursor, not just a processed-step
count. Keyboard interruption or SIGTERM saves at a safe boundary where possible;
hard kills or interruption during accumulation recover from the last completed
rolling checkpoint. Recheck numerical replay on the H100 before attributing a
branch difference to an intervention.

For a frozen final comparison, pass `--test-data-dir` to a direct
`scripts/train.py --reproduction` command. Test data is evaluated only at the
endpoint and is never used for event thresholds. Do not pass it during screening
or tune against those test results. The launcher intentionally uses validation only.

## Software verification scope

The patch includes offline CPU tests of the recurrence contract, actual parameter
counts, full recurrent gradients, low-precision reductions, accumulation,
warmup, replay of data/RNG/updates, nonfinite-gradient handling, worker sharding,
event classification, and CLI train/resume. The separate pinned-reference check
tests numerical forward and gradient agreement. H100 memory, BF16 kernel behavior,
throughput, and occurrence of instability still require the on-device pilot.
