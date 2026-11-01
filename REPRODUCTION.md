# Small scale instability reproduction

This patch supplies a single-GPU harness for testing whether increasing training
depth from four to eight loops produces repeatable instability. It does not
establish that instability occurs or that a gate intervention repairs it.

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
`gpu:h100-96:1`, 12 requested hours, 8 CPUs, and 96G of host RAM. The batch job
sends END/FAIL notifications to `e1121685@u.nus.edu`. Twelve hours is the requested
job duration, not an assertion about the partition's maximum time limit.
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
.uv/reference-venv/bin/python scripts/verify_ouro_reference.py
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
