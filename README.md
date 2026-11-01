# LoopLM

Reimplementation of [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) (Ouro, arXiv:2510.25741v4).

A decoder-only transformer where a shared-weight layer stack is applied recurrently, enabling adaptive computation depth at inference time.

## Architecture

- **Shared-weight recurrence** — a stack of N layers is applied T times, reusing the same weights each pass
- **Sandwich normalization** — RMSNorm before both attention and FFN (critical for recurrent stability)
- **Exit gate** — at each step, a learned gate predicts whether to stop early
- **Multi-step loss** — all T steps contribute to training via a weighted sum, where weights come from the learned exit distribution
- **SwiGLU FFN**, **RoPE** positional encoding, 49,152 token vocabulary (pinned Ouro tokenizer for reproduction; SmolLM2 for legacy runs)

## Setup

The reproduction pilot supports standard Python `venv` and pip; `uv` is optional.
See [REPRODUCTION.md](REPRODUCTION.md) for the CUDA/PyTorch installation choice,
the separate reference checker, and offline cluster setup.

```bash
git clone <repo>
cd looplm
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
source .venv/bin/activate
# Install a CUDA-enabled PyTorch build appropriate for your cluster first.
python -m pip install -r requirements-reproduction.txt
if [ ! -f cluster.env ]; then cp cluster.env.example cluster.env; fi
# Check storage/Python paths; gpu-long is preset and an account is optional.
```

## Instability reproduction pilot

The reproduction harness is implemented; no H100 instability results have been
established by this patch. See [REPRODUCTION.md](REPRODUCTION.md) for preparation,
screening, matched four/eight-loop runs, replay, and output interpretation.

Use `scripts/train.py --reproduction` for the controlled pilot. Its `small_repro`
preset has approximately 118M parameters, untied embeddings, and the released
Ouro loop-boundary normalization. The existing `small` preset retains the legacy
approximately 80M architecture. Shared RMSNorm and probability reductions now use
FP32 where needed; bitwise agreement with older BF16 runs is not promised.

The pilot retains the original joint objective and full recurrent backpropagation.
Detached-gate and uniform-supervision interventions remain future work, conditional
on reproducing a meaningful instability.

The primary corpus is a fixed, pinned FineWeb-Edu subset with content-hash-based
document splits. Preparation defaults to the pinned released Ouro tokenizer;
SmolLM2 has different special-token mappings despite the same vocabulary size.
Raw WikiText-103 (`wikitext-103-raw-v1`) is an optional pipeline smoke test, with
row-level separation rather than reconstructed article-level separation.

## Training

### Local

```bash
python scripts/train.py \
    --model-config small \
    --dataset wikitext \
    --dataset-config wikitext-103-raw-v1 \
    --max-steps 10000 \
    --batch-size 8 \
    --seq-len 512 \
    --lr 3e-4 \
    --output-dir ./checkpoints
```

### Mano task

`scripts/mano.slurm` — Rewritten as a SLURM job array:
  - --array=0-8 runs 9 tasks (3 configs × 3 seeds), each with 1 GPU
  - Maps SLURM_ARRAY_TASK_ID → (config, seed) using bash arrays
  - Each task writes to $OUT_DIR/task_<ID>/mano_results.csv
  - Logs use %A_%a pattern (array job ID + task ID)

  `scripts/analyze.py` — Added mano-collect subcommand:
  - Reads all */mano_results.csv files from the input directory
  - Groups by (num_layers, loop_count, total_depth), computes mean ± std
  - Prints a formatted table and writes mano_results_combined.csv

  Usage:
  # Submit all 9 jobs in parallel
  ```bash
  sbatch scripts/mano.slurm
  ```

  # After all complete, aggregate results
  ```bash
  python scripts/analyze.py mano-collect --input-dir ${SCRATCH}/looplm/mano
  ```

### SLURM / H100

**One-time setup on login node** (needs internet):

```bash
source .venv/bin/activate
bash scripts/download_artifacts.sh   # prepares the bounded pinned FineWeb-Edu pilot
```

**Submit job:**

```bash
# Reads saved cluster.env settings, including after reconnecting:
MODE=screen bash scripts/submit_reproduction.sh
```

Defaults match the supplied cluster header: partition `gpu-long`, one
`gpu:h100-96:1`, 12 requested hours, 8 CPUs, and 96G host RAM, with END/FAIL
email notifications. `PILOT_ACCOUNT` may be empty; an explicit account is passed
only when configured. Additional `sbatch` options remain available, for example
`bash scripts/submit_reproduction.sh --time=06:00:00`.

**Monitor:**

```bash
squeue --me
tail -f logs/ouro_reproduce_12345.out  # replace 12345 with your job ID
tail -f logs/ouro_reproduce_12345.err
```

The reproduction harness writes local JSONL metrics and checkpoints; see
[REPRODUCTION.md](REPRODUCTION.md) for interpretation. Stage prepared data and
installed dependencies before using an offline compute node. The legacy
`scripts/train.slurm` supports offline wandb logging for legacy runs.
It also accepts `PYTHON_BIN`; install `wandb` in that environment if using it.
The reproduction wrappers load `cluster.env` automatically; see
[cluster.env.example](cluster.env.example). Existing environment-only usage is
still supported. Keep site-specific Python/CUDA module commands in the saved
configuration if needed on compute nodes.
The submission wrapper creates `logs/` before calling `sbatch`, so SLURM can
open its output/error files before the job script starts. Reuse your existing
`.venv` through `PYTHON_BIN`; no `uv` installation is needed. For an older
`cluster.env` template, clear any `YOUR_ACCOUNT` value and set the partition to
`gpu-long` before submitting.

## Running Tests

```bash
python -m pytest tests/ -v
```

Stop at first failure:

```bash
python -m pytest tests/ -x -v
```

## Project Structure

```
src/
  model/
    config.py        # LoopLMConfig dataclass (small ~100M, ouro_1_4b, ouro_2_6b)
    rope.py          # Rotary position embeddings with configurable base frequency
    transformer.py   # TransformerBlock with sandwich norm, SwiGLU, RoPE
    looplm.py        # LoopLM: recurrent forward pass, exit gate, per-step logits
  training/
    objectives.py    # Stage I entropy-regularized loss; Stage II adaptive gate loss
    trainer.py       # Training loop: AdamW, grad clip, checkpointing, wandb logging
    data.py          # HuggingFace dataset loading and tokenization
scripts/
  train.py                  # Training entry point (argparse CLI)
  train.slurm               # SLURM job script for JUWELS westai partition
  download_artifacts.sh     # Pre-stage HF artifacts on login node
configs/
  small.yaml                # ~100M params (6 layers, hidden 768, 12 heads)
  medium.yaml               # ~400M params
  ouro_1_4b.yaml            # 1.4B, 24 layers, hidden 2048
  ouro_2_6b.yaml            # 2.6B, 48 layers, hidden 2048
tests/                      # 74 tests covering all modules above
```

## Model Configs

| Config      | Params | Layers | Hidden | Heads | Recurrent steps |
|-------------|--------|--------|--------|-------|-----------------|
| `small()`   | ~100M  | 6      | 768    | 12    | 4               |
| `ouro_1_4b` | 1.4B   | 24     | 2048   | 16    | 4               |
| `ouro_2_6b` | 2.6B   | 48     | 2048   | 16    | 4               |

## Implementation Status

- [x] Config system (`LoopLMConfig`)
- [x] RoPE with configurable base frequency
- [x] Transformer block (sandwich norm, SwiGLU, RoPE attention)
- [x] LoopLM model (recurrence, exit gate, per-step logits)
- [x] Stage I training objective (entropy-regularized, uniform prior)
- [x] Stage II adaptive gate loss (detached improvement signal)
- [x] Training loop (AdamW, grad clip, checkpointing, wandb)
- [x] Data pipeline (HuggingFace datasets, tokenization)
- [x] Training entry point (`scripts/train.py`)
- [x] SLURM job script for JUWELS westai (H100)
- [ ] Early exit inference strategies (Q-exit, hidden state diff)
- [ ] KV cache sharing for efficient decode
- [ ] Evaluation harness integration (lm-eval)
- [ ] Multi-stage training (stage transitions, RoPE base annealing)
- [ ] Scale to 1.4B / upcycling to 2.6B

## Future Ideas

### Detached recurrence (truncated BPTT)
Only backprop through the last recurrent step, detaching hidden states between iterations. [HRM](https://arxiv.org/) does this. Eliminates gradient explosion entirely but loses cross-step gradient coordination. Each step still gets a direct loss signal, and since weights are shared, improving one step tends to improve all.

### Diffusion-style step sampling
Instead of computing all T steps and weighting by the exit distribution, **sample a single recurrent step per batch** and importance-weight the loss. Direct parallel to diffusion training where you sample a random timestep t rather than denoising the full chain.

Relevant diffusion insights that transfer:
- [Min-SNR](https://arxiv.org/abs/2303.09556) — weight timesteps by clamped signal-to-noise ratio to resolve conflicting gradients between steps (3.4x faster convergence)
- CLTS — curriculum from uniform to focused Gaussian sampling (easy steps first, hard steps later)
- [A Closer Look at Time Steps](https://arxiv.org/html/2405.17403v1) — non-uniform timestep sampling for 3x speedup

Looped transformer work already exploring this direction:
- [LoopFormer](https://loopformer.github.io/) — conditions on internal time t and step size, allowing step-skipping (analogous to DDIM)
- [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/html/2311.12424v2) — randomly samples iteration count per training batch
- [Retrofitted Recurrence](https://arxiv.org/html/2511.07384v1) — schedules max backprop depth over training

A curriculum schedule (early recurrence steps first, shift to later steps as training progresses) could both stabilize training and improve sample efficiency.

## Reference

- Paper: [arXiv:2510.25741v4](https://arxiv.org/abs/2510.25741)
- Project page: https://ouro-llm.github.io
