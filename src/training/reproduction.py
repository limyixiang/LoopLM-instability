"""Instrumented single-GPU reproduction of depth-related training instability.

Only the original joint objective is used. Completion is an observation horizon,
not evidence of permanent stability or a successful stabilization method.
"""

import dataclasses
import hashlib
import json
import math
import os
import platform
import signal
import shutil
import statistics
import subprocess
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src.training.objectives import compute_looplm_loss
from src.training.replay import TokenCorpus, ReplayableBatchStream, capture_rng_state, restore_rng_state
from src.training.trainer import Trainer, TrainerConfig, TrainingDivergence, _merge_diagnostics


OURO_REFERENCE = "ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1"


def json_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class InstabilityMonitor:
    """Descriptive thresholds declared in the pilot protocol, not a predictor."""

    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps
        self.history = deque(maxlen=100)
        self.gate_count = 0
        self.best_validation = None
        self.bad_validation_count = 0
        self.counts = {"loss_spike": 0, "gate_collapse": 0, "persistent_collapse": 0}

    def train_events(self, row: dict) -> list[str]:
        events = []
        ce = row["uniform_ce"]
        if row["step"] > self.warmup_steps and len(self.history) == 100:
            median = statistics.median(self.history)
            mad = statistics.median(abs(value - median) for value in self.history)
            if ce > median + max(0.5, 5 * mad):
                events.append("loss_spike")
        self.history.append(ce)
        self.gate_count = self.gate_count + 1 if row["exit_probs"][0] > 0.95 else 0
        if self.gate_count == 100:
            events.append("gate_collapse")
        for event in events:
            self.counts[event] += 1
        return events

    def validation_events(self, ce: float) -> list[str]:
        if self.best_validation is None:
            self.best_validation = ce
        self.bad_validation_count = self.bad_validation_count + 1 if ce > self.best_validation + 0.5 else 0
        self.best_validation = min(self.best_validation, ce)
        if self.bad_validation_count == 3:
            self.counts["persistent_collapse"] += 1
            return ["persistent_collapse"]
        return []

    def state_dict(self) -> dict:
        return {**self.__dict__, "history": list(self.history)}

    def load_state_dict(self, state: dict) -> None:
        self.__dict__.update(state)
        self.history = deque(state["history"], maxlen=100)


@torch.no_grad()
def evaluate_corpus(trainer: Trainer, corpus: TokenCorpus, batch_size: int, max_tokens: int) -> dict:
    """Read every forced exit in one recurrent forward pass per batch."""
    rng = capture_rng_state()
    was_training = trainer.model.training
    totals, n_tokens = {}, 0.0
    stream = ReplayableBatchStream(corpus, batch_size, seed=0, shuffle=False)
    batches = min(len(corpus) // batch_size, max(1, math.ceil(max_tokens / (batch_size * corpus.seq_len))))
    try:
        trainer.model.eval()
        for _ in range(batches):
            batch = next(stream).to(trainer.device)
            with torch.autocast(trainer.device.type, dtype=torch.bfloat16,
                                enabled=trainer.config.use_amp and trainer.device.type == "cuda"):
                out = trainer.raw_model(batch[:, :-1], num_steps=trainer.num_recurrent_steps)
                _, diagnostics = compute_looplm_loss(out.logits, out.exit_lambdas, batch[:, 1:],
                                                      beta=trainer.config.beta_kl)
            weight = float(diagnostics["valid_tokens"])
            _merge_diagnostics(totals, diagnostics, weight)
            n_tokens += weight
        result = {key: value / n_tokens for key, value in totals.items()}
        if not all(torch.isfinite(value).all().item() for value in result.values()):
            raise TrainingDivergence("Nonfinite held-out prediction or gate statistics")
        return {**json_value(result), "valid_tokens": int(n_tokens)}
    finally:
        trainer.model.train(was_training)
        restore_rng_state(rng)


def _git_info() -> dict:
    root = Path(__file__).resolve().parents[2]
    def git(*arguments):
        result = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", *arguments],
                                cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    # Include newly created, untracked source files in experiment provenance.
    digest = hashlib.sha256()
    paths = [root / "pyproject.toml", root / "uv.lock"]
    for directory in (root / "src", root / "scripts"):
        paths.extend(path for path in directory.rglob("*") if path.suffix in (".py", ".sh"))
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    source_info = {"source_tree_sha256": digest.hexdigest()}
    try:
        return {**source_info, "commit": git("rev-parse", "HEAD"), "status": git("status", "--short"),
                "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()}
    except FileNotFoundError:
        return {**source_info, "commit": "unavailable"}


def run_reproduction(args, model_config) -> int:
    if "RANK" in os.environ or any((args.stage1a_steps, args.stage1b_steps, args.stage2_steps, args.use_q_act)):
        raise ValueError("Reproduction uses one GPU, one fixed depth, and the joint objective only")
    if not args.data_dir or not args.val_data_dir:
        raise ValueError("--reproduction requires separate --data-dir and --val-data-dir .npy corpora")
    if min(args.max_steps, args.batch_size, args.seq_len, args.val_every, args.val_tokens,
           args.save_every, args.keep_checkpoints, args.keep_event_checkpoints,
           args.endpoint_val_tokens, args.test_tokens) < 1:
        raise ValueError("Step, batch, evaluation, and retention settings must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume_from and any((output / name).exists() for name in ("run.json", "metrics.jsonl")):
        raise ValueError("Output already contains a run; choose a new directory or resume explicitly")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    config = TrainerConfig(lr=args.lr, max_steps=args.max_steps, beta_kl=args.beta_kl,
                           num_recurrent_steps=args.num_recurrent_steps, seed=args.seed,
                           warmup_steps=args.warmup_steps, grad_accum_steps=args.grad_accum_steps,
                           device=args.device, use_amp=not args.no_amp,
                           checkpoint_dir=str(output / "checkpoints"),
                           log_every=args.log_every, save_every=args.save_every)
    corpus = TokenCorpus(args.data_dir, args.seq_len)
    validation = TokenCorpus(args.val_data_dir, args.seq_len)
    test = TokenCorpus(args.test_data_dir, args.seq_len) if args.test_data_dir else None
    if set(item["sha256"] for item in corpus.manifest) & set(item["sha256"] for item in validation.manifest):
        raise ValueError("Training and validation contain identical token shards")
    if test is not None and set(item["sha256"] for item in test.manifest) & set(
            item["sha256"] for item in corpus.manifest + validation.manifest):
        raise ValueError("Test data contains a training or validation shard")
    preparations = [item.preparation for item in (corpus, validation, test) if item is not None and item.preparation]
    for prepared in preparations:
        tokenizer = prepared["tokenizer"]
        if tokenizer["repository"] != args.tokenizer_id or tokenizer["revision"] != args.tokenizer_revision:
            raise ValueError("Prepared tokenizer differs from the declared tokenizer identity/revision")
        if tokenizer["vocab_size"] != model_config.vocab_size:
            raise ValueError("Prepared tokenizer and model vocabulary differ")
        if prepared["source"] != preparations[0]["source"] or tokenizer["fingerprint"] != preparations[0]["tokenizer"]["fingerprint"]:
            raise ValueError("Training and held-out corpora have different sources or tokenizer behavior")
    preparation_fingerprints = {"train": corpus.preparation["manifest_sha256"] if corpus.preparation else None,
                                "validation": validation.preparation["manifest_sha256"] if validation.preparation else None}
    stream = ReplayableBatchStream(corpus, args.batch_size, args.data_seed)
    monitor = InstabilityMonitor(args.warmup_steps)
    trainer = Trainer(model_config, config)
    trainer.raw_model.gradient_checkpointing = args.gradient_checkpointing
    trainer.raw_model.collect_diagnostics = True
    attempt_id = uuid.uuid4().hex[:12]
    protocol = {"val_every": args.val_every, "val_tokens": args.val_tokens,
                "deterministic": args.deterministic,
                "gradient_checkpointing": args.gradient_checkpointing}
    metadata = {"attempt_id": attempt_id, "objective": "joint", "reference": OURO_REFERENCE,
                "model": dataclasses.asdict(model_config), "trainer": dataclasses.asdict(config),
                "arguments": vars(args), "parameters": sum(p.numel() for p in trainer.raw_model.parameters()),
                "train_fingerprint": corpus.fingerprint, "validation_fingerprint": validation.fingerprint,
                "train_shards": corpus.manifest, "validation_shards": validation.manifest,
                "test_shards": test.manifest if test is not None else None,
                "data_preparation": {"train": corpus.preparation, "validation": validation.preparation,
                                     "test": test.preparation if test is not None else None},
                "git": _git_info(), "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "cuda": torch.version.cuda,
                "device": str(trainer.device),
                "gpu": torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else None}
    if args.resume_from:
        saved = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if saved.get("model_config") != dataclasses.asdict(model_config):
            raise ValueError("Resume architecture differs from the checkpoint")
        previous = saved["trainer_config"]
        for key in ("lr", "warmup_steps", "grad_accum_steps", "beta_kl", "num_recurrent_steps", "seed", "use_amp"):
            if previous.get(key) != getattr(config, key):
                raise ValueError(f"Resume configuration mismatch: {key}; use --resume-lr for an explicit LR fork")
        trainer.load_checkpoint(args.resume_from, lr_override=args.resume_lr)
        if trainer.restored_data_state is None:
            raise ValueError("Checkpoint has no replay state; it cannot resume a controlled reproduction")
        stream.load_state_dict(trainer.restored_data_state["stream"])
        monitor.load_state_dict(trainer.restored_data_state["monitor"])
        if trainer.restored_data_state["validation_fingerprint"] != validation.fingerprint:
            raise ValueError("Resume validation corpus differs")
        if trainer.restored_data_state.get("protocol") != protocol:
            raise ValueError("Resume changes validation cadence or numerical settings")
        if trainer.restored_data_state.get("preparation_fingerprints", {"train": None, "validation": None}) != preparation_fingerprints:
            raise ValueError("Resume changes dataset or tokenizer preparation provenance")
        write_json(output / f"resume_{trainer.step:07d}_{attempt_id}.json", metadata)
    else:
        write_json(output / "run.json", metadata)
    if trainer.step > args.max_steps:
        raise ValueError("Training horizon precedes the resumed checkpoint")

    def data_state():
        return {"stream": stream.state_dict(), "monitor": monitor.state_dict(),
                "validation_fingerprint": validation.fingerprint, "protocol": protocol,
                "preparation_fingerprints": preparation_fingerprints}

    def checkpoint(initial=False):
        path = trainer.save_checkpoint(data_state(), filename="initial.pt" if initial else None)
        paths = sorted(path.parent.glob("step_*.pt"))
        for old in paths[:-args.keep_checkpoints]:
            old.unlink()
        return path

    initial = checkpoint(initial=True) if not args.resume_from else Path(args.resume_from)
    last_healthy = initial
    started, trained_seconds = time.perf_counter(), 0.0
    starting_tokens = trainer.tokens_seen
    status, failure, failing_batch = "completed", None, None
    # Reset validation uses a fresh deterministic iterator and does not consume model RNG.
    metrics = (output / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)

    def log(row):
        row["attempt_id"] = attempt_id
        # Retain parents of early recoverable events even after rolling pruning.
        if row.get("events"):
            directory = output / "events"
            directory.mkdir(exist_ok=True)
            if len(list(directory.glob("*.pt"))) < args.keep_event_checkpoints:
                destination = directory / f"event_{row['step']:07d}_{attempt_id}_parent_{last_healthy.name}"
                try:
                    os.link(last_healthy, destination)
                except OSError:
                    shutil.copyfile(last_healthy, destination)
                row["pre_event_checkpoint"] = str(destination)
        metrics.write(json.dumps(json_value(row), allow_nan=False) + "\n")

    previous_termination = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        probe = evaluate_corpus(trainer, validation, args.batch_size, args.val_tokens)
        # On resume, do not count an extra evaluation toward the collapse threshold.
        events = [] if args.resume_from else monitor.validation_events(probe["uniform_ce"])
        log({"kind": "validation", "step": trainer.step, **probe, "events": events})
        while trainer.step < args.max_steps:
            before_stream = stream.state_dict()
            batch = next(stream)
            if batch.min().item() < 0 or batch.max().item() >= model_config.vocab_size:
                raise ValueError("Token IDs outside the configured vocabulary")
            failing_batch = batch
            tick = time.perf_counter()
            diagnostics = trainer.train_step(batch)
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            trained_seconds += time.perf_counter() - tick
            if not diagnostics.get("_stepped", True):
                continue
            row = {"kind": "train", "step": trainer.step, **json_value(diagnostics),
                   "elapsed_seconds": time.perf_counter() - started,
                   "training_seconds": trained_seconds, "batch_indices": stream.last_indices,
                   "data_epoch": stream.epoch, "data_offset": stream.offset}
            row["events"] = monitor.train_events(row)
            row["tokens_per_second"] = (trainer.tokens_seen - starting_tokens) / max(trained_seconds, 1e-9)
            row["end_to_end_tokens_per_second"] = (trainer.tokens_seen - starting_tokens) / max(row["elapsed_seconds"], 1e-9)
            if trainer.device.type == "cuda":
                row["peak_memory_bytes"] = torch.cuda.max_memory_allocated(trainer.device)
            log(row)
            if trainer.step % args.log_every == 0:
                trainer._log(diagnostics)
            if trainer.step % args.val_every == 0:
                probe = evaluate_corpus(trainer, validation, args.batch_size, args.val_tokens)
                log({"kind": "validation", "step": trainer.step, **probe,
                     "events": monitor.validation_events(probe["uniform_ce"])})
            if trainer.step % args.save_every == 0:
                last_healthy = checkpoint()
        probe = evaluate_corpus(trainer, validation, args.batch_size, args.endpoint_val_tokens)
        log({"kind": "endpoint_validation", "step": trainer.step, **probe})
        if test is not None:
            held_out = evaluate_corpus(trainer, test, args.batch_size, args.test_tokens)
            log({"kind": "endpoint_test", "step": trainer.step, **held_out})
        last_healthy = checkpoint()
    except TrainingDivergence as error:
        status, failure = "numerical_divergence", str(error)
        torch.save({"batch": failing_batch, "stream_before_batch": locals().get("before_stream"),
                    "last_healthy_checkpoint": str(last_healthy), "rng_state": capture_rng_state()},
                   output / "failure_context.pt")
        log({"kind": "numerical_divergence", "step": trainer.step, "message": failure,
             "tokens_seen": trainer.tokens_seen, "last_healthy_checkpoint": str(last_healthy)})
    except KeyboardInterrupt:
        status = "interrupted"
        # Never label a checkpoint with unsaved partial accumulation as replayable.
        if not trainer._accum_count:
            last_healthy = checkpoint()
    except Exception as error:
        status, failure = "error", str(error)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_termination)
        metrics.close()
        write_json(output / "result.json", {"attempt_id": attempt_id, "status": status, "failure": failure,
                   "completed_updates": trainer.step, "tokens_seen": trainer.tokens_seen,
                   "events": monitor.counts, "last_healthy_checkpoint": str(last_healthy),
                   "elapsed_seconds": time.perf_counter() - started,
                   "scope": "Observed horizon only; no causal or permanent-stability conclusion"})
    return 42 if failure else (130 if status == "interrupted" else 0)
