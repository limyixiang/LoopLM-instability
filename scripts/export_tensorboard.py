#!/usr/bin/env python
"""Export recorded reproduction metrics to TensorBoard without running training."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


SCALARS = (
    "loss", "task_loss", "uniform_ce", "entropy", "normalized_entropy",
    "avg_exit_step", "grad_norm", "clipped", "lr", "tokens_seen", "valid_tokens",
    "elapsed_seconds", "training_seconds", "tokens_per_second",
    "end_to_end_tokens_per_second",
)
VECTORS = {
    "per_step_losses": "exit", "exit_probs": "exit",
    "raw_state_rms": "loop", "state_rms": "loop", "residual_rms": "loop",
}
EVENTS = {
    "train": ("loss_spike", "gate_collapse"),
    "validation": ("persistent_collapse",),
    "numerical_divergence": ("numerical_divergence",),
}
KINDS = {"train", "validation", "endpoint_validation", "endpoint_test", "numerical_divergence"}


def read_attempts(path, raw):
    attempts = {}
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{number}; finish writing or repair the log first") from error
        attempt = str(row.get("attempt_id", "legacy"))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", attempt):
            raise ValueError(f"Invalid attempt_id at {path}:{number}")
        if row.get("kind") not in KINDS:
            raise ValueError(f"Unknown metric kind at {path}:{number}")
        if type(row.get("step")) is not int or row["step"] < 0:
            raise ValueError(f"Invalid optimizer step at {path}:{number}")
        attempts.setdefault(attempt, []).append(row)
    return attempts


def export_tensorboard(input_dir, output_dir):
    source, destination = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    if destination == source or source.is_relative_to(destination):
        raise ValueError("Output must be separate from, or a new subdirectory of, the input")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty output directory; appending a second export would duplicate curves")
    files = sorted(source.rglob("metrics.jsonl"))
    if not files:
        raise ValueError(f"No metrics.jsonl files found under {source}")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("Install the optional viewer dependency: python -m pip install tensorboard") from error

    destination.mkdir(parents=True, exist_ok=True)
    report = {"source": str(source), "scope": "Post-hoc visualization only; original metrics and checkpoints unchanged",
              "x_axis": "optimizer updates; event-file wall time is export time, not original training time",
              "attempt_policy": "Separate TensorBoard runs per attempt; rolled-back steps are never joined",
              "runs": [], "nonfinite_values_skipped": 0}
    for path in files:
        relative = path.parent.relative_to(source)
        raw = path.read_bytes()
        attempts = read_attempts(path, raw)
        if not attempts:
            raise ValueError(f"No metric records in {path}; wait for the run to write its logs")
        result_path = path.parent / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        for attempt, rows in attempts.items():
            run_dir = destination / relative / f"attempt_{attempt}"
            scalars, invalid = 0, 0
            with SummaryWriter(log_dir=str(run_dir)) as writer:
                writer.add_text("provenance/attempt_id", attempt, global_step=0)
                writer.add_text("provenance/source", str(path), global_step=0)

                def scalar(tag, value, step):
                    nonlocal scalars, invalid
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        raise ValueError(f"Expected a numeric value for {tag} at step {step} in {path}")
                    if not math.isfinite(value):
                        invalid += 1
                        writer.add_scalar("export/nonfinite_metric", 1, step)
                        return
                    writer.add_scalar(tag, value, step)
                    scalars += 1

                for row in rows:
                    kind, step = row["kind"], row["step"]
                    for key in SCALARS:
                        if key in row:
                            scalar(f"{kind}/{key}", row[key], step)
                    for key, unit in VECTORS.items():
                        for index, value in enumerate(row.get(key, []), 1):
                            scalar(f"{kind}/{key}/{unit}_{index}", value, step)
                    if "peak_memory_bytes" in row:
                        scalar(f"{kind}/peak_memory_gib", float(row["peak_memory_bytes"]) / 2**30, step)
                    for event in EVENTS.get(kind, ()):
                        flagged = event == kind or event in row.get("events", [])
                        scalar(f"{kind}/events/{event}", int(flagged), step)

            report["nonfinite_values_skipped"] += invalid
            report["runs"].append({"source": path.relative_to(source).as_posix(),
                                   "source_sha256": hashlib.sha256(raw).hexdigest(),
                                   "tensorboard_run": run_dir.relative_to(destination).as_posix(),
                                   "attempt_id": attempt, "records": len(rows), "scalar_points": scalars,
                                   "first_step": min(row["step"] for row in rows),
                                   "last_step": max(row["step"] for row in rows),
                                   "matches_final_result_attempt": result.get("attempt_id") == attempt,
                                   "result_status": result.get("status") if result.get("attempt_id") == attempt else None,
                                   "nonfinite_values_skipped": invalid})
    (destination / "export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Screen/reproduction root, or a single run directory")
    parser.add_argument("--output-dir", required=True, help="New or empty directory for derived event files")
    args = parser.parse_args()
    try:
        report = export_tensorboard(args.input_dir, args.output_dir)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Exported {len(report['runs'])} attempt curves to {args.output_dir}")
    print("Use the Step axis. TensorBoard wall time reflects export time; inspect elapsed_seconds/throughput for runtime.")
    if report["nonfinite_values_skipped"]:
        print(f"Warning: {report['nonfinite_values_skipped']} nonfinite values omitted and flagged; inspect the source logs.")


if __name__ == "__main__":
    main()
