"""Synchronized, bounded measurements of the real reproduction training path."""

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device):
    """Allocator peaks plus end-of-phase device/process observations (not peaks)."""
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    result = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_free_bytes_observed": free, "device_total_bytes": total,
        "device_used_bytes_observed": total - free,
        "process_used_bytes_observed": None,
    }
    try:
        response = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,gpu_uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False)
        processes = []
        for line in response.stdout.splitlines():
            pid, mib, gpu_uuid = (item.strip() for item in line.split(","))
            if int(pid) == os.getpid():
                processes.append({"gpu_uuid": gpu_uuid, "bytes": int(mib) * 1024**2})
        result["process_devices_observed"] = processes
        if processes:
            result["process_used_bytes_observed"] = sum(item["bytes"] for item in processes)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass  # Unavailable under some containers/MIG/driver configurations.
    return result


class ReproductionProfile:
    """Warmup is complete optimizer updates, including Adam allocation.

    Per-update timing includes data reads/transfers, the joint objective, full
    recurrent forward/backward, diagnostics, clipping, optimizer, finite checks,
    and synchronization. JSON output and nvidia-smi sampling occur outside it.
    Validation and serialization are measured separately, with no CUDA cache
    flush between phases. All quantities describe this instrumented eager path.
    """

    def __init__(self, output: Path, device, warmup_updates: int):
        self.output, self.device, self.warmup_updates = output, device, warmup_updates
        self.updates, self.operations = [], []
        self.adam_state_bytes = None
        self.started = time.perf_counter()
        self._tick = None

    def begin(self):
        synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._tick = time.perf_counter()

    def _end(self):
        synchronize(self.device)
        seconds = time.perf_counter() - self._tick
        return seconds, memory_snapshot(self.device)

    def _write(self, row):
        with (self.output / "profile_measurements.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def end_update(self, trainer, diagnostics):
        seconds, memory = self._end()
        phase = "warmup" if trainer.step <= self.warmup_updates else "steady"
        row = {"kind": "optimizer_update", "phase": phase, "step": trainer.step,
               "seconds": seconds, "target_tokens": int(diagnostics["valid_tokens"]), **memory}
        self.updates.append(row)
        self._write(row)
        if trainer.step == self.warmup_updates:
            # Adam's moments must exist before any steady measurements count.
            moments = [value for state in trainer.optimizer.state.values()
                       for key, value in state.items() if key in ("exp_avg", "exp_avg_sq")]
            self.adam_state_bytes = sum(value.numel() * value.element_size() for value in moments)
            expected = sum(2 * parameter.numel() * parameter.element_size()
                           for parameter in trainer.raw_model.parameters() if parameter.requires_grad)
            if self.adam_state_bytes != expected:
                raise RuntimeError("Profile warmup did not allocate both Adam moments for every trainable parameter")

    def measure(self, name, function):
        self.begin()
        value = function()
        seconds, memory = self._end()
        row = {"kind": name, "seconds": seconds, **memory}
        if isinstance(value, Path):
            row.update(path=str(value), file_bytes=value.stat().st_size)
        elif isinstance(value, dict) and "valid_tokens" in value:
            row["target_tokens"] = value["valid_tokens"]
        self.operations.append(row)
        self._write(row)
        return value

    def finish(self, status):
        steady = [row for row in self.updates if row["phase"] == "steady"]
        tokens = sum(row["target_tokens"] for row in steady)
        seconds = sum(row["seconds"] for row in steady)
        rate = tokens / seconds if seconds else None
        failure_memory = memory_snapshot(self.device) if status != "completed" else {}
        report = {
            "status": status, "purpose": "performance_profile_not_instability_replication",
            "warmup_updates_requested": self.warmup_updates,
            "warmup_updates_completed": len(self.updates) - len(steady),
            "steady_updates": len(steady), "steady_target_tokens": tokens,
            "steady_seconds": seconds, "steady_tokens_per_second": rate,
            "steady_update_seconds_median": statistics.median(row["seconds"] for row in steady) if steady else None,
            "adam_moment_bytes_after_warmup": self.adam_state_bytes,
            "profile_elapsed_seconds": time.perf_counter() - self.started,
            "operations": self.operations,
            "failure_memory_observation": failure_memory,
            "max_measured_peak_allocated_bytes": max((row.get("peak_allocated_bytes", 0)
                                                       for row in self.updates + self.operations + [failure_memory]), default=0),
            "max_measured_peak_reserved_bytes": max((row.get("peak_reserved_bytes", 0)
                                                      for row in self.updates + self.operations + [failure_memory]), default=0),
            "training_only_hours_by_target_tokens": {
                str(target): target / rate / 3600 if rate else None for target in (50331648, 100663296)},
            "limits": [
                "No training failure probability or long-horizon stability conclusion",
                "Training timing excludes validation, checkpoints, JSON writes, and nvidia-smi sampling",
                "Device/process usage is sampled at phase boundaries, not a device/process peak",
                "Allocator peaks reset per phase; reserved memory can be retained from earlier phases",
                "Checkpoint time includes torch.save and atomic rename, without an fsync durability guarantee",
                "Short-profile throughput and validation/checkpoint cost may vary over longer jobs"],
        }
        from src.training.reproduction import write_json
        write_json(self.output / "profile.json", report)
        return report
