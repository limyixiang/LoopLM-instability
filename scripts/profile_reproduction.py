#!/usr/bin/env python
"""Bounded, fresh-process profiles of the real joint-objective reproduction path.

Default: one Ouro-1.4B-size T8 case, context 512, microbatch 1. No weights
downloaded, no old checkpoints resumed, no automatic model-size sweep.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.config import LoopLMConfig
from src.training.reproduction import write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", choices=["ouro_1_4b_repro", "depth24_repro"], default="ouro_1_4b_repro")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cases", nargs="+", default=["512:1"], metavar="SEQ:MICRO")
    parser.add_argument("--depths", nargs="+", type=int, choices=[4, 8], default=[8])
    parser.add_argument("--effective-tokens", type=int, default=16384)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measure-updates", type=int, default=4)
    parser.add_argument("--case-timeout-seconds", type=int, default=1800)
    parser.add_argument("--budget-seconds", type=int, default=3600)
    parser.add_argument("--val-tokens", type=int, default=8192)
    parser.add_argument("--endpoint-val-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--keep-checkpoint", action="store_true", help="Retain each large profile checkpoint after measurement")
    parser.add_argument("--dry-run", action="store_true", help="Write plan/commands without constructing a model or running CUDA")
    args = parser.parse_args(argv)
    if min(args.effective_tokens, args.warmup_updates, args.measure_updates,
           args.case_timeout_seconds, args.budget_seconds, args.val_tokens, args.endpoint_val_tokens) < 1:
        parser.error("Budgets, tokens, and update counts must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("LR must be finite and positive")
    args.case_shapes = []
    for case in args.cases:
        try:
            seq, micro = map(int, case.split(":"))
            if not 1 <= seq <= 4096 or micro < 1 or args.effective_tokens % (seq * micro):
                raise ValueError
        except ValueError:
            parser.error("Cases must be SEQ:MICRO, 1 <= SEQ <= 4096, with SEQ*MICRO dividing effective tokens")
        args.case_shapes.append((seq, micro))
    if len(set(args.case_shapes)) != len(args.case_shapes) or len(set(args.depths)) != len(args.depths):
        parser.error("Duplicate shapes/depths do not constitute independent profiles")
    return args


def commands(args, output):
    steps = args.warmup_updates + args.measure_updates
    for seq, micro in args.case_shapes:
        for depth in args.depths:
            directory = output / f"S{seq}_M{micro}_A{args.effective_tokens // (seq * micro)}_T{depth}"
            command = [sys.executable, "-u", str(Path(__file__).with_name("train.py")),
                       "--reproduction", "--device", "cuda", "--model-config", args.model_config,
                       "--num-recurrent-steps", str(depth), "--seq-len", str(seq), "--batch-size", str(micro),
                       "--grad-accum-steps", str(args.effective_tokens // (seq * micro)),
                       "--data-dir", args.data_dir, "--val-data-dir", args.val_data_dir,
                       "--seed", str(args.seed), "--data-seed", str(args.seed),
                       "--lr", str(args.lr), "--warmup-steps", "200", "--beta-kl", "0.1",
                       "--profile-warmup-updates", str(args.warmup_updates), "--max-steps", str(steps),
                       "--gradient-checkpointing", "--repack-stream", "--log-every", "1",
                       "--save-every", str(steps + 1), "--keep-checkpoints", "1", "--keep-event-checkpoints", "1",
                       "--val-every", str(steps + 1), "--val-tokens", str(args.val_tokens),
                       "--endpoint-val-tokens", str(args.endpoint_val_tokens), "--output-dir", str(directory)]
            yield {"sequence_length": seq, "microbatch": micro, "depth": depth,
                   "effective_target_tokens_per_update": args.effective_tokens,
                   "output": str(directory), "command": command}


def paired_estimates(cases):
    """Only pair measured depths with identical sequence/microbatch/update size."""
    estimates = []
    groups = {}
    for case in cases:
        if case.get("status") == "completed" and case.get("profile", {}).get("steady_tokens_per_second"):
            key = case["sequence_length"], case["microbatch"], case["effective_target_tokens_per_update"]
            groups.setdefault(key, {})[case["depth"]] = case["profile"]
    for (seq, micro, effective), depths in groups.items():
        if set(depths) != {4, 8}:
            continue
        for target in (50331648, 100663296):
            hours = 0.0
            training_hours = 0.0
            for profile in depths.values():
                training = target / profile["steady_tokens_per_second"]
                validation = next(item for item in profile["operations"] if item["kind"] == "endpoint_validation")
                checkpoint = next(item for item in profile["operations"] if item["kind"] == "checkpoint")
                updates = math.ceil(target / effective)
                # Initial + every-100 probes + 2M endpoint. Use measured per-token cost.
                val_seconds = ((1 + updates // 100) * 131072 + 2000000) * validation["seconds"] / validation["target_tokens"]
                # Conservative: count initial as an Adam-populated write too.
                checkpoint_seconds = (updates // 100 + 2) * checkpoint["seconds"]
                training_hours += training / 3600
                hours += (training + val_seconds + checkpoint_seconds) / 3600
            estimates.append({"sequence_length": seq, "microbatch": micro, "target_tokens_per_run": target,
                              "pair_training_only_hours": training_hours,
                              "pair_hours_with_scaled_validation_and_checkpoint_cost": hours,
                              "two_seed_pairs_hours_before_replays_and_setup": 2 * hours,
                              "scope": "Projection, not measured long-run duration; excludes setup, logging/sampling overhead, contention and replay"})
    return estimates


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.output_root).resolve()
    config = getattr(LoopLMConfig, args.model_config)()
    output = root / args.model_config / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    plan = {"purpose": "performance_profile_not_instability_replication", "arguments": vars(args),
            "parameters": config.num_parameters(), "fp32_parameters_gradients_adam_bytes": config.num_parameters() * 16,
            "estimated_populated_checkpoint_tensor_bytes": config.num_parameters() * 12,
            "cases": list(commands(args, output)),
            "scope": "Random initialization; no pretrained weights; Stage-1-style RoPE 10000; explicit context/batch adaptations"}
    write_json(output / "profile_plan.json", plan)
    print(f"Profile plan: {output / 'profile_plan.json'}", flush=True)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    started, results = time.perf_counter(), []
    for case in plan["cases"]:
        remaining = args.budget_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            results.append({**case, "status": "budget_exhausted"})
            break
        directory = Path(case["output"])
        directory.mkdir()
        free = shutil.disk_usage(directory).free
        if free < plan["estimated_populated_checkpoint_tensor_bytes"] + 2 * 1024**3:
            results.append({**case, "status": "insufficient_checkpoint_space", "free_disk_bytes": free})
            break
        timeout = min(args.case_timeout_seconds, remaining)
        print(f"Profiling S={case['sequence_length']} micro={case['microbatch']} T={case['depth']}; log: {directory / 'console.log'}", flush=True)
        tick = time.perf_counter()
        code, timed_out = None, False
        with (directory / "console.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(case["command"], stdout=log, stderr=subprocess.STDOUT,
                                         timeout=timeout, check=False, cwd=Path(__file__).resolve().parents[1])
                code = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True  # subprocess.run kills and waits for this fresh child.
        profile_path = directory / "profile.json"
        profile = json.loads(profile_path.read_text()) if profile_path.is_file() else None
        tail = (directory / "console.log").read_bytes()[-65536:].decode("utf-8", errors="replace")
        status = ("timed_out" if timed_out else "completed" if code == 0 else
                  "out_of_memory" if code == 43 or "OutOfMemoryError" in tail or "CUDA out of memory" in tail else "error")
        if status == "completed" and (not profile or profile.get("status") != "completed"
                                      or profile.get("steady_updates") != args.measure_updates):
            status = "incomplete_measurements"
        removed = []
        if not args.keep_checkpoint:
            # Only files created in this unique case; no recursive deletion or old run paths.
            for checkpoint in (directory / "checkpoints").glob("*.pt"):
                if checkpoint.resolve().parent != (directory / "checkpoints").resolve():
                    raise ValueError("Profile checkpoint escaped its case directory")
                removed.append(str(checkpoint))
                checkpoint.unlink()
            for temporary in (directory / "checkpoints").glob("*.tmp"):
                if temporary.resolve().parent == (directory / "checkpoints").resolve():
                    temporary.unlink()
        result = {**case, "status": status, "returncode": code, "elapsed_seconds": time.perf_counter() - tick,
                  "profile": profile, "removed_profile_checkpoints": removed}
        write_json(directory / "case_status.json", result)
        results.append(result)
        write_json(output / "profile_summary.json", {"cases": results, "paired_cost_projections": paired_estimates(results)})
        if profile:
            print(f"{status}; steady tokens/s={profile['steady_tokens_per_second']}; peak allocated={profile['max_measured_peak_allocated_bytes']} bytes", flush=True)
        else:
            print(f"{status}; inspect {directory / 'console.log'}", flush=True)
        # Do not spend the remaining allocation repeating a timeout or a code/config error.
        if status not in ("completed", "out_of_memory"):
            break
    summary = {"cases": results, "paired_cost_projections": paired_estimates(results),
               "elapsed_seconds": time.perf_counter() - started,
               "scope": "Short performance measurements; independent stability replication and repair remain untested"}
    write_json(output / "profile_summary.json", summary)
    print(f"Review: {output / 'profile_summary.json'}", flush=True)
    return 0 if results and all(case["status"] == "completed" for case in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
