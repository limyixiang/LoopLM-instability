#!/usr/bin/env python
"""Freeze a matrix, dispatch one cell per SLURM task, and collect profile costs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def positive(value, name):
    if not str(value).isdigit() or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def words(value, name, convert=str):
    items = [convert(item) for item in value.split()]
    if not items or len(set(items)) != len(items):
        raise ValueError(f"{name} must be nonempty and contain no duplicate cells")
    return items


def create_plan(kind, environment, concurrency=2, root=ROOT):
    if concurrency not in (1, 2):
        raise ValueError("Array concurrency must be 1 or 2")
    root = Path(root).resolve()
    env = dict(environment)
    for name in ("TRAIN_DATA_DIR", "VAL_DATA_DIR"):
        if not env.get(name):
            raise ValueError(f"Set {name} before creating an array")
    # Module/PATH state is loaded in the batch wrapper. Freeze experiment inputs
    # and interpreter here so queued cells cannot change when cluster.env changes.
    frozen = {name: env[name] for name in (
        "HF_HOME", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS") if name in env}
    frozen.update(REPO_DIR=str(root), PYTHON_BIN=sys.executable,
                  TRAIN_DATA_DIR=str(Path(env["TRAIN_DATA_DIR"]).resolve()),
                  VAL_DATA_DIR=str(Path(env["VAL_DATA_DIR"]).resolve()))
    output_root = Path(env.get("OUTPUT_ROOT", root / "runs/reproduction")).resolve()
    identity = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    tasks = []
    if kind == "profile":
        from scripts.profile_reproduction import parse_args
        profile_root = Path(env.get("PROFILE_ROOT", output_root / "profiles")).resolve()
        destination = profile_root / "arrays" / identity
        arguments = ["--model-config", env.get("PROFILE_MODEL_CONFIG", "ouro_1_4b_repro"),
                     "--data-dir", frozen["TRAIN_DATA_DIR"], "--val-data-dir", frozen["VAL_DATA_DIR"],
                     "--output-root", str(destination), "--cases", *env.get("PROFILE_CASES", "512:1").split(),
                     "--depths", *env.get("PROFILE_DEPTHS", "8").split(),
                     "--effective-tokens", env.get("PROFILE_EFFECTIVE_TOKENS", "16384"),
                     "--warmup-updates", env.get("PROFILE_WARMUP_UPDATES", "2"),
                     "--measure-updates", env.get("PROFILE_MEASURE_UPDATES", "4"),
                     "--budget-seconds", env.get("PROFILE_BUDGET_SECONDS", "3600"),
                     "--case-timeout-seconds", env.get("PROFILE_CASE_TIMEOUT_SECONDS", "1800")]
        args = parse_args(arguments)
        # Each task calls the existing bounded profiler with exactly one case.
        for seq, micro in args.case_shapes:
            for depth in args.depths:
                task_id = len(tasks)
                task_root = destination / f"task_{task_id:03d}"
                command = [sys.executable, "-u", str(root / "scripts/profile_reproduction.py"),
                           "--model-config", args.model_config, "--data-dir", args.data_dir,
                           "--val-data-dir", args.val_data_dir, "--output-root", str(task_root),
                           "--cases", f"{seq}:{micro}", "--depths", str(depth),
                           "--effective-tokens", str(args.effective_tokens),
                           "--warmup-updates", str(args.warmup_updates), "--measure-updates", str(args.measure_updates),
                           "--budget-seconds", str(args.budget_seconds),
                           "--case-timeout-seconds", str(args.case_timeout_seconds)]
                tasks.append({"id": task_id, "label": f"S{seq} micro={micro} T{depth}",
                              "command": command, "environment": frozen.copy(), "profile_output_root": str(task_root)})
    elif kind == "reproduce":
        mode = env.get("MODE", "reproduce")
        if mode not in ("screen", "reproduce"):
            raise ValueError("MODE must be screen or reproduce")
        if mode == "reproduce" and not env.get("LRS"):
            raise ValueError("Set LRS to the selected learning rate")
        seeds = words(env.get("SEEDS", "0" if mode == "screen" else "101 202 303"), "SEEDS", int)
        if min(seeds) < 0:
            raise ValueError("Seeds must be nonnegative")
        depths = words(env.get("DEPTHS", "4 8"), "DEPTHS", lambda value: positive(value, "depth"))
        lrs = words(env.get("LRS", "1e-4 3e-4"), "LRS")
        if len({float(lr) for lr in lrs}) != len(lrs) or any(not math.isfinite(float(lr)) or float(lr) <= 0 for lr in lrs):
            raise ValueError("Learning rates must be finite, positive and distinct")
        model = env.get("MODEL_CONFIG", "small_repro")
        if model not in ("small_repro", "ouro_1_4b_repro", "depth24_repro"):
            raise ValueError("Choose a reproduction-safe model preset")
        seq = positive(env.get("SEQ_LEN", "512"), "SEQ_LEN")
        if seq > 4096:
            raise ValueError("SEQ_LEN must not exceed 4096")
        frozen.update(MODE=mode, MODEL_CONFIG=model, SEQ_LEN=str(seq), OUTPUT_ROOT=str(output_root),
                      MAX_STEPS=str(positive(env.get("MAX_STEPS", "512" if mode == "screen" else "3072"), "MAX_STEPS")),
                      MICRO_BATCH=str(positive(env.get("MICRO_BATCH", "2"), "MICRO_BATCH")),
                      GRAD_ACCUM=str(positive(env.get("GRAD_ACCUM", "16"), "GRAD_ACCUM")),
                      KEEP_CHECKPOINTS=str(positive(env.get("KEEP_CHECKPOINTS", "5" if model == "small_repro" else "1"), "KEEP_CHECKPOINTS")),
                      KEEP_EVENT_CHECKPOINTS=str(positive(env.get("KEEP_EVENT_CHECKPOINTS", "3" if model == "small_repro" else "1"), "KEEP_EVENT_CHECKPOINTS")),
                      REPACK_STREAM=env.get("REPACK_STREAM", "0"))
        if frozen["REPACK_STREAM"] not in ("0", "1"):
            raise ValueError("REPACK_STREAM must be 0 or 1")
        run_root = output_root if model == "small_repro" and seq == 512 else output_root / "models" / model / f"seq_{seq}"
        destination = output_root / "array_plans" / identity
        for seed in seeds:
            for lr in lrs:
                for depth in depths:
                    cell = {**frozen, "SEEDS": str(seed), "LRS": lr, "DEPTHS": str(depth)}
                    tasks.append({"id": len(tasks), "label": f"seed={seed} lr={lr} T{depth}",
                                  "command": ["bash", str(root / "scripts/reproduce.sh")], "environment": cell,
                                  "expected_output_directory": str(run_root / mode / f"seed_{seed}" / f"lr_{lr}" / f"T{depth}")})
    else:
        raise ValueError("kind must be profile or reproduce")
    assert tasks and [task["id"] for task in tasks] == list(range(len(tasks)))
    destinations = [task.get("profile_output_root", task.get("expected_output_directory")) for task in tasks]
    assert len(set(destinations)) == len(tasks)
    destination.mkdir(parents=True, exist_ok=False)
    plan = {"schema_version": 1, "kind": kind, "concurrency": concurrency, "repo": str(root),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tasks": tasks,
            "scope": "Independent one-GPU cells; arrays reduce elapsed waiting, not aggregate GPU-hours or per-run memory"}
    manifest = destination / "manifest.json"
    write_json(manifest, plan)
    return manifest


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("concurrency") not in (1, 2):
        raise ValueError("Invalid array manifest")
    if [task["id"] for task in plan["tasks"]] != list(range(len(plan["tasks"]))) or not plan["tasks"]:
        raise ValueError("Invalid array task mapping")
    return plan


def run_task(manifest, task_id, dry_run=False):
    manifest = Path(manifest).resolve()
    plan = load_plan(manifest)
    if not 0 <= task_id < len(plan["tasks"]):
        raise ValueError("Array index is outside the frozen matrix")
    task = plan["tasks"][task_id]
    if dry_run:
        print(json.dumps(task, indent=2))
        return 0
    environment = {**os.environ, **task["environment"], "OURO_ARRAY_FROZEN_ENV": "1", "PYTHONUNBUFFERED": "1"}
    write_json(manifest.parent / f"dispatch_{task_id:03d}_{uuid.uuid4().hex[:8]}.json",
               {"task_id": task_id, "label": task["label"], "host": socket.gethostname(),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "slurm": {name: os.environ.get(name) for name in (
                    "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES")}})
    os.chdir(plan["repo"])
    os.execvpe(task["command"][0], task["command"], environment)
    return 0


def collect_profiles(manifest):
    manifest = Path(manifest).resolve()
    plan = load_plan(manifest)
    if plan["kind"] != "profile":
        raise ValueError("Profile collection requires a profile array")
    from scripts.profile_reproduction import paired_estimates
    cases = []
    for task in plan["tasks"]:
        summaries = list(Path(task["profile_output_root"]).rglob("profile_summary.json"))
        if len(summaries) > 1:
            raise ValueError("A task has multiple invocations; inspect its lineage before collecting")
        if not summaries:
            cases.append({"array_task_id": task["id"], "label": task["label"], "status": "not_finished"})
        else:
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            if len(summary["cases"]) != 1:
                raise ValueError("An array task must contain exactly one profile case")
            cases.append({**summary["cases"][0], "array_task_id": task["id"], "label": task["label"]})
    report = {"manifest": str(manifest), "cases": cases, "paired_cost_projections": paired_estimates(cases),
              "complete": all(case["status"] == "completed" for case in cases),
              "scope": "Short concurrent profiles; timing/I/O can reflect contention. Pair sums are aggregate GPU-hours; parallel pair elapsed estimates assume both tasks start together, excluding queue delays."}
    destination = manifest.parent / "array_profile_summary.json"
    write_json(destination, report)
    print(f"Collected profile results: {destination}")
    for case in cases:
        profile = case.get("profile") or {}
        print(f"task {case['array_task_id']}: {case['label']}: {case['status']}; tokens/s={profile.get('steady_tokens_per_second')}")
    return 0 if report["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("--kind", choices=["profile", "reproduce"], required=True)
    create.add_argument("--concurrency", type=int, default=2)
    for name in ("count", "describe", "run", "collect"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name == "run":
            command.add_argument("--task-id", type=int, required=True)
            command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "create":
        print(create_plan(args.kind, os.environ, args.concurrency))
    elif args.action in ("count", "describe"):
        plan = load_plan(args.manifest)
        if args.action == "count":
            print(len(plan["tasks"]))
        else:
            for task in plan["tasks"]:
                print(f"task {task['id']}: {task['label']}")
    elif args.action == "run":
        return run_task(args.manifest, args.task_id, args.dry_run)
    else:
        return collect_profiles(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
