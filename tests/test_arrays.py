"""One-GPU array cell mapping, frozen dispatch, resume and cost collection."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from scripts.array_jobs import collect_profiles, create_plan, load_plan, run_task


SOURCE = Path(__file__).resolve().parents[1]


def bash_binary():
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash required for array launcher integration")
    return str(bash)


def settings(tmp_path, **extra):
    return {"TRAIN_DATA_DIR": str(tmp_path / "train data"), "VAL_DATA_DIR": str(tmp_path / "validation data"),
            "OUTPUT_ROOT": str(tmp_path / "runs"), "PROFILE_ROOT": str(tmp_path / "profiles"), **extra}


def test_profile_array_maps_each_case_once_with_independent_paths(tmp_path):
    manifest = create_plan("profile", settings(tmp_path, PROFILE_CASES="512:2 512:4", PROFILE_DEPTHS="4 8"))
    plan = load_plan(manifest)
    assert plan["concurrency"] == 2 and len(plan["tasks"]) == 4
    assert [task["label"] for task in plan["tasks"]] == ["S512 micro=2 T4", "S512 micro=2 T8", "S512 micro=4 T4", "S512 micro=4 T8"]
    assert len({task["profile_output_root"] for task in plan["tasks"]}) == 4
    for task in plan["tasks"]:
        command = task["command"]
        start = command.index("--cases")
        assert command[start + 2] == "--depths"
        assert command[start + 4] == "--effective-tokens"
        assert command[command.index("--effective-tokens") + 1] == "16384"
    with pytest.raises(ValueError, match="outside"):
        run_task(manifest, 4, dry_run=True)


def test_training_array_requires_explicit_learning_rate_by_default(tmp_path):
    with pytest.raises(ValueError, match="Set LRS"):
        create_plan("reproduce", settings(tmp_path))


@pytest.mark.parametrize("extra,concurrency", [({"SEEDS": "101 101"}, 2), ({"LRS": "3e-4 0.0003"}, 2), ({}, 3)])
def test_array_rejects_duplicate_runs_and_excess_concurrency(tmp_path, extra, concurrency):
    with pytest.raises(ValueError):
        create_plan("reproduce", settings(tmp_path, MODE="reproduce", LRS="3e-4", **extra) if "LRS" not in extra else
                    settings(tmp_path, MODE="reproduce", **extra), concurrency=concurrency)


def test_array_dispatch_freezes_one_cell_and_uses_existing_resume(tmp_path, monkeypatch):
    root = tmp_path / "repo with spaces"
    (root / "scripts").mkdir(parents=True)
    for name in ("reproduce.sh", "load_cluster_env.sh"):
        shutil.copyfile(SOURCE / "scripts" / name, root / "scripts" / name)
    # If a queued task reloads this edited config, it would change the experiment.
    (root / "cluster.env").write_text('export SEEDS="999"\nexport DEPTHS="4 8"\nexport MODEL_CONFIG="small_repro"\n', encoding="utf-8")
    environment = settings(tmp_path, MODE="reproduce", MODEL_CONFIG="ouro_1_4b_repro", SEEDS="101", LRS="3e-4",
                           DEPTHS="4 8", MICRO_BATCH="1", GRAD_ACCUM="32", MAX_STEPS="6144")
    manifest = create_plan("reproduce", environment, root=root)
    plan = load_plan(manifest)
    assert len(plan["tasks"]) == 2
    old = Path(plan["tasks"][1]["expected_output_directory"])
    (old / "checkpoints").mkdir(parents=True)
    checkpoint = old / "checkpoints" / "step_0003072.pt"
    checkpoint.write_bytes(b"retained full training checkpoint")
    fake_python = root / "fake trainer"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "<%s>\\n" "$@"\n', encoding="utf-8", newline="\n")
    fake_python.chmod(0o755)
    captured = {}
    def execute(program, command, env):
        captured.update(env)
        # Replace only the training executable with a recorder; run the real
        # selected Bash launcher and its real checkpoint discovery/config loading.
        result = subprocess.run([bash_binary(), command[1]], env={**env, "PYTHON_BIN": fake_python.as_posix()},
                                cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        captured["stdout"] = result.stdout.replace("\\", "/")
    monkeypatch.setattr(os, "execvpe", execute)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SEEDS", "999")
    assert run_task(manifest, 1) == 0
    assert captured["SEEDS"] == "101" and captured["DEPTHS"] == "8"
    assert captured["OURO_ARRAY_FROZEN_ENV"] == "1"
    assert captured["stdout"].count("<--model-config>") == 1
    assert "<ouro_1_4b_repro>" in captured["stdout"]
    assert "<--resume-from>" in captured["stdout"] and "step_0003072.pt" in captured["stdout"]
    assert not Path(plan["tasks"][0]["expected_output_directory"]).exists()
    assert checkpoint.read_bytes() == b"retained full training checkpoint"
    assert len(list(manifest.parent.glob("dispatch_001_*.json"))) == 1


def test_collect_profiles_combines_independent_tasks_without_assuming_missing_depth(tmp_path):
    manifest = create_plan("profile", settings(tmp_path, PROFILE_CASES="512:2", PROFILE_DEPTHS="4 8"))
    plan = load_plan(manifest)
    def complete(task, rate):
        root = Path(task["profile_output_root"]) / "ouro_1_4b_repro" / "unique_run"
        root.mkdir(parents=True)
        case = {"status": "completed", "depth": 4 if task["id"] == 0 else 8, "sequence_length": 512,
                "microbatch": 2, "effective_target_tokens_per_update": 16384,
                "profile": {"steady_tokens_per_second": rate,
                            "operations": [{"kind": "endpoint_validation", "seconds": 5, "target_tokens": 32768},
                                           {"kind": "checkpoint", "seconds": 35}]}}
        (root / "profile_summary.json").write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    complete(plan["tasks"][0], 1400)
    assert collect_profiles(manifest) == 1
    partial = json.loads((manifest.parent / "array_profile_summary.json").read_text())
    assert not partial["complete"] and partial["paired_cost_projections"] == []
    complete(plan["tasks"][1], 700)
    assert collect_profiles(manifest) == 0
    report = json.loads((manifest.parent / "array_profile_summary.json").read_text())
    assert report["complete"] and len(report["paired_cost_projections"]) == 2
    cost = report["paired_cost_projections"][0]
    assert cost["pair_elapsed_hours_if_started_together_on_two_gpus"] == max(cost["per_depth_hours_with_scaled_overhead"].values())
    assert cost["pair_elapsed_hours_if_started_together_on_two_gpus"] < cost["pair_hours_with_scaled_validation_and_checkpoint_cost"]


def test_array_submission_dry_run_and_mock_sbatch(tmp_path):
    root = tmp_path / "array repo with spaces"
    (root / "scripts").mkdir(parents=True)
    for name in ("submit_array.sh", "array_jobs.py", "array_task.sh", "load_cluster_env.sh"):
        shutil.copyfile(SOURCE / "scripts" / name, root / "scripts" / name)
    env = os.environ.copy()
    for name in ("REPO_DIR", "CLUSTER_ENV_FILE", "PILOT_ACCOUNT", "PILOT_PARTITION", "PROFILE_CASES", "PROFILE_DEPTHS",
                 "OURO_ARRAY_FROZEN_ENV", "PROFILE_ROOT", "ARRAY_CONCURRENCY"):
        env.pop(name, None)
    env["PYTHONPATH"] = str(SOURCE)
    config = {"PYTHON_BIN": sys.executable.replace("\\", "/"), **settings(tmp_path),
              "PROFILE_CASES": "512:2 512:4", "PROFILE_DEPTHS": "4 8", "PILOT_ACCOUNT": ""}
    (root / "cluster.env").write_text(
        "".join(f"export {name}={shlex.quote(value)}\n" for name, value in config.items() if name != "PILOT_ACCOUNT")
        + 'export PILOT_ACCOUNT="${PILOT_ACCOUNT:-}"\n', encoding="utf-8")
    fake_bin = root / "bin"
    fake_bin.mkdir()
    marker = root / "submitted"
    fake = fake_bin / "sbatch"
    fake.write_text('#!/usr/bin/env bash\n[[ -d "$REPO_DIR/logs" ]] || exit 7\n'
                    'printf "<%s>\\n" "$@"\nprintf "submitted" > "$REPO_DIR/submitted"\n', encoding="utf-8", newline="\n")
    fake.chmod(0o755)
    env["PATH"] = fake_bin.as_posix() + (";" if os.name == "nt" else ":") + env["PATH"]
    def invoke(*arguments, extra=None):
        return subprocess.run([bash_binary(), str(root / "scripts/submit_array.sh"), *arguments],
                              cwd=root, env={**env, **(extra or {})}, capture_output=True, text=True)
    dry = invoke("profile", "--dry-run")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert "0-3%2" in dry.stdout and not marker.exists()
    submitted = invoke("profile", "--time=01:00:00")
    assert submitted.returncode == 0, submitted.stdout + submitted.stderr
    assert "<--array=0-3%2>" in submitted.stdout and "--account" not in submitted.stdout
    assert "<--time=01:00:00>" in submitted.stdout and marker.is_file()
    explicit = invoke("profile", extra={"PILOT_ACCOUNT": "chosen", "ARRAY_CONCURRENCY": "1"})
    assert "<--array=0-3%1>" in explicit.stdout and "<--account=chosen>" in explicit.stdout
    invalid = invoke("profile", "--array=0-99%8")
    assert invalid.returncode != 0
    header = (root / "scripts/array_task.sh").read_text()
    assert "#SBATCH --gres=gpu:h100-47:1" in header and "#SBATCH --nodes=1" in header
    assert "#SBATCH --output=logs/%x_%A_%a.out" in header
