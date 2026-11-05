"""Profile real optimizer work on tiny CPU models; never allocate a 1.4B model."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.profile_reproduction import commands, paired_estimates, parse_args
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.training.replay import TokenCorpus
from tests.test_reproduction import tiny_config, reproduction_arguments


def test_full_size_shapes_and_actual_meta_parameter_count_preserve_legacy():
    config = LoopLMConfig.ouro_1_4b_repro()
    assert (config.num_layers, config.hidden_size, config.num_heads, config.intermediate_size) == (24, 2048, 16, 5632)
    assert config.num_parameters() == 1434652673
    assert config.normalize_recurrent_state and config.fp32_gate and not config.tie_word_embeddings
    assert config.rope_base == 10000 and config.max_seq_len == 4096
    # Shape/count validation without allocating gigabytes or drawing full weights.
    with torch.device("meta"):
        model = LoopLM(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1434652673
    assert LoopLMConfig.depth24_repro().num_parameters() == 245442049
    legacy = LoopLMConfig.ouro_1_4b()
    assert legacy.intermediate_size == 5504 and legacy.tie_word_embeddings
    assert not legacy.normalize_recurrent_state and not legacy.fp32_gate


def test_profile_counts_whole_accumulated_updates_and_one_populated_checkpoint(tmp_path, monkeypatch):
    from src.training.reproduction import run_reproduction
    args = reproduction_arguments(tmp_path, monkeypatch)
    args.profile_warmup_updates = 1
    args.grad_accum_steps = 2
    args.save_every = args.val_every = 4
    assert run_reproduction(args, tiny_config()) == 0
    output = tmp_path / "run"
    report = json.loads((output / "profile.json").read_text())
    assert report["status"] == "completed"
    assert report["warmup_updates_completed"] == 1 and report["steady_updates"] == 2
    assert report["steady_target_tokens"] == 64  # 2 updates * 2 microsteps * 2 sequences * 8 targets
    assert report["steady_tokens_per_second"] > 0
    assert report["adam_moment_bytes_after_warmup"] > 0
    assert [row["kind"] for row in report["operations"]] == ["initial_validation", "endpoint_validation", "checkpoint"]
    assert report["operations"][-1]["file_bytes"] > 0
    checkpoints = list((output / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 1 and checkpoints[0].name == "step_0000003.pt"
    state = torch.load(checkpoints[0], weights_only=False)
    assert state["optimizer_state_dict"]["state"] and state["tokens_seen"] == 96
    rows = [json.loads(row) for row in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows if row["kind"] == "train"] == [1, 2, 3]
    assert all(len(row["per_step_losses"]) == 3 for row in rows if row["kind"] == "train")


def test_profile_rejects_resume_and_records_oom_without_stability_claim(tmp_path, monkeypatch):
    import src.training.reproduction as reproduction
    args = reproduction_arguments(tmp_path, monkeypatch)
    args.profile_warmup_updates = 1
    args.resume_from = "not-a-small-checkpoint.pt"
    with pytest.raises(ValueError, match="fresh initialization"):
        reproduction.run_reproduction(args, tiny_config())
    args.resume_from = None
    def out_of_memory(self, batch):
        raise torch.cuda.OutOfMemoryError("Injected profile OOM")
    monkeypatch.setattr(reproduction.Trainer, "train_step", out_of_memory)
    assert reproduction.run_reproduction(args, tiny_config()) == 43
    report = json.loads((tmp_path / "run" / "profile.json").read_text())
    assert report["status"] == "out_of_memory" and report["steady_updates"] == 0
    assert report["steady_tokens_per_second"] is None


def test_repack_preserves_verified_source_and_requires_explicit_opt_in(tmp_path):
    from src.training.corpus import prepare_corpora, WIKITEXT_DATASET, WIKITEXT_CONFIG
    class Tokenizer:
        eos_token_id = 0
        def __len__(self):
            return 32
        def encode(self, text, add_special_tokens=False):
            return [ord(character) % 31 + 1 for character in text]
    destination = tmp_path / "prepared"
    prepare_corpora({"train": [{"text": "training" * 20}],
                     "validation": [{"text": "heldout" * 20}], "test": [{"text": "test" * 20}]},
                    Tokenizer(), destination, {"train": 64, "validation": 64, "test": 64},
                    1024, {"dataset": WIKITEXT_DATASET, "config": WIKITEXT_CONFIG}, {}, seq_len=8)
    original = TokenCorpus(destination / "train", 8)
    with pytest.raises(ValueError, match="sequence length"):
        TokenCorpus(destination / "train", 16)
    longer = TokenCorpus(destination / "train", 16, repack_stream=True)
    assert original.manifest[0]["sha256"] == longer.manifest[0]["sha256"]
    assert original.preparation["manifest_sha256"] == longer.preparation["manifest_sha256"]
    assert original.preparation["records_sha256"] == longer.preparation["records_sha256"]
    assert "runtime_packing" not in original.preparation
    assert longer.preparation["runtime_packing"]["sequence_length"] == 16
    assert longer.preparation["runtime_packing"]["target_positions"] == len(longer) * 16
    assert longer.fingerprint != original.fingerprint
    # Repacking never bypasses shard verification.
    corrupted = tmp_path / "corrupted"
    shutil.copytree(destination, corrupted)
    np.save(corrupted / "train" / original.manifest[0]["name"], np.arange(200, dtype=np.uint16))
    with pytest.raises(ValueError, match="differ from manifest"):
        TokenCorpus(corrupted / "train", 16, repack_stream=True)


def test_driver_plan_has_safe_namespace_and_constant_target_batch(tmp_path):
    args = parse_args(["--data-dir", "train", "--val-data-dir", "val", "--output-root", str(tmp_path),
                       "--cases", "512:1", "512:4", "4096:1", "--depths", "8", "4"])
    plan = list(commands(args, tmp_path / "profiles"))
    assert len(plan) == 6
    assert len({item["output"] for item in plan}) == 6
    for item in plan:
        command = item["command"]
        value = lambda key: command[command.index(key) + 1]
        assert value("--model-config") == "ouro_1_4b_repro"
        assert "--resume-from" not in command and "--gradient-checkpointing" in command
        assert int(value("--seq-len")) * int(value("--batch-size")) * int(value("--grad-accum-steps")) == 16384
        assert int(value("--save-every")) > int(value("--max-steps"))
    with pytest.raises(SystemExit):
        parse_args(["--data-dir", "x", "--val-data-dir", "y", "--output-root", "z", "--cases", "4096:3"])


def test_budget_projection_requires_measured_matched_depths():
    def case(depth, micro=1):
        return {"status": "completed", "depth": depth, "microbatch": micro, "sequence_length": 512,
                "effective_target_tokens_per_update": 16384,
                "profile": {"steady_tokens_per_second": 1000 if depth == 8 else 2000,
                            "operations": [{"kind": "endpoint_validation", "seconds": 1, "target_tokens": 32768},
                                           {"kind": "checkpoint", "seconds": 1}]}}
    assert paired_estimates([case(8)]) == []
    assert paired_estimates([case(8), case(4, micro=2)]) == []
    projection = paired_estimates([case(4), case(8)])[0]
    assert projection["pair_training_only_hours"] == pytest.approx(20.97152)
    assert projection["pair_hours_with_scaled_validation_and_checkpoint_cost"] > projection["pair_training_only_hours"]


def test_cluster_wrappers_preserve_old_runs_and_isolate_large_profiles(tmp_path):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash is required for launcher integration")
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "repo with spaces"
    (root / "scripts").mkdir(parents=True)
    for name in ("load_cluster_env.sh", "reproduce.sh", "profile_reproduction.sh", "submit_profile.sh"):
        shutil.copyfile(source / "scripts" / name, root / "scripts" / name)
    fake_python = root / "fake python"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "<%s>\\n" "$@"\n', encoding="utf-8", newline="\n")
    fake_python.chmod(0o755)
    (root / "cluster.env").write_text(
        f'export PYTHON_BIN={shlex.quote(fake_python.as_posix())}\n'
        f'export OUTPUT_ROOT={shlex.quote((root / "runs").as_posix())}\n'
        'export TRAIN_DATA_DIR="train data"\nexport VAL_DATA_DIR="val data"\n', encoding="utf-8")
    env = os.environ.copy()
    for name in ("CLUSTER_ENV_FILE", "REPO_DIR", "MODEL_CONFIG", "SEQ_LEN", "PROFILE_MODEL_CONFIG",
                 "PROFILE_ROOT", "PROFILE_DEPTHS", "PROFILE_CASES", "REPACK_STREAM", "PILOT_ACCOUNT", "PILOT_PARTITION"):
        env.pop(name, None)
    env.update(MODE="reproduce", LRS="3e-4", SEEDS="101", MAX_STEPS="3", DEPTHS="8")
    def run(script, extra=None, *arguments):
        result = subprocess.run([str(bash), str(root / "scripts" / script), *arguments], env={**env, **(extra or {})},
                                cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.replace("\\", "/")
    old = root / "runs" / "reproduce" / "seed_101" / "lr_3e-4" / "T8"
    (old / "checkpoints").mkdir(parents=True)
    (old / "checkpoints" / "step_0000001.pt").write_bytes(b"existing small checkpoint")
    original = run("reproduce.sh")
    assert "<small_repro>" in original and "<--resume-from>" in original
    large = run("reproduce.sh", {"MODEL_CONFIG": "ouro_1_4b_repro", "SEQ_LEN": "1024", "REPACK_STREAM": "1"})
    assert "<ouro_1_4b_repro>" in large and "<--resume-from>" not in large
    assert "/models/ouro_1_4b_repro/seq_1024/reproduce/seed_101/" in large
    assert "<--repack-stream>" in large and "<--keep-checkpoints>\n<1>" in large
    profile = run("profile_reproduction.sh", {"PROFILE_CASES": "512:2 4096:1", "PROFILE_DEPTHS": "4 8"}, "--dry-run")
    assert "<--cases>\n<512:2>\n<4096:1>" in profile
    assert "<--depths>\n<4>\n<8>" in profile and "<--dry-run>" in profile
    # Submission must create logs first and omit account unless explicitly set.
    fake_bin = root / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text('#!/usr/bin/env bash\n[[ -d "$REPO_DIR/logs" ]] || exit 7\nprintf "<%s>\\n" "$@"\n',
                      encoding="utf-8", newline="\n")
    sbatch.chmod(0o755)
    path_override = fake_bin.as_posix() + (";" if os.name == "nt" else ":") + env["PATH"]
    submitted = run("submit_profile.sh", {"PATH": path_override}, "--time=01:00:00")
    assert "--account" not in submitted and "<--time=01:00:00>" in submitted
    explicit = run("submit_profile.sh", {"PATH": path_override, "PILOT_ACCOUNT": "explicit-account"})
    assert "<--account=explicit-account>" in explicit
    assert (old / "checkpoints" / "step_0000001.pt").read_bytes() == b"existing small checkpoint"


@pytest.mark.parametrize("timeout", [False, True])
def test_profile_parent_records_child_outcome_and_cleans_only_its_checkpoint(tmp_path, monkeypatch, timeout):
    import scripts.profile_reproduction as driver
    old = tmp_path / "existing_small.pt"
    old.write_bytes(b"untouched")
    def fake_commands(args, output):
        directory = output / "case"
        # Exercise the real subprocess/deadline/summary/cleanup machinery, without
        # allocating any model. A timeout must preserve the child's partial log.
        script = ("import json,time; from pathlib import Path; "
                  "p=Path(" + repr(str(directory)) + "); "
                  "(p/'checkpoints').mkdir(); (p/'checkpoints'/'step_0000006.pt').write_bytes(b'profile'); "
                  "print('child started', flush=True); ")
        if timeout:
            script += "time.sleep(30)"
        else:
            profile = {"status": "completed", "steady_updates": args.measure_updates,
                       "steady_tokens_per_second": 123.0, "max_measured_peak_allocated_bytes": 0,
                       "operations": []}
            script += "(p/'profile.json').write_text(" + repr(json.dumps(profile)) + ")"
        yield {"sequence_length": 512, "microbatch": 1, "depth": 8,
               "effective_target_tokens_per_update": 16384, "output": str(directory),
               "command": [sys.executable, "-c", script]}
    monkeypatch.setattr(driver, "commands", fake_commands)
    code = driver.main(["--data-dir", "unused", "--val-data-dir", "unused",
                        "--output-root", str(tmp_path / "profiles"), "--case-timeout-seconds", "1"])
    summary_path = next((tmp_path / "profiles").rglob("profile_summary.json"))
    summary = json.loads(summary_path.read_text())
    case = summary["cases"][0]
    assert code == (1 if timeout else 0)
    assert case["status"] == ("timed_out" if timeout else "completed")
    assert len(case["removed_profile_checkpoints"]) == 1
    assert not list(Path(case["output"]).rglob("*.pt"))
    assert "child started" in (Path(case["output"]) / "console.log").read_text()
    assert old.read_bytes() == b"untouched"
