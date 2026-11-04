"""Read back real event files; optional viewer dependency, synthetic metrics only."""

import json
import pytest

pytest.importorskip("tensorboard")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scripts.export_tensorboard import export_tensorboard


def write_run(root, rows):
    run = root / "seed_0" / "lr_3e-4" / "T8"
    run.mkdir(parents=True)
    path = run / "metrics.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run / "result.json").write_text(json.dumps({"attempt_id": rows[-1]["attempt_id"], "status": "completed"}))
    return path


def events(root, name):
    return EventAccumulator(str(root / "seed_0/lr_3e-4/T8" / name),
                            size_guidance={"scalars": 0}).Reload()


def test_values_attempts_and_probe_sizes_are_preserved(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    path = write_run(source, [
        {"attempt_id": "first", "kind": "train", "step": 10, "uniform_ce": 3.25,
         "grad_norm": 8.0, "clipped": True, "per_step_losses": [3.5, 3.0],
         "raw_state_rms": [1.0, 2.0], "events": ["loss_spike"]},
        {"attempt_id": "resumed", "kind": "train", "step": 10, "uniform_ce": 2.75, "events": []},
        {"attempt_id": "resumed", "kind": "validation", "step": 10, "uniform_ce": 2.5, "valid_tokens": 128},
        {"attempt_id": "resumed", "kind": "endpoint_validation", "step": 10, "uniform_ce": 2.6, "valid_tokens": 2048},
    ])
    before = path.read_bytes()
    report = export_tensorboard(source, output)
    first, resumed = events(output, "attempt_first"), events(output, "attempt_resumed")
    assert first.Scalars("train/uniform_ce")[0].value == 3.25
    assert resumed.Scalars("train/uniform_ce")[0].value == 2.75
    assert first.Scalars("train/per_step_losses/exit_2")[0].value == 3.0
    assert first.Scalars("train/raw_state_rms/loop_2")[0].value == 2.0
    assert first.Scalars("train/events/loss_spike")[0].value == 1
    assert resumed.Scalars("train/events/loss_spike")[0].value == 0
    assert resumed.Scalars("validation/valid_tokens")[0].value == 128
    assert resumed.Scalars("endpoint_validation/valid_tokens")[0].value == 2048
    assert path.read_bytes() == before
    assert [run["matches_final_result_attempt"] for run in report["runs"]] == [False, True]
    assert (output / "export.json").is_file()


def test_nonfinite_values_and_numerical_failure_are_visible(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    write_run(source, [
        {"attempt_id": "a", "kind": "train", "step": 1, "uniform_ce": "nan", "events": []},
        {"attempt_id": "a", "kind": "numerical_divergence", "step": 1, "message": "nonfinite gradient"},
    ])
    report = export_tensorboard(source, output)
    record = events(output, "attempt_a")
    assert report["nonfinite_values_skipped"] == 1
    assert "train/uniform_ce" not in record.Tags()["scalars"]
    assert record.Scalars("export/nonfinite_metric")[0].value == 1
    assert record.Scalars("numerical_divergence/events/numerical_divergence")[0].value == 1


def test_reexport_refuses_to_append_duplicate_curves(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    write_run(source, [{"attempt_id": "a", "kind": "train", "step": 1, "uniform_ce": 3.0}])
    export_tensorboard(source, output)
    before = sorted(output.rglob("events.out.tfevents.*"))
    with pytest.raises(ValueError, match="empty output"):
        export_tensorboard(source, output)
    assert sorted(output.rglob("events.out.tfevents.*")) == before


def test_malformed_log_is_reported_without_changing_it(tmp_path):
    source = tmp_path / "input"
    path = write_run(source, [{"attempt_id": "a", "kind": "train", "step": 1}])
    with path.open("a") as stream:
        stream.write('{"unfinished":')
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Invalid JSON"):
        export_tensorboard(source, tmp_path / "output")
    assert path.read_bytes() == before
