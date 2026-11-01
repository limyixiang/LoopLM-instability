"""Scientific controls for the joint-objective reproduction path (CPU, no network)."""

import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM, compute_exit_distribution
from src.model.transformer import RMSNorm
from src.training.data import PreTokenizedDataset
from src.training.objectives import compute_looplm_loss
from src.training.replay import TokenCorpus, ReplayableBatchStream
from src.training.reproduction import InstabilityMonitor, evaluate_corpus
from src.training.trainer import Trainer, TrainerConfig, TrainingDivergence
from src.training.corpus import prepare_corpora, WIKITEXT_DATASET, WIKITEXT_CONFIG


def tiny_config(**overrides):
    values = dict(vocab_size=32, hidden_size=16, num_layers=1, num_heads=2,
                  intermediate_size=32, max_seq_len=8, max_recurrent_steps=3,
                  tie_word_embeddings=False, normalize_recurrent_state=True, fp32_gate=True)
    return LoopLMConfig(**{**values, **overrides})


def trainer(tmp_path, **overrides):
    values = dict(device="cpu", seed=101, lr=1e-3, checkpoint_dir=str(tmp_path))
    return Trainer(tiny_config(), TrainerConfig(**{**values, **overrides}))


def corpus_at(path, offset=0):
    path.mkdir()
    np.save(path / "tokens.npy", (np.arange(9 * 40, dtype=np.uint16) + offset) % 32)
    return TokenCorpus(path, seq_len=8)


def test_legacy_architecture_defaults_and_parameter_counts():
    legacy = LoopLMConfig.small()
    assert legacy.tie_word_embeddings and not legacy.normalize_recurrent_state
    assert LoopLMConfig.small_repro().num_parameters() == 117984769
    for tied in (True, False):
        config = tiny_config(tie_word_embeddings=tied)
        model = LoopLM(config)
        assert config.num_parameters() == sum(parameter.numel() for parameter in model.parameters())
        assert (model.embed.weight is model.lm_head.weight) == tied


def test_released_recurrence_contract_normalizes_gate_and_next_loop():
    model = LoopLM(tiny_config())
    gate_inputs, loop_inputs, normalized = [], [], []
    handles = [
        model.exit_gate.register_forward_pre_hook(lambda module, inputs: gate_inputs.append(inputs[0].detach().clone())),
        model.layers[0].register_forward_pre_hook(lambda module, inputs: loop_inputs.append(inputs[0].detach().clone())),
        model.final_norm.register_forward_hook(lambda module, inputs, result: normalized.append(result.detach().clone())),
    ]
    ids = torch.arange(8).view(1, 8)
    result = model(ids)
    for handle in handles:
        handle.remove()
    assert len(normalized) == 3  # One boundary norm per loop; no second readout norm.
    for step in range(3):
        torch.testing.assert_close(gate_inputs[step], normalized[step])
        torch.testing.assert_close(result.logits[step], model.lm_head(normalized[step]))
        if step:
            torch.testing.assert_close(loop_inputs[step], normalized[step - 1])


def test_joint_backprop_reaches_earlier_recurrent_uses():
    model = LoopLM(tiny_config())
    states = []
    def record(module, inputs, output):
        output.retain_grad()
        states.append(output)
    handle = model.layers[0].register_forward_hook(record)
    result = model(torch.arange(8).view(1, 8))
    result.logits[-1].square().mean().backward()
    handle.remove()
    assert all(state.grad is not None and state.grad.norm() > 0 for state in states)


def test_low_precision_norm_and_survival_are_numerically_usable():
    norm = RMSNorm(4)
    value = torch.full((2, 4), 1000.0, dtype=torch.float16)
    torch.testing.assert_close(norm(value), torch.ones(2, 4))
    gates = [torch.full((2, 3), 0.9, dtype=torch.bfloat16, requires_grad=True) for _ in range(16)]
    probabilities = compute_exit_distribution(gates)
    assert probabilities.dtype == torch.float32
    torch.testing.assert_close(probabilities.sum(0), torch.ones(2, 3))
    probabilities[-1].sum().backward()
    assert torch.isfinite(gates[0].grad).all()


def test_accumulation_matches_full_batch_metrics_and_update(tmp_path):
    full = trainer(tmp_path / "full")
    accumulated = trainer(tmp_path / "accumulated", grad_accum_steps=2)
    batch = torch.arange(4 * 9).reshape(4, 9) % 32
    expected = full.train_step(batch)
    assert not accumulated.train_step(batch[:2])["_stepped"]
    actual = accumulated.train_step(batch[2:])
    for key in ("loss", "task_loss", "entropy", "uniform_ce", "per_step_losses", "exit_probs"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-5)
    for first, second in zip(full.model.parameters(), accumulated.model.parameters()):
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-4)
    assert actual["valid_tokens"] == 32 and accumulated.tokens_seen == 32


def test_warmup_is_identical_when_extending_horizon(tmp_path):
    first = trainer(tmp_path / "short", warmup_steps=4, max_steps=5)
    second = trainer(tmp_path / "long", warmup_steps=4, max_steps=20)
    batch = torch.arange(18).reshape(2, 9) % 32
    for expected in (0.00025, 0.0005, 0.00075, 0.001, 0.001):
        assert first.train_step(batch)["lr"] == pytest.approx(expected)
        assert second.train_step(batch)["lr"] == pytest.approx(expected)


def test_checkpoint_replays_data_rng_and_optimizer_update(tmp_path):
    corpus = corpus_at(tmp_path / "data")
    original = trainer(tmp_path / "original", warmup_steps=4)
    stream = ReplayableBatchStream(corpus, 2, seed=17)
    original.train_step(next(stream))
    checkpoint = original.save_checkpoint({"stream": stream.state_dict()})
    expected_random = (random.random(), np.random.rand(), torch.rand(1))
    batch = next(stream)
    expected = original.train_step(batch)
    resumed = trainer(tmp_path / "resumed", warmup_steps=4)
    resumed.load_checkpoint(checkpoint)
    actual_random = (random.random(), np.random.rand(), torch.rand(1))
    assert expected_random[:2] == actual_random[:2]
    torch.testing.assert_close(expected_random[2], actual_random[2])
    replay = ReplayableBatchStream(corpus, 2, seed=17)
    replay.load_state_dict(resumed.restored_data_state["stream"])
    torch.testing.assert_close(next(replay), batch)
    actual = resumed.train_step(batch)
    torch.testing.assert_close(actual["loss"], expected["loss"], atol=0, rtol=0)
    for first, second in zip(original.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    resumed.load_checkpoint(checkpoint, lr_override=2e-4)
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)  # second warmup update


def test_replay_rejects_changed_data(tmp_path):
    first = corpus_at(tmp_path / "a")
    second = corpus_at(tmp_path / "b", offset=1)
    state = ReplayableBatchStream(first, 2, 17).state_dict()
    with pytest.raises(ValueError, match="fingerprint"):
        ReplayableBatchStream(second, 2, 17).load_state_dict(state)


def test_validation_preserves_training_rng_and_stream(tmp_path):
    corpus = corpus_at(tmp_path / "data")
    instance = trainer(tmp_path / "checkpoints")
    stream = ReplayableBatchStream(corpus, 2, 17)
    before_rng, before_stream = torch.get_rng_state().clone(), stream.state_dict()
    result = evaluate_corpus(instance, corpus, 2, max_tokens=32)
    assert result["valid_tokens"] == 32
    assert len(result["per_step_losses"]) == 3
    torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
    assert stream.state_dict() == before_stream and instance.model.training


def test_nonfinite_gradient_never_reaches_optimizer(tmp_path):
    instance = trainer(tmp_path)
    before = [parameter.detach().clone() for parameter in instance.model.parameters()]
    handle = instance.raw_model.embed.weight.register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(TrainingDivergence, match="gradients"):
        instance.train_step(torch.arange(18).reshape(2, 9) % 32)
    handle.remove()
    assert instance.step == 0 and not instance.optimizer.state
    for previous, current in zip(before, instance.model.parameters()):
        torch.testing.assert_close(previous, current, atol=0, rtol=0)


def test_nonfinite_loss_never_reaches_backward(tmp_path, monkeypatch):
    instance = trainer(tmp_path)
    import src.training.trainer as training
    def invalid(*args, **kwargs):
        return torch.tensor(float("nan"), requires_grad=True), {}
    monkeypatch.setattr(training, "compute_looplm_loss", invalid)
    with pytest.raises(TrainingDivergence, match="loss"):
        instance.train_step(torch.arange(18).reshape(2, 9) % 32)
    assert instance.step == 0 and not instance.optimizer.state
    assert all(parameter.grad is None for parameter in instance.model.parameters())


def test_trainer_loop_handles_accumulation_and_forbids_partial_checkpoint(tmp_path):
    instance = trainer(tmp_path, grad_accum_steps=2, max_steps=2, log_every=1, save_every=1)
    batch = torch.arange(18).reshape(2, 9) % 32
    instance.train_step(batch)
    with pytest.raises(ValueError, match="boundaries"):
        instance.save_checkpoint()
    instance.train([batch])
    assert instance.step == 2


def test_iterable_workers_do_not_duplicate_shards(tmp_path):
    for index in range(2):
        np.save(tmp_path / f"{index}.npy", np.arange(12, dtype=np.uint16) + 20 * index)
    dataset = PreTokenizedDataset(str(tmp_path), seq_len=3, shuffle_shards=False)
    batches = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    starts = [int(batch[0, 0]) for batch in batches]
    assert sorted(starts) == [0, 4, 8, 20, 24, 28]


def test_monitor_roundtrip_and_categories():
    monitor = InstabilityMonitor(warmup_steps=0)
    for step in range(100):
        monitor.train_events({"step": step + 1, "uniform_ce": 1.0, "exit_probs": [0.5, 0.5]})
    assert monitor.train_events({"step": 101, "uniform_ce": 2.0, "exit_probs": [0.5, 0.5]}) == ["loss_spike"]
    assert not monitor.validation_events(1.0)
    assert not monitor.validation_events(2.0)
    assert not monitor.validation_events(2.0)
    assert monitor.validation_events(2.0) == ["persistent_collapse"]
    restored = InstabilityMonitor(200)
    restored.load_state_dict(monitor.state_dict())
    assert restored.state_dict() == monitor.state_dict()


def test_preparation_reserves_documents_and_writes_integer_shards(tmp_path):
    class Tokenizer:
        eos_token_id = 0
        def __len__(self):
            return 32
        def encode(self, text, add_special_tokens=False):
            return [ord(character) % 31 + 1 for character in text]
    destination = tmp_path / "prepared"
    manifest = prepare_corpora({"validation": [{"text": "shared"}, {"text": "held out"}],
                                "test": [{"text": "only test"}],
                                "train": [{"text": "shared"}, {"text": "training"}]},
                               Tokenizer(), destination, {"validation": 14, "test": 8, "train": 8}, 10,
                               {"dataset": WIKITEXT_DATASET, "config": WIKITEXT_CONFIG}, {}, seq_len=8)
    assert manifest["splits"]["validation"]["source_records"] == 2
    assert manifest["splits"]["train"]["source_records"] == 1
    assert manifest["selection"]["unit"] == "text_row"
    assert manifest["status"] == "complete"
    assert np.load(destination / "train" / "tokens_00000.npy").dtype == np.uint16


def reproduction_arguments(tmp_path, monkeypatch):
    from scripts.train import parse_args
    corpus_at(tmp_path / "train")
    corpus_at(tmp_path / "validation", offset=1)
    monkeypatch.setattr(sys, "argv", ["train.py", "--reproduction", "--device", "cpu", "--no-amp",
                                      "--data-dir", str(tmp_path / "train"),
                                      "--val-data-dir", str(tmp_path / "validation"),
                                      "--batch-size", "2", "--seq-len", "8", "--max-steps", "3",
                                      "--val-tokens", "32", "--endpoint-val-tokens", "32",
                                      "--save-every", "1", "--keep-checkpoints", "1",
                                      "--log-every", "99", "--output-dir", str(tmp_path / "run")])
    return parse_args()


def test_runner_preserves_event_parent_after_pruning(tmp_path, monkeypatch):
    import src.training.reproduction as reproduction
    args = reproduction_arguments(tmp_path, monkeypatch)
    monkeypatch.setattr(reproduction.InstabilityMonitor, "train_events", lambda self, row: ["loss_spike"])
    assert reproduction.run_reproduction(args, tiny_config()) == 0
    parents = list((tmp_path / "run" / "events").glob("*.pt"))
    assert len(parents) == 3
    assert not (tmp_path / "run" / "checkpoints" / "step_0000001.pt").exists()
    assert any("step_0000001.pt" in path.name for path in parents)
    assert all("data_state" in torch.load(path, weights_only=False) for path in parents)


def test_runner_records_divergence_instead_of_false_completion(tmp_path, monkeypatch):
    import src.training.reproduction as reproduction
    args = reproduction_arguments(tmp_path, monkeypatch)
    def diverge(self, batch):
        raise TrainingDivergence("Injected nonfinite gradient event")
    monkeypatch.setattr(reproduction.Trainer, "train_step", diverge)
    assert reproduction.run_reproduction(args, tiny_config()) == 42
    result = json.loads((tmp_path / "run" / "result.json").read_text())
    assert result["status"] == "numerical_divergence" and result["completed_updates"] == 0
    failure = torch.load(tmp_path / "run" / "failure_context.pt", weights_only=False)
    assert failure["batch"].shape == (2, 9)
    assert Path(failure["last_healthy_checkpoint"]).exists()


def test_cli_smoke_resume_and_provenance(tmp_path):
    corpus_at(tmp_path / "train")
    corpus_at(tmp_path / "validation", offset=1)
    output = tmp_path / "run"
    command = [sys.executable, "scripts/train.py", "--reproduction", "--device", "cpu", "--no-amp",
               "--hidden-size", "16", "--num-layers", "1", "--num-heads", "2", "--intermediate-size", "32",
               "--num-recurrent-steps", "2", "--seq-len", "8", "--batch-size", "2", "--grad-accum-steps", "2",
               "--data-dir", str(tmp_path / "train"), "--val-data-dir", str(tmp_path / "validation"),
               "--val-every", "1", "--val-tokens", "32", "--save-every", "1", "--log-every", "1",
               "--output-dir", str(output), "--max-steps", "2"]
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["objective"] == "joint" and not metadata["model"]["tie_word_embeddings"]
    assert len(metadata["git"]["source_tree_sha256"]) == 64
    assert json.loads((output / "result.json").read_text())["tokens_seen"] == 64
    command[-1] = "3"
    command.extend(["--resume-from", str(output / "checkpoints" / "step_0000002.pt")])
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((output / "result.json").read_text())["tokens_seen"] == 96
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows if row["kind"] == "train"] == [1, 2, 3]
