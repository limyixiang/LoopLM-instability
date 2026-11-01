"""Offline checks; synthetic records only, with no dataset or tokenizer downloads."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.training.corpus import (prepare_corpora, split_for_hash, canonical_hash,
    FINEWEB_DATASET, FINEWEB_CONFIG, FINEWEB_REVISION, OURO_TOKENIZER, OURO_TOKENIZER_REVISION)
from src.training.replay import TokenCorpus, ReplayableBatchStream


class TinyTokenizer:
    eos_token_id = 0
    def __len__(self):
        return 32
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) % 31 + 1 for character in text]


def records():
    values = [{"id": f"document-{index}", "text": f"synthetic document {index:04d}"} for index in range(500)]
    return [values[0], {"id": "duplicate-text", "text": values[0]["text"]},
            {"id": values[0]["id"], "text": "different content with repeated ID"}] + values[1:]


def prepare(destination, **overrides):
    arguments = dict(sources={"train": records()}, tokenizer=TinyTokenizer(), destination=destination,
                     targets={"train": 160, "validation": 80, "test": 80}, shard_tokens=54,
                     source={"dataset": FINEWEB_DATASET, "config": FINEWEB_CONFIG, "revision": FINEWEB_REVISION},
                     tokenizer_info={"repository": OURO_TOKENIZER, "revision": OURO_TOKENIZER_REVISION,
                                     "fingerprint": "synthetic-tokenizer", "vocab_size": 32},
                     seed=42, validation_fraction=0.2, test_fraction=0.2,
                     max_source_records=500, seq_len=8)
    return prepare_corpora(**{**arguments, **overrides})


def selected(destination, split):
    return [json.loads(line) for line in (destination / split / "records.jsonl").read_text().splitlines()]


def test_same_settings_reproduce_selected_records_and_token_files(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    a, b = prepare(first), prepare(second)
    assert a == b and a["status"] == "complete"
    for split in a["splits"]:
        assert selected(first, split) == selected(second, split)
        assert TokenCorpus(first / split, 8).fingerprint == TokenCorpus(second / split, 8).fingerprint
    assert a["source"]["revision"] == FINEWEB_REVISION


def test_disjoint_ids_and_text_hashes_and_correct_assignment(tmp_path):
    manifest = prepare(tmp_path / "data")
    ids, hashes = set(), set()
    for split in manifest["splits"]:
        for record in selected(tmp_path / "data", split):
            assert record["id"] not in ids and record["text_sha256"] not in hashes
            assert split_for_hash(record["text_sha256"], 42, 0.2, 0.2) == split
            ids.add(record["id"])
            hashes.add(record["text_sha256"])
    assert manifest["selection"]["exact_duplicates_skipped"] == 2


def test_split_is_order_independent_and_seed_is_material():
    hashes = [hashlib.sha256(str(index).encode()).hexdigest() for index in range(100)]
    forward = {digest: split_for_hash(digest, 42) for digest in hashes}
    reverse = {digest: split_for_hash(digest, 42) for digest in reversed(hashes)}
    assert forward == reverse
    assert any(split_for_hash(digest, 43) != value for digest, value in forward.items())


def test_whole_records_keep_eos_and_actual_counts(tmp_path):
    destination = tmp_path / "data"
    manifest = prepare(destination)
    texts = {hashlib.sha256(record["text"].encode()).hexdigest(): record["text"] for record in records()}
    for split, stats in manifest["splits"].items():
        expected = []
        for record in selected(destination, split):
            expected.extend(TinyTokenizer().encode(texts[record["text_sha256"]]) + [0])
            assert record["eos_appended"]
        actual = np.concatenate([np.load(path) for path in sorted((destination / split).glob("*.npy"))]).tolist()
        assert actual == expected and actual[-1] == 0
        assert stats["tokens"] == len(actual) >= stats["target_tokens"]
        corpus = TokenCorpus(destination / split, 8)
        assert len(corpus) * 8 == stats["target_positions"]
        assert corpus.preparation["tokenizer"]["revision"] == OURO_TOKENIZER_REVISION


def test_bounded_read_records_incomplete_manifest(tmp_path):
    consumed = []
    def source():
        for document in records():
            consumed.append(document)
            yield document
    destination = tmp_path / "limited"
    manifest = prepare(destination, sources={"train": source()}, max_source_records=7,
                       targets={"train": 10000, "validation": 10000, "test": 10000})
    assert len(consumed) == 7
    assert manifest["status"] == "incomplete"
    assert manifest["selection"]["stop_reason"] == "max_source_records"
    assert json.loads((destination / "manifest.json").read_text())["status"] == "incomplete"
    split = next(name for name, stats in manifest["splits"].items() if stats["packed_sequences"])
    with pytest.raises(ValueError, match="incomplete"):
        TokenCorpus(destination / split, 8)


def test_changed_selected_record_manifest_is_detected(tmp_path):
    destination = tmp_path / "data"
    prepare(destination)
    with (destination / "train" / "records.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="provenance"):
        TokenCorpus(destination / "train", 8)


def test_prepared_data_preserves_paired_order_and_replay(tmp_path):
    destination = tmp_path / "data"
    prepare(destination)
    corpus = TokenCorpus(destination / "train", 8)
    first = ReplayableBatchStream(corpus, 2, 101)
    paired = ReplayableBatchStream(corpus, 2, 101)
    torch.testing.assert_close(next(first), next(paired), atol=0, rtol=0)
    state = first.state_dict()
    resumed = ReplayableBatchStream(corpus, 2, 101)
    resumed.load_state_dict(state)
    torch.testing.assert_close(next(first), next(resumed), atol=0, rtol=0)


def test_prepare_cli_defaults_and_smoke_mode(monkeypatch, tmp_path):
    from scripts.prepare_reproduction_data import parse_args
    import sys
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--output-dir", str(tmp_path)])
    args = parse_args()
    assert args.dataset == FINEWEB_DATASET and args.dataset_config == FINEWEB_CONFIG
    assert args.revision == FINEWEB_REVISION and args.tokenizer_revision == OURO_TOKENIZER_REVISION
    assert args.train_tokens == 250000000
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--output-dir", str(tmp_path), "--dataset", "wikitext"])
    args = parse_args()
    assert args.dataset_config == "wikitext-103-raw-v1" and args.train_tokens == 1000000


def test_prepared_provenance_reaches_run_and_resume(tmp_path, monkeypatch):
    from tests.test_reproduction import reproduction_arguments, tiny_config
    from src.training.reproduction import run_reproduction
    destination = tmp_path / "prepared"
    manifest = prepare(destination)
    args = reproduction_arguments(tmp_path, monkeypatch)
    args.data_dir, args.val_data_dir = str(destination / "train"), str(destination / "validation")
    args.val_tokens, args.endpoint_val_tokens, args.max_steps = 16, 16, 1
    assert run_reproduction(args, tiny_config()) == 0
    metadata = json.loads((tmp_path / "run" / "run.json").read_text())
    assert metadata["data_preparation"]["train"]["source"] == manifest["source"]
    assert metadata["data_preparation"]["train"]["tokenizer"]["fingerprint"] == "synthetic-tokenizer"
    args.resume_from = str(tmp_path / "run" / "checkpoints" / "step_0000001.pt")
    args.max_steps = 2
    assert run_reproduction(args, tiny_config()) == 0
    data = json.loads((destination / "manifest.json").read_text())
    data["selection"]["seed"] += 1
    (destination / "manifest.json").write_text(json.dumps(data))
    args.resume_from = str(tmp_path / "run" / "checkpoints" / "step_0000002.pt")
    args.max_steps = 3
    with pytest.raises(ValueError, match="provenance"):
        run_reproduction(args, tiny_config())
