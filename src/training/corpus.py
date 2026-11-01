"""Bounded, reproducible corpus selection and tokenization for the pilot."""

from array import array
import hashlib
import importlib.metadata
import json
from itertools import islice
from pathlib import Path
import platform
import re

import numpy as np

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
OURO_TOKENIZER = "ByteDance/Ouro-1.4B"
OURO_TOKENIZER_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
SMOLLM_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
SMOLLM_TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_revision(repository, revision, repo_type):
    if revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    from huggingface_hub import HfApi
    resolved = HfApi().repo_info(repository, revision=revision, repo_type=repo_type).sha
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Source must resolve to an immutable commit")
    return resolved


def load_tokenizer(repository=OURO_TOKENIZER, revision=OURO_TOKENIZER_REVISION,
                   cache_dir=None, offline=False):
    """Load standard assets only and fingerprint actual encoding behavior."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    resolved = resolve_revision(repository, revision, "model")
    tokenizer = AutoTokenizer.from_pretrained(repository, revision=resolved, use_fast=True,
                                              trust_remote_code=False, cache_dir=cache_dir,
                                              local_files_only=offline)
    files = {}
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = hf_hub_download(repository, filename, revision=resolved, cache_dir=cache_dir,
                               local_files_only=offline)
        files[filename] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS separator")
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    settings = {"special_tokens_map": tokenizer.special_tokens_map,
                "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "add_bos_token": getattr(tokenizer, "add_bos_token", None),
                "add_eos_token": getattr(tokenizer, "add_eos_token", None),
                "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side,
                "clean_up_tokenization_spaces": tokenizer.clean_up_tokenization_spaces}
    info = {"repository": repository, "revision": resolved, "vocab_size": len(tokenizer),
            "vocab_sha256": canonical_hash(tokenizer.get_vocab()), "files_sha256": files,
            "backend_sha256": canonical_hash(backend), "settings": settings,
            "fingerprint": canonical_hash({"backend": backend, "settings": settings}),
            "encoding": "add_special_tokens=False; append one EOS per complete source record"}
    return tokenizer, info


def split_for_hash(text_sha256, seed, validation_fraction=0.01, test_fraction=0.01):
    if seed < 0 or min(validation_fraction, test_fraction) <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Use a nonnegative seed and positive held-out fractions totaling less than one")
    digest = hashlib.sha256(f"{seed}:split:{text_sha256}".encode()).digest()
    value = int.from_bytes(digest[:8], "big")
    if value < int(validation_fraction * (1 << 64)):
        return "validation"
    if value < int((validation_fraction + test_fraction) * (1 << 64)):
        return "test"
    return "train"


class SplitWriter:
    """Whole records and EOS separators; the final record may exceed the target."""

    def __init__(self, destination, tokenizer, target_tokens, shard_tokens, seq_len=512, unit="document"):
        if min(target_tokens, shard_tokens, seq_len) < 1 or tokenizer.eos_token_id is None:
            raise ValueError("Positive sizes and an explicit EOS separator are required")
        self.destination, self.tokenizer, self.target = destination, tokenizer, target_tokens
        self.shard_tokens, self.seq_len, self.unit = shard_tokens, seq_len, unit
        self.dtype = np.uint16 if len(tokenizer) <= 65536 else np.uint32
        self.buffer = array("H" if self.dtype == np.uint16 else "I")
        self.total, self.records, self.shards = 0, 0, []
        self.record_digest = hashlib.sha256()
        destination.mkdir(parents=True, exist_ok=True)
        self.record_file = (destination / "records.jsonl").open("wb")

    @property
    def full(self):
        return self.total >= self.target

    def _flush(self):
        if not self.buffer:
            return
        path = self.destination / f"tokens_{len(self.shards):05d}.npy"
        np.save(path, np.asarray(self.buffer, dtype=self.dtype))
        count = len(self.buffer)
        self.shards.append({"name": path.name, "tokens": count,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "packed_sequences": count // (self.seq_len + 1),
                            "unused_tokens": count % (self.seq_len + 1)})
        self.buffer = array(self.buffer.typecode)

    def add(self, text, text_hash, record_id, source_split, source_index):
        if self.full:
            return
        ids = self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        if min(ids) < 0 or max(ids) >= len(self.tokenizer):
            raise ValueError("Tokenizer emitted IDs outside its vocabulary")
        record = {"id": record_id, "text_sha256": text_hash, "source_split": source_split,
                  "source_index": source_index, "token_start": self.total,
                  "tokens": len(ids), "eos_appended": True}
        line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
        self.record_file.write(line)
        self.record_digest.update(line)
        offset = 0
        while offset < len(ids):
            count = min(len(ids) - offset, self.shard_tokens - len(self.buffer))
            self.buffer.extend(ids[offset:offset + count])
            offset += count
            if len(self.buffer) == self.shard_tokens:
                self._flush()
        self.total += len(ids)
        self.records += 1

    def close(self):
        self._flush()
        self.record_file.close()
        sequences = sum(shard["packed_sequences"] for shard in self.shards)
        return {"unit": self.unit, "source_records": self.records, "tokens": self.total,
                "target_tokens": self.target, "target_reached": self.full,
                "records_file": f"{self.destination.name}/records.jsonl",
                "records_sha256": self.record_digest.hexdigest(), "shards": self.shards,
                "packed_sequences": sequences, "target_positions": sequences * self.seq_len,
                "unused_tokens": sum(shard["unused_tokens"] for shard in self.shards),
                "dtype": str(np.dtype(self.dtype))}


def prepare_corpora(sources, tokenizer, destination, targets, shard_tokens, source,
                    tokenizer_info, seed=42, validation_fraction=0.01, test_fraction=0.01,
                    max_source_records=2_000_000, seq_len=512):
    """Assign and deduplicate before tokenization; consume only declared sources."""
    if source["dataset"] == FINEWEB_DATASET:
        mode, unit = "content_hash_split", "document"
        split_for_hash("0" * 64, seed, validation_fraction, test_fraction)
        streams = [("train", sources["train"])]
    elif source["dataset"] in (WIKITEXT_DATASET, "wikitext"):
        if source["config"] != WIKITEXT_CONFIG:
            raise ValueError("The WikiText smoke test must use wikitext-103-raw-v1")
        mode, unit = "official_splits_exact_row_deduplication", "text_row"
        streams = [(split, sources[split]) for split in ("validation", "test", "train")]
    else:
        raise ValueError("Unsupported reproduction source")
    if max_source_records < 1:
        raise ValueError("max_source_records must be positive")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output must be empty; preparation never mixes or overwrites corpora")
    destination.mkdir(parents=True, exist_ok=True)
    writers = {split: SplitWriter(destination / split, tokenizer, targets[split], shard_tokens,
                                  seq_len=seq_len, unit=unit) for split in ("train", "validation", "test")}
    seen_text, seen_ids = set(), set()
    scanned, duplicates, empty, quota_skips = 0, 0, 0, 0
    stop_reason = "source_exhausted"
    try:
        for source_split, documents in streams:
            for source_index, document in enumerate(islice(documents, max_source_records - scanned)):
                scanned += 1
                text = document.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    empty += 1
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                record_id = str(document["id"]) if document.get("id") is not None else "sha256:" + digest
                if digest in seen_text or record_id in seen_ids:
                    duplicates += 1
                    continue
                seen_text.add(digest)
                seen_ids.add(record_id)
                split = split_for_hash(digest, seed, validation_fraction, test_fraction) if mode == "content_hash_split" else source_split
                if writers[split].full:
                    quota_skips += 1
                    continue
                writers[split].add(text, digest, record_id, source_split, source_index)
                if all(writer.full for writer in writers.values()):
                    stop_reason = "targets_reached"
                    break
                if mode != "content_hash_split" and writers[split].full:
                    break
            if stop_reason == "targets_reached" or scanned >= max_source_records:
                if stop_reason != "targets_reached":
                    stop_reason = "max_source_records"
                break
    finally:
        splits = {split: writer.close() for split, writer in writers.items()}
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("datasets", "transformers", "tokenizers", "huggingface_hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {"version": 2, "status": "complete" if all(writer.full for writer in writers.values()) else "incomplete",
                "source": source, "tokenizer": tokenizer_info, "software": versions,
                "selection": {"method": mode, "unit": unit, "seed": seed,
                              "validation_fraction": validation_fraction if mode == "content_hash_split" else None,
                              "test_fraction": test_fraction if mode == "content_hash_split" else None,
                              "order": "pinned source order; first eligible complete records to each target",
                              "hash": "SHA256(seed:split:SHA256(UTF8 text)), first 64 bits big-endian",
                              "max_source_records": max_source_records, "scanned_records": scanned,
                              "exact_duplicates_skipped": duplicates, "empty_records_skipped": empty,
                              "full_split_records_skipped": quota_skips, "stop_reason": stop_reason},
                "packing": {"sequence_length": seq_len, "shard_tokens": shard_tokens,
                            "separator": "one EOS after every complete source record; no automatic BOS",
                            "boundary_policy": "pack each shard independently; discard incomplete S+1 tails",
                            "quota_policy": "whole records; token target may be exceeded by the final record"},
                "splits": splits}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
