"""Deterministic, indexed token batches for the single-GPU reproduction pilot."""

import bisect
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


class TokenCorpus:
    """Memory-mapped .npy files: 1-D token streams or packed (N, S+1) arrays.

    Each file is packed independently; unused tails are recorded in the manifest.
    File hashes make replay reject changed data even when paths stay identical.
    """

    def __init__(self, directory: str | Path, seq_len: int):
        self.seq_len = seq_len
        self.paths = sorted(Path(directory).glob("*.npy"))
        if not self.paths:
            raise ValueError(f"No .npy token files in {directory}")
        self.arrays, self.ends, self.manifest = [], [], []
        total = 0
        for path in self.paths:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype.kind not in "iu":
                raise ValueError(f"Token IDs must be integers: {path}")
            if array.ndim == 1:
                count = len(array) // (seq_len + 1)
                unused = len(array) % (seq_len + 1)
            elif array.ndim == 2 and array.shape[1] == seq_len + 1:
                count, unused = len(array), 0
            else:
                raise ValueError(f"Expected a token stream or (N, {seq_len + 1}) array: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            self.manifest.append({"name": path.name, "sha256": digest.hexdigest(),
                                  "shape": list(array.shape), "dtype": str(array.dtype),
                                  "chunks": count, "unused_tokens": unused})
            self.arrays.append(array)
            total += count
            self.ends.append(total)
        if total == 0:
            raise ValueError("Corpus has no complete sequences")
        # Names and location may change when staging identical data on a cluster.
        identity = repr((seq_len, [(m["sha256"], m["shape"]) for m in self.manifest]))
        self.fingerprint = hashlib.sha256(identity.encode()).hexdigest()
        self.preparation = None
        manifest_path = Path(directory).parent / "manifest.json"
        if manifest_path.is_file():
            raw = manifest_path.read_bytes()
            manifest = json.loads(raw)
            if manifest.get("version") == 2:
                if manifest.get("status") != "complete":
                    raise ValueError("Prepared corpus has incomplete token targets")
                if manifest["packing"]["sequence_length"] != seq_len:
                    raise ValueError("Prepared corpus sequence length differs from the requested packing")
                split = Path(directory).name
                expected = manifest["splits"][split]
                if [(s["name"], s["sha256"]) for s in expected["shards"]] != [(s["name"], s["sha256"]) for s in self.manifest]:
                    raise ValueError("Prepared shard files differ from manifest.json")
                records = Path(directory) / "records.jsonl"
                digest = hashlib.sha256()
                with records.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected["records_sha256"]:
                    raise ValueError("Selected-record provenance differs from manifest.json")
                self.preparation = {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
                                    "source": manifest["source"], "selection": manifest["selection"],
                                    "tokenizer": manifest["tokenizer"], "packing": manifest["packing"],
                                    "split": split, "records_sha256": expected["records_sha256"],
                                    "prepared_tokens": expected["tokens"],
                                    "target_positions": expected["target_positions"]}

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[shard - 1] if shard else 0)
        array = self.arrays[shard]
        row = array[local] if array.ndim == 2 else array[local * (self.seq_len + 1):(local + 1) * (self.seq_len + 1)]
        return torch.from_numpy(np.array(row, dtype=np.int64, copy=True))


class ReplayableBatchStream:
    """Batch order is independent of model RNG, loop count, and validation."""

    def __init__(self, corpus: TokenCorpus, batch_size: int, seed: int, shuffle: bool = True):
        if batch_size < 1 or len(corpus) < batch_size:
            raise ValueError("Corpus must contain at least one complete positive-size batch")
        self.corpus, self.batch_size, self.seed, self.shuffle = corpus, batch_size, seed, shuffle
        self.epoch, self.offset = 0, 0
        self.last_indices: list[int] = []
        self._set_order()

    def _set_order(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        self.order = rng.permutation(len(self.corpus)) if self.shuffle else np.arange(len(self.corpus))

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        if self.offset + self.batch_size > len(self.corpus):
            self.epoch += 1
            self.offset = 0
            self._set_order()
        self.last_indices = self.order[self.offset:self.offset + self.batch_size].tolist()
        batch = torch.stack([self.corpus[index] for index in self.last_indices])
        self.offset += self.batch_size
        return batch

    def state_dict(self) -> dict:
        return {"version": 1, "fingerprint": self.corpus.fingerprint,
                "seed": self.seed, "shuffle": self.shuffle, "batch_size": self.batch_size,
                "epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict) -> None:
        current = self.state_dict()
        for key in ("version", "fingerprint", "seed", "shuffle", "batch_size"):
            if state.get(key) != current[key]:
                raise ValueError(f"Replay stream mismatch: {key}")
        if state["epoch"] < 0 or not 0 <= state["offset"] <= len(self.corpus):
            raise ValueError("Invalid replay position")
        if state["offset"] % self.batch_size:
            raise ValueError("Replay position must be at a batch boundary")
        self.epoch, self.offset = state["epoch"], state["offset"]
        self._set_order()
