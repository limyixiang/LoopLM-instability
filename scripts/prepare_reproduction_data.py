#!/usr/bin/env python
"""Prepare a bounded, pinned FineWeb-Edu corpus; raw WikiText is a smoke test."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.corpus import (
    FINEWEB_DATASET, FINEWEB_CONFIG, FINEWEB_REVISION,
    WIKITEXT_DATASET, WIKITEXT_CONFIG, WIKITEXT_REVISION,
    OURO_TOKENIZER, OURO_TOKENIZER_REVISION,
    load_tokenizer, resolve_revision, prepare_corpora,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default=FINEWEB_DATASET, choices=[FINEWEB_DATASET, WIKITEXT_DATASET, "wikitext"])
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer-id", default=OURO_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--train-tokens", type=int, default=None)
    parser.add_argument("--validation-tokens", type=int, default=None)
    parser.add_argument("--test-tokens", type=int, default=None)
    parser.add_argument("--shard-tokens", type=int, default=10_000_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--max-source-records", type=int, default=2_000_000)
    args = parser.parse_args()
    fineweb = args.dataset == FINEWEB_DATASET
    args.dataset = FINEWEB_DATASET if fineweb else WIKITEXT_DATASET
    args.dataset_config = args.dataset_config or (FINEWEB_CONFIG if fineweb else WIKITEXT_CONFIG)
    args.revision = args.revision or (FINEWEB_REVISION if fineweb else WIKITEXT_REVISION)
    if args.tokenizer_revision is None and args.tokenizer_id == OURO_TOKENIZER:
        args.tokenizer_revision = OURO_TOKENIZER_REVISION
    for name, default in (("train_tokens", 250_000_000 if fineweb else 1_000_000),
                          ("validation_tokens", 2_000_000 if fineweb else 131_072),
                          ("test_tokens", 2_000_000 if fineweb else 131_072)):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if min(args.train_tokens, args.validation_tokens, args.test_tokens, args.shard_tokens,
           args.seq_len, args.max_source_records) < 1 or args.selection_seed < 0:
        parser.error("Use positive sizes and a nonnegative selection seed")
    if not fineweb and args.dataset_config != WIKITEXT_CONFIG:
        parser.error("WikiText is an optional raw-v1 row-level pipeline smoke test")
    return args


def main():
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; do not mix previously prepared corpora")
    from datasets import load_dataset
    revision = resolve_revision(args.dataset, args.revision, "dataset")
    tokenizer, tokenizer_info = load_tokenizer(args.tokenizer_id, args.tokenizer_revision)
    source_splits = ("train",) if args.dataset == FINEWEB_DATASET else ("train", "validation", "test")
    sources = {split: load_dataset(args.dataset, args.dataset_config, split=split,
                                  revision=revision, streaming=True) for split in source_splits}
    manifest = prepare_corpora(sources, tokenizer, output,
                              {"train": args.train_tokens, "validation": args.validation_tokens, "test": args.test_tokens},
                              args.shard_tokens, {"dataset": args.dataset, "config": args.dataset_config,
                                                  "revision": revision}, tokenizer_info,
                              seed=args.selection_seed, validation_fraction=args.validation_fraction,
                              test_fraction=args.test_fraction, max_source_records=args.max_source_records,
                              seq_len=args.seq_len)
    for split, stats in manifest["splits"].items():
        print(f"{split}: {stats['tokens']:,} tokens from {stats['source_records']:,} {stats['unit']} records")
    if manifest["status"] != "complete":
        raise SystemExit("Source limit reached before all token targets; inspect manifest.json and prepare a new directory")


if __name__ == "__main__":
    main()
