#!/usr/bin/env python
"""Compare pinned Ouro/SmolLM2 vocabularies, encoding settings, and sample IDs."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.corpus import (load_tokenizer, OURO_TOKENIZER, OURO_TOKENIZER_REVISION,
                                 SMOLLM_TOKENIZER, SMOLLM_TOKENIZER_REVISION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=".uv/tokenizer-check")
    parser.add_argument("--output", default=".uv/tokenizer-check/verification.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    ouro, ouro_info = load_tokenizer(OURO_TOKENIZER, OURO_TOKENIZER_REVISION, args.cache_dir, args.offline)
    smol, smol_info = load_tokenizer(SMOLLM_TOKENIZER, SMOLLM_TOKENIZER_REVISION, args.cache_dir, args.offline)
    first, second = ouro.get_vocab(), smol.get_vocab()
    differences = [{"token": token, "ouro_id": first.get(token), "smollm2_id": second.get(token)}
                   for token in sorted(set(first) | set(second)) if first.get(token) != second.get(token)]
    probes = ["A short English sentence.", "中文 café naïve 😀", "def f(x):\n    return x + 1\n",
              "<think>42</think>", "<repo_name>org/project<reponame>", "<|endoftext|>"]
    results = [{"text": text, "ouro": ouro.encode(text, add_special_tokens=False),
                "smollm2": smol.encode(text, add_special_tokens=False)} for text in probes]
    report = {"ouro": ouro_info, "smollm2": smol_info,
              "equivalent": ouro_info["fingerprint"] == smol_info["fingerprint"],
              "vocabulary_differences": differences, "encoding_probes": results,
              "selected_default": {"repository": OURO_TOKENIZER, "revision": OURO_TOKENIZER_REVISION},
              "scope": "Tokenizer asset and encoding verification only; no corpus download or GPU training"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"equivalent": report["equivalent"], "vocabulary_differences": differences,
                      "selected_default": report["selected_default"], "report": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
