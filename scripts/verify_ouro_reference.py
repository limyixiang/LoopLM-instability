#!/usr/bin/env python
"""Compare tiny FP32 forwards and joint-loss gradients with pinned released Ouro.

Downloads code/configuration only, never model weights. The pinned Python model
code is loaded through Transformers' explicit trust_remote_code mechanism.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.training.objectives import compute_looplm_loss
from src.training.reproduction import OURO_REFERENCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=".uv/ouro-reference")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    cache = Path(args.cache_dir).resolve()
    os.environ["HF_MODULES_CACHE"] = str(cache / "modules")
    from huggingface_hub import hf_hub_download
    import transformers
    if transformers.__version__ != "4.55.0":
        raise RuntimeError("The pinned released code requires Transformers 4.55.0. "
                           "Use the separate reference-check virtual environment in REPRODUCTION.md; "
                           "install with python -m pip install 'transformers==4.55.0'.")
    from transformers import AutoConfig, AutoModelForCausalLM
    repository, revision = OURO_REFERENCE.split("@")
    source = cache / "tiny_reference"
    source.mkdir(parents=True, exist_ok=True)
    for name in ("configuration_ouro.py", "modeling_ouro.py", "config.json"):
        path = hf_hub_download(repository, name, revision=revision,
                               cache_dir=str(cache / "downloads"), local_files_only=args.offline)
        (source / name).write_bytes(Path(path).read_bytes())
    config_values = json.loads((source / "config.json").read_text())
    config_values.update(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         head_dim=8, max_position_embeddings=16, total_ut_steps=4,
                         rope_theta=10000.0, layer_types=["full_attention"] * 2,
                         use_cache=False, tie_word_embeddings=False)
    (source / "config.json").write_text(json.dumps(config_values), encoding="utf-8")
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    reference = AutoModelForCausalLM.from_config(config, trust_remote_code=True).float().eval()
    local_config = LoopLMConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                               num_layers=2, num_heads=4, max_seq_len=16,
                               max_recurrent_steps=4, tie_word_embeddings=False,
                               normalize_recurrent_state=True, fp32_gate=True)
    torch.manual_seed(101)
    local = LoopLM(local_config).eval()
    mapping = {"embed.weight": "model.embed_tokens.weight", "lm_head.weight": "lm_head.weight",
               "final_norm.weight": "model.norm.weight", "exit_gate.weight": "model.early_exit_gate.weight",
               "exit_gate.bias": "model.early_exit_gate.bias"}
    submodules = {"pre_attn_norm": "input_layernorm", "post_attn_norm": "input_layernorm_2",
                  "pre_ffn_norm": "post_attention_layernorm", "post_ffn_norm": "post_attention_layernorm_2",
                  "attn.q_proj": "self_attn.q_proj", "attn.k_proj": "self_attn.k_proj",
                  "attn.v_proj": "self_attn.v_proj", "attn.o_proj": "self_attn.o_proj",
                  "ffn.w_gate": "mlp.gate_proj", "ffn.w_up": "mlp.up_proj", "ffn.w_down": "mlp.down_proj"}
    for layer in range(local_config.num_layers):
        for first, second in submodules.items():
            mapping[f"layers.{layer}.{first}.weight"] = f"model.layers.{layer}.{second}.weight"
    local_state, reference_state = local.state_dict(), reference.state_dict()
    if set(mapping) != set(local_state) or set(mapping.values()) != set(reference_state):
        raise AssertionError("Reference parameter mapping is incomplete; inspect the pinned implementation")
    with torch.no_grad():
        for name, reference_name in mapping.items():
            reference_state[reference_name].copy_(local_state[name])
    batch = torch.arange(18).reshape(2, 9) % 64
    result = local(batch[:, :-1])
    _, states, gate_logits = reference.model(batch[:, :-1], use_cache=False)
    logits = [reference.lm_head(state) for state in states]
    lambdas = [value.squeeze(-1).sigmoid() for value in gate_logits]
    maximum_forward_error = 0.0
    for first, second in zip(result.logits + result.exit_lambdas, logits + lambdas):
        torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-4)
        maximum_forward_error = max(maximum_forward_error, (first - second).abs().max().item())
    local_loss, _ = compute_looplm_loss(result.logits, result.exit_lambdas, batch[:, 1:])
    reference_loss, _ = compute_looplm_loss(logits, lambdas, batch[:, 1:])
    local_loss.backward()
    reference_loss.backward()
    maximum_gradient_error = 0.0
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in local.named_parameters():
        expected = reference_parameters[mapping[name]].grad
        torch.testing.assert_close(parameter.grad, expected, atol=2e-5, rtol=2e-3)
        maximum_gradient_error = max(maximum_gradient_error, (parameter.grad - expected).abs().max().item())
    report = {"reference": OURO_REFERENCE, "precision": "FP32", "loops": 4,
              "max_forward_absolute_error": maximum_forward_error,
              "max_gradient_absolute_error": maximum_gradient_error,
              "scope": "Tiny copied-weight forward/gradient agreement; no H100 training or instability result"}
    (cache / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
