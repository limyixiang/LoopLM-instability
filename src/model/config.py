from dataclasses import dataclass


@dataclass
class LoopLMConfig:
    vocab_size: int = 49152
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    # SwiGLU intermediate size; paper uses ~2.67x hidden (rounded to multiple of 64)
    intermediate_size: int = 2048
    max_seq_len: int = 2048
    max_recurrent_steps: int = 4
    rope_base: float = 10000.0
    dropout: float = 0.0
    beta_kl: float = 0.1
    # Defaults retain the architecture used by existing checkpoints.
    tie_word_embeddings: bool = True
    normalize_recurrent_state: bool = False
    fp32_gate: bool = False

    def num_parameters(self) -> int:
        """Estimate total parameter count."""
        embed = self.vocab_size * self.hidden_size
        # Each transformer layer:
        #   Attention: Q, K, V projections + output projection
        attn = 4 * self.hidden_size * self.hidden_size
        #   SwiGLU FFN: gate + up projections + down projection
        ffn = 3 * self.hidden_size * self.intermediate_size
        #   4 RMSNorm (sandwich: pre-attn, post-attn, pre-ffn, post-ffn) — each hidden_size params
        norms = 4 * self.hidden_size
        layer = attn + ffn + norms
        # Readout norm, optional untied LM head, and exit gate.
        # Exit gate: hidden_size -> 1
        gate = self.hidden_size + 1
        head = 0 if self.tie_word_embeddings else embed
        return embed + head + self.num_layers * layer + self.hidden_size + gate

    @classmethod
    def small_repro(cls) -> "LoopLMConfig":
        """~118M pilot matching released Ouro's loop-boundary normalization."""
        return cls(tie_word_embeddings=False, normalize_recurrent_state=True,
                   fp32_gate=True)

    @classmethod
    def small(cls) -> "LoopLMConfig":
        """~100M parameter config for prototyping."""
        return cls(
            vocab_size=49152,
            hidden_size=768,
            num_layers=6,
            num_heads=12,
            intermediate_size=2048,
            max_seq_len=2048,
            max_recurrent_steps=4,
            rope_base=10000.0,
            dropout=0.0,
            beta_kl=0.1,
        )

    @classmethod
    def ouro_1_4b_repro(cls) -> "LoopLMConfig":
        """Released 1.4B tensor shapes, with Stage-1 context/RoPE settings.

        Random initialization, not a loader for the released trained weights.
        Keep the legacy ouro_1_4b preset unchanged for existing checkpoints.
        """
        return cls(hidden_size=2048, num_layers=24, num_heads=16,
                   intermediate_size=5632, max_seq_len=4096,
                   max_recurrent_steps=8, rope_base=10000.0,
                   tie_word_embeddings=False, normalize_recurrent_state=True,
                   fp32_gate=True)

    @classmethod
    def depth24_repro(cls) -> "LoopLMConfig":
        """~245M fallback: 24 shared layers with the validated pilot width."""
        config = cls.small_repro()
        config.num_layers = 24
        config.max_seq_len = 4096
        config.max_recurrent_steps = 8
        return config

    @classmethod
    def ouro_1_4b(cls) -> "LoopLMConfig":
        """1.4B parameter config (Table 2)."""
        return cls(
            vocab_size=49152,
            hidden_size=2048,
            num_layers=24,
            num_heads=16,
            intermediate_size=5504,
            max_seq_len=4096,
            max_recurrent_steps=4,
            rope_base=10000.0,
            dropout=0.0,
            beta_kl=0.1,
        )

    @classmethod
    def ouro_2_6b(cls) -> "LoopLMConfig":
        """2.6B parameter config — 48 layers, same hidden dim as 1.4B (Table 2)."""
        return cls(
            vocab_size=49152,
            hidden_size=2048,
            num_layers=48,
            num_heads=16,
            intermediate_size=5504,
            max_seq_len=4096,
            max_recurrent_steps=4,
            rope_base=10000.0,
            dropout=0.0,
            beta_kl=0.1,
        )
