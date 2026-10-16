"""Shape of the looped transformer."""

from __future__ import annotations

from dataclasses import dataclass

from .layers import ROPE_THETA, validate_rope_theta

VOCAB_SIZE = 32001  # Llama-2 tokenizer plus <|assistant_end|>
PADDED_VOCAB_SIZE = 32064
NORM_EPS = 1e-6


@dataclass
class ModelConfig:
    """Prelude, recurrent block and coda layers; T forward loops of which the last B are backpropagated.

    The defaults are the reference cell of Schwethelm et al. (2026): width 640, five 128-wide heads, a 2-4-2
    layer split and four fully trained loops.
    """

    vocab_size: int = VOCAB_SIZE
    padded_vocab_size: int | None = None
    hidden_size: int = 640
    intermediate_size: int = 2560
    num_attention_heads: int = 5
    head_dim: int = 128
    max_position_embeddings: int = 2048
    rope_theta: float = ROPE_THETA
    rms_norm_eps: float = NORM_EPS
    num_prelude_layers: int = 2
    num_recurrent_layers: int = 4
    num_coda_layers: int = 2
    recurrence_steps: int = 4
    backprop_steps: int = 4
    untie_loops: bool = False
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_attention_heads", "head_dim",
                     "max_position_embeddings", "num_recurrent_layers", "recurrence_steps", "backprop_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.num_prelude_layers) < 0 or int(self.num_coda_layers) < 0:
            raise ValueError("prelude and coda layer counts cannot be negative")
        if int(self.head_dim) % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if int(self.hidden_size) != int(self.num_attention_heads) * int(self.head_dim):
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if int(self.intermediate_size) != 4 * int(self.hidden_size):
            raise ValueError("the feed-forward width must be 4 * hidden_size")
        if float(self.rms_norm_eps) != NORM_EPS:
            raise ValueError(f"the model uses RMSNorm epsilon {NORM_EPS}")
        validate_rope_theta(self.rope_theta)
        if self.padded_vocab_size is None:
            self.padded_vocab_size = ((self.vocab_size + 63) // 64) * 64
        if self.padded_vocab_size < self.vocab_size:
            raise ValueError("padded_vocab_size must cover the active vocabulary")
        if self.untie_loops and int(self.recurrence_steps) < 2:
            raise ValueError("untie_loops needs at least two loops")

    @property
    def physical_depth(self) -> int:
        return int(self.num_prelude_layers) + int(self.num_recurrent_layers) + int(self.num_coda_layers)

    @property
    def logical_depth(self) -> int:
        return (int(self.num_prelude_layers) + int(self.recurrence_steps) * int(self.num_recurrent_layers)
                + int(self.num_coda_layers))


def parameter_counts(width: int, layers: int, padded_vocab: int = PADDED_VOCAB_SIZE) -> dict[str, int]:
    """Non-embedding parameters of `layers` blocks plus the model-level norms, and the total with both
    vocabulary matrices; the input injection (2 * width^2 per loop copy) is counted separately."""
    matrices = layers * 12 * width**2
    norms = (2 * layers + 3) * width
    return {
        "non_embedding": matrices + norms,
        "total": matrices + norms + 2 * padded_vocab * width,
    }
