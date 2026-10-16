"""Layers of the looped transformer: RMSNorm, BF16 projections, rotary attention and initialization.

The block follows Schwethelm et al. (2026, arXiv:2604.21106): pre-norm, parameter-free QK RMSNorm after
RoPE, squared-ReLU feed-forward, no biases. The residual stream and the norm scales stay in FP32 while the
projections run in the weight dtype (BF16 in training).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROPE_THETA = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        # Megatron's Float16Module keeps parameters marked this way in FP32.
        self.weight.keep_in_fp32 = True

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps) * self.weight


class Linear(nn.Linear):
    """Projection computed in the weight dtype, applied to the FP32 residual stream."""

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs.to(self.weight.dtype), self.weight, self.bias)


def squared_relu(x: Tensor) -> Tensor:
    return F.relu(x).square()


@torch.no_grad()
def initialize_weight(weight: Tensor, role: str = "matrix") -> None:
    """Sample in FP32, then copy into the parameter's storage dtype."""
    sample = torch.empty_like(weight, dtype=torch.float32)
    if role == "embedding":
        nn.init.normal_(sample, std=1.0)
    elif role == "output":
        nn.init.normal_(sample, std=0.001)
    elif role == "matrix":
        bound = math.sqrt(3 / weight.shape[1])
        nn.init.uniform_(sample, a=-bound, b=bound)
    else:
        raise ValueError(f"unknown parameter role: {role}")
    weight.copy_(sample)


def validate_rope_theta(value) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("RoPE base must be finite and positive")
    return value


class RotaryAttention(nn.Module):
    """Causal SDPA with RoPE followed by a parameter-free RMSNorm of Q and K.

    The cos/sin tables are kept in BF16, as in the reference implementation, also when the model runs in FP32.
    """

    def __init__(self, rope_theta: float = ROPE_THETA, qk_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.rope_theta = validate_rope_theta(rope_theta)
        self.qk_norm_eps = float(qk_norm_eps)
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def _rotate(self, x: Tensor, positions: Tensor, max_seqlen: int) -> Tensor:
        dim = x.shape[-1]
        if self.cos.device != x.device or self.cos.size(0) < max_seqlen:
            freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
            angles = torch.outer(torch.arange(max_seqlen, device=x.device).float(), freq)
            self.cos = angles.cos().bfloat16()[:, None, :]
            self.sin = angles.sin().bfloat16()[:, None, :]
        cos, sin = self.cos[positions], self.sin[positions]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1)

    def _project(self, x: Tensor, positions: Tensor, max_seqlen: int) -> Tensor:
        x = self._rotate(x, positions, max_seqlen)
        return F.rms_norm(x, (x.size(-1),), eps=self.qk_norm_eps)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Q/K/V in [sequence, batch, heads, dim]; returns [sequence, batch, heads * dim].

        The sequence-first layout and the copies it implies set the memory layouts that the GPU kernels see; they
        are part of the numerics of the paper's runs.
        """
        if query.ndim != 4:
            raise ValueError("self-attention expects Q/K/V [sequence, batch, heads, dim]")
        if key.shape != value.shape or query.shape != key.shape:
            raise ValueError("self-attention expects Q, K and V of identical shape")
        sequence, batch, heads, dim = query.shape
        q, k, v = (x.transpose(0, 1).reshape(-1, heads, dim) for x in (query, key, value))
        positions = torch.arange(sequence, device=query.device).repeat(batch)
        q, k = (self._project(x, positions, sequence) for x in (q, k))
        q, k, v = (x.reshape(-1, sequence, heads, dim).transpose(1, 2) for x in (q, k, v))
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).reshape(-1, heads, dim)
        return output.reshape(batch, sequence, heads * dim).transpose(0, 1).contiguous()
