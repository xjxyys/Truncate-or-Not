"""Transformer block shared by the prelude, the recurrent block and the coda."""

from torch import Tensor, nn

from .layers import Linear, RMSNorm, RotaryAttention, squared_relu


class SelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.core = RotaryAttention(float(config.rope_theta))
        self.q_proj = Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, sequence, _ = inputs.shape
        query = self.q_proj(inputs).view(batch, sequence, self.num_heads, self.head_dim)
        key = self.k_proj(inputs).view(batch, sequence, self.num_heads, self.head_dim)
        value = self.v_proj(inputs).view(batch, sequence, self.num_heads, self.head_dim)
        attended = self.core(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1))
        return self.o_proj(attended.transpose(0, 1))


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.down_proj(squared_relu(self.up_proj(inputs)))


class Block(nn.Module):
    """Pre-norm attention and feed-forward with an FP32 residual stream."""

    def __init__(self, config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = SelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attn(self.attn_norm(inputs))
        return hidden + self.mlp(self.mlp_norm(hidden))
