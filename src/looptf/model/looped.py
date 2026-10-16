"""Looped transformer: a prelude, one recurrent block applied T times, and a coda.

Each loop starts with the input injection W [e ; s] of the prelude output e and the current state s and ends with
an RMSNorm. The state starts at e and W is initialized to [I | 0], so the first loop reads the prelude output
unchanged. Truncated BPTT runs the first T - B loops without gradients and backpropagates through the last B.
With `untie_loops`, every loop has its own copy of the recurrent block, the injection and the closing norm.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .block import Block
from .config import ModelConfig
from .layers import Linear, RMSNorm, initialize_weight
from .losses import next_token_loss

LOGIT_SOFTCAP = 15.0


@dataclass(frozen=True)
class DepthSchedule:
    """One forward pass: `no_grad_steps` loops without gradients followed by `grad_steps` backpropagated loops."""

    no_grad_steps: int
    grad_steps: int

    def __post_init__(self) -> None:
        if self.no_grad_steps < 0 or self.grad_steps < 0:
            raise ValueError("loop counts cannot be negative")
        if self.total_steps < 1:
            raise ValueError("a forward pass needs at least one loop")

    @property
    def total_steps(self) -> int:
        return self.no_grad_steps + self.grad_steps


class InputInjection(nn.Module):
    """W [e ; s]: the prelude output e first, then the recurrent state s."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        # The paper's runs created a default nn.Linear here and then replaced it; both constructions draw from
        # the global RNG before the model is initialized, so both are kept to reproduce their initial weights.
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.proj = Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(self, state: Tensor, prepared_input: Tensor) -> Tensor:
        return self.proj(torch.cat((prepared_input, state), dim=-1))


@dataclass
class LoopedOutput:
    logits: Tensor
    loss: Tensor | None = None


class LoopedTransformer(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or ModelConfig()
        hidden_size = int(config.hidden_size)
        vocab = int(config.padded_vocab_size)
        self.embed_tokens = nn.Embedding(vocab, hidden_size)
        self.prelude = nn.ModuleList(Block(config) for _ in range(int(config.num_prelude_layers)))
        self.recurrent_core = nn.ModuleList(Block(config) for _ in range(int(config.num_recurrent_layers)))
        self.coda = nn.ModuleList(Block(config) for _ in range(int(config.num_coda_layers)))
        self.input_write = InputInjection(hidden_size)
        self.norm_emb = RMSNorm(hidden_size)
        self.norm_recur = RMSNorm(hidden_size)
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = Linear(hidden_size, vocab, bias=False)
        self._reset_parameters()
        if config.untie_loops:
            self._add_untied_copies()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    nn.init.ones_(module.weight)
                elif isinstance(module, (nn.Linear, nn.Embedding)):
                    role = ("embedding" if module is self.embed_tokens else
                            "output" if module is self.lm_head else "matrix")
                    initialize_weight(module.weight, role)
            self._init_injection(self.input_write)

    def _init_injection(self, injection: InputInjection) -> None:
        # [I | 0] over [e ; s]; its Frobenius norm sqrt(width) matches the uniform initialization in expectation,
        # so the MuonH sphere radius is unaffected.
        width = int(self.config.hidden_size)
        weight = injection.proj.weight
        weight.zero_()
        weight[:, :width].copy_(torch.eye(width, dtype=weight.dtype))

    def _add_untied_copies(self) -> None:
        """Copies 1 .. T-1 of the recurrent block, injection and recurrent norm (copy 0 is the tied modules),
        initialized after the tied model from the global RNG, so all data-parallel ranks agree."""
        cores, writes, norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        with torch.no_grad():
            for _ in range(int(self.config.recurrence_steps) - 1):
                core = copy.deepcopy(self.recurrent_core)
                write = copy.deepcopy(self.input_write)
                norm = copy.deepcopy(self.norm_recur)
                for module in core.modules():
                    if isinstance(module, RMSNorm):
                        nn.init.ones_(module.weight)
                    elif isinstance(module, nn.Linear):
                        initialize_weight(module.weight, "matrix")
                self._init_injection(write)
                nn.init.ones_(norm.weight)
                cores.append(core)
                writes.append(write)
                norms.append(norm)
        self.untied_recurrent_cores, self.untied_input_writes, self.untied_norm_recurs = cores, writes, norms

    def _loop_modules(self, loop: int) -> tuple[nn.Module, nn.ModuleList, nn.Module]:
        if loop == 0 or not self.config.untie_loops:
            return self.input_write, self.recurrent_core, self.norm_recur
        return (self.untied_input_writes[loop - 1], self.untied_recurrent_cores[loop - 1],
                self.untied_norm_recurs[loop - 1])

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def _schedule(self, num_steps: int | None, backprop_steps: int | None) -> DepthSchedule:
        if num_steps is not None:
            if num_steps < 1:
                raise ValueError("num_steps must be positive")
            if backprop_steps is None:
                backprop_steps = min(int(self.config.backprop_steps), num_steps) if self.training else 0
            if not 0 <= backprop_steps <= num_steps:
                raise ValueError("backprop_steps must be between zero and num_steps")
            return DepthSchedule(num_steps - backprop_steps, backprop_steps)
        if backprop_steps is not None:
            raise ValueError("backprop_steps requires num_steps")
        count = int(self.config.recurrence_steps)
        if self.training:
            grads = min(count, int(self.config.backprop_steps))
            return DepthSchedule(count - grads, grads)
        return DepthSchedule(count, 0)

    @staticmethod
    def _run_layers(hidden: Tensor, layers: nn.ModuleList) -> Tensor:
        for layer in layers:
            hidden = layer(hidden)
        return hidden

    def _loop(self, state: Tensor, prepared_input: Tensor, loop: int, *, checkpoint_step: bool) -> Tensor:
        write, core, norm = self._loop_modules(loop)

        def run(hidden: Tensor, fixed_input: Tensor) -> Tensor:
            hidden = write(hidden, fixed_input)
            # The injection is a BF16 projection; the recurrent blocks take the FP32 residual stream.
            hidden = hidden.float()
            hidden = self._run_layers(hidden, core)
            return norm(hidden)

        if checkpoint_step:
            # One activation checkpoint per loop, including its input injection.
            return checkpoint(run, state, prepared_input, use_reentrant=False)
        return run(state, prepared_input)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        *,
        num_steps: int | None = None,
        backprop_steps: int | None = None,
    ) -> LoopedOutput:
        """Logits over the active vocabulary for causal input_ids [batch, sequence]; with `labels` (usually the
        input_ids), also the mean next-token loss. `num_steps` loops (default: recurrence_steps) of which the last
        `backprop_steps` are backpropagated (default in training: backprop_steps; in evaluation: none)."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > int(self.config.max_position_embeddings):
            raise ValueError("sequence exceeds max_position_embeddings")
        prepared_input = self.norm_emb(self.embed_tokens(input_ids).float())
        prepared_input = self._run_layers(prepared_input, self.prelude)
        state = prepared_input.clone()

        schedule = self._schedule(num_steps, backprop_steps)
        if self.config.untie_loops and schedule.total_steps != int(self.config.recurrence_steps):
            raise ValueError("with untie_loops every forward pass must run all recurrence_steps loops")
        loop = 0
        with torch.no_grad() if schedule.no_grad_steps else contextlib.nullcontext():
            for _ in range(schedule.no_grad_steps):
                state = self._loop(state, prepared_input, loop, checkpoint_step=False)
                loop += 1
        for _ in range(schedule.grad_steps):
            state = self._loop(state, prepared_input, loop,
                               checkpoint_step=self.config.gradient_checkpointing and self.training)
            loop += 1

        hidden = self._run_layers(state, self.coda)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden).float()
        logits = LOGIT_SOFTCAP * torch.tanh(logits[..., : self.config.vocab_size] / LOGIT_SOFTCAP)
        return LoopedOutput(logits=logits, loss=None if labels is None else next_token_loss(logits, labels))
