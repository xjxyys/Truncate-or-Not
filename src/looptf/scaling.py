"""Model sizes and per-budget training settings of the paper's scaling grid.

Sizes: width d = 128 x heads, feed-forward width 4d, prelude / recurrent / coda layers (P, R, C). 79M, 162M and 344M
are the widened models of the compute-matched (T, B) = (2, 2) comparison.

Settings at a token budget D (in units of D0 = 2^30 tokens):
  global batch  64 sqrt(D / D0) sequences of 2,048 tokens, rounded in log space to the nearest of
                16, 24, 32, 48, 64, 88, 128, 176, 256, 352, 512
  steps         D / (batch x 2048), rounded up to a multiple of 8
  schedule      linear warmup over round(0.02 x steps) steps, then cosine decay to 0.1x the peak
  evaluation    every steps / 8 steps on 4,096 held-out sequences, at the end on 69,120; batches that do not
                divide 4,096 evaluate with a batch of 64
  learning rate multiplier 2^(k/8) nearest to (D / D0)^-0.08 on top of the recipe's HyperP transfer, which raises
                the token exponent from 0.32 to 0.40; the D0/16 runs used 1.1892

Training compute is counted as C = F(T, B) D + F_opt S (S optimizer steps): F is the forward + backward FLOPs per
token with the accounting of Schwethelm et al. (2026, Sec. 3.2), 6 FLOPs per parameter and token for the blocks
run once, 2 per forward and 4 per backward loop of the recurrent block, plus attention (the embedding lookup is
free, the output layer is a matrix multiply); F_opt counts the matrix multiplies of MuonH's five Newton-Schulz
iterations per step (an s x l matrix, s <= l, costs 4 s^2 l + 2 s^3 per iteration), once and not per data-parallel
rank. Element-wise work (norms, activations, AdamW, momentum) is not counted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model.config import PADDED_VOCAB_SIZE
from .recipe import LR_MULTIPLIERS, SEQUENCE_LENGTH

D0 = 2**30
SIZES = {
    "60M": (768, 2, 4, 2),
    "120M": (1024, 2, 5, 2),
    "240M": (1280, 3, 6, 3),
    "480M": (1664, 3, 7, 4),
    "1B": (2176, 4, 8, 4),
    "79M": (896, 2, 4, 2),
    "162M": (1152, 2, 5, 3),
    "344M": (1536, 3, 6, 3),
}
MAIN_SIZES = ("60M", "120M", "240M", "480M", "1B")
BATCH_LADDER = (16, 24, 32, 48, 64, 88, 128, 176, 256, 352, 512)
BUDGETS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32, 64)
EVAL_SEQUENCES = 4096
FINAL_EVAL_SEQUENCES = 69120
DEFAULT_EVAL_BATCH = 64


def global_batch(budget: float) -> int:
    """Global batch in sequences at `budget` x D0 tokens."""
    return min(BATCH_LADDER, key=lambda b: abs(math.log(b / (64 * math.sqrt(budget)))))


def train_iters(budget: float, batch: int) -> int:
    return 8 * math.ceil(budget * D0 / (batch * SEQUENCE_LENGTH) / 8)


def lr_multiplier(budget: float) -> str:
    """Label of the learning-rate multiplier (a key of looptf.recipe.LR_MULTIPLIERS)."""
    k = 2 if budget == 1 / 16 else round(8 * math.log2(budget**-0.08))
    return min(LR_MULTIPLIERS, key=lambda label: abs(float(label) - 2 ** (k / 8)))


@dataclass(frozen=True)
class Evaluation:
    interval: int
    global_batch_size: int
    iters: int
    final_iters: int


def warmup_iters(steps: int) -> int:
    return round(0.02 * steps)


def evaluation(steps: int, batch: int) -> Evaluation:
    eval_batch = batch if EVAL_SEQUENCES % batch == 0 else DEFAULT_EVAL_BATCH
    return Evaluation(max(1, steps // 8), eval_batch, EVAL_SEQUENCES // eval_batch, FINAL_EVAL_SEQUENCES // eval_batch)


def parameter_counts(width: int, prelude: int, recurrent: int, coda: int) -> tuple[int, int]:
    """(N_once, N_rec): prelude and coda blocks with the embedding and final norms; recurrent block with the input
    injection and the recurrent norm. Non-embedding parameters N = N_once + N_rec (one copy of the block)."""
    outer = prelude + coda
    once = 12 * outer * width**2 + (2 * outer + 2) * width
    rec = (12 * recurrent + 2) * width**2 + (2 * recurrent + 1) * width
    return once, rec


def flops_per_token(width: int, outer_layers: int, recurrent_layers: int, loops: int, backprop_loops: int) -> int:
    """F(T, B): forward + backward FLOPs per training token."""
    d, u, k = width, outer_layers, recurrent_layers
    return (6 * (12 * u * d * d + PADDED_VOCAB_SIZE * d) + (2 * loops + 4 * backprop_loops) * (12 * k + 2) * d * d
            + d * SEQUENCE_LENGTH * (12 * u + (4 * loops + 8 * backprop_loops) * k))


def _newton_schulz(rows: int, cols: int, iterations: int = 5) -> int:
    short, long_side = min(rows, cols), max(rows, cols)
    return iterations * (4 * short * short * long_side + 2 * short**3)


def optimizer_flops_per_step(width: int, outer_layers: int, recurrent_layers: int, copies: int = 1) -> int:
    """F_opt; `copies` is the number of recurrent-block copies (T for untied loops)."""
    layer = 4 * _newton_schulz(width, width) + 2 * _newton_schulz(4 * width, width)
    return (outer_layers + copies * recurrent_layers) * layer + copies * _newton_schulz(width, 2 * width)
