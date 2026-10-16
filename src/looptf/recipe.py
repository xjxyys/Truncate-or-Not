"""Training recipe: MuonH for matrices and AdamW for embeddings, output layer and norms, with the HyperP
learning-rate transfer and schedule of Schwethelm et al. (2026, arXiv:2604.21106; reference implementation
kschwethelm/looped-lm-scaling, scripts/base_train.py).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

ADAM_BETAS = (0.8, 0.95)
ADAM_EPS = 1e-10
MUONH_MOMENTUM = 0.95
SEQUENCE_LENGTH = 2048
FINAL_LR_FRACTION = 0.1
ROLES = ("matrix", "embedding", "unembedding", "norm", "other")
BASE_LRS = {"matrix": 0.014, "embedding": 0.3, "unembedding": 0.004, "norm": 0.005, "other": 0.005}

# HyperP reference: width 640, the 0-20-0 model's 10 tokens per non-embedding parameter, 262,144-token batches.
REFERENCE_WIDTH = 640
REFERENCE_TOKENS = 10 * (20 * (12 * REFERENCE_WIDTH**2 + 2 * REFERENCE_WIDTH) + 3 * REFERENCE_WIDTH)
REFERENCE_BATCH_TOKENS = 262_144


def _scaled(matrix, embedding, unembedding, norm):
    return {"matrix": matrix, "embedding": embedding, "unembedding": unembedding, "norm": norm, "other": norm}


# Base learning rates times 2^(k/8), as the decimal values the paper's runs used.
LR_MULTIPLIERS = {
    "1": dict(BASE_LRS),
    "0.25": _scaled(0.0035, 0.075, 0.001, 0.00125),
    "0.3536": _scaled(0.0049497475, 0.1060660172, 0.0014142136, 0.0017677670),
    "0.3856": _scaled(0.0053977379, 0.1156658119, 0.0015422108, 0.0019277635),
    "0.4204": _scaled(0.0058862749, 0.1261344623, 0.0016817928, 0.0021022410),
    "0.4585": _scaled(0.0064190283, 0.1375506065, 0.0018340081, 0.0022925101),
    "0.5": _scaled(0.007, 0.15, 0.002, 0.0025),
    "0.5453": _scaled(0.0076335541, 0.1635761599, 0.0021810155, 0.0027262693),
    "0.5946": _scaled(0.0083244498, 0.1783810673, 0.0023784142, 0.0029730178),
    "0.6484": _scaled(0.0090778769, 0.1945259332, 0.0025936791, 0.0032420989),
    "0.7071": _scaled(0.0098994949, 0.2121320344, 0.0028284271, 0.0035355339),
    "0.7711": _scaled(0.0107954758, 0.2313316238, 0.0030844217, 0.0038555271),
    "0.8409": _scaled(0.0117725498, 0.2522689246, 0.0033635857, 0.0042044821),
    "0.9170": _scaled(0.0128380566, 0.2751012130, 0.0036680162, 0.0045850202),
    "1.0905": _scaled(0.0152671083, 0.3271523198, 0.0043620309, 0.0054525387),
    "1.1892": _scaled(0.0166488996, 0.3567621345, 0.0047568285, 0.0059460356),
    "1.2968": _scaled(0.0181557538, 0.3890518664, 0.0051873582, 0.0064841978),
    "1.4142": _scaled(0.0197989899, 0.4242640687, 0.0056568542, 0.0070710678),
    "1.5422": _scaled(0.0215909516, 0.4626632476, 0.0061688433, 0.0077110541),
    "2": _scaled(0.028, 0.6, 0.008, 0.01),
    "4": _scaled(0.056, 1.2, 0.016, 0.02),
}


def base_lrs(lr_multiplier: str = "1") -> dict[str, float]:
    try:
        return dict(LR_MULTIPLIERS[lr_multiplier])
    except KeyError as error:
        raise ValueError(f"unsupported learning-rate multiplier {lr_multiplier!r}; choose one of "
                         f"{', '.join(LR_MULTIPLIERS)}") from error


@dataclass(frozen=True)
class LearningRateTransfer:
    """HyperP transfer of the base learning rates to a width, token horizon and batch."""

    width: int
    total_tokens: int
    batch_tokens: int = REFERENCE_BATCH_TOKENS
    reference_width: int = REFERENCE_WIDTH
    reference_tokens: int = REFERENCE_TOKENS
    reference_batch_tokens: int = REFERENCE_BATCH_TOKENS

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def peak_lrs(self, base: dict[str, float] | None = None) -> dict[str, float]:
        base = BASE_LRS if base is None else base
        width = (self.width / self.reference_width) ** -0.5
        batch = (self.batch_tokens / self.reference_batch_tokens) ** 0.558
        data = (self.total_tokens / self.reference_tokens) ** -0.32
        return {
            "matrix": base["matrix"] * batch * data,
            "embedding": base["embedding"] * width * batch,
            "unembedding": base["unembedding"] * width * batch,
            "norm": base["norm"] * width,
            "other": base["other"] * width,
        }


@dataclass(frozen=True)
class Schedule:
    """Linear warmup, then either a linear decay over the remaining steps (the recipe default) or a stable phase
    followed by a cooldown that starts at int(steps * (1 - cooldown_fraction))."""

    kind: str = "linear"
    warmup_steps: int = 0
    cooldown_fraction: float = 0.0
    cooldown_style: str = "linear"
    final_lr_fraction: float = FINAL_LR_FRACTION

    def __post_init__(self):
        if self.kind not in {"linear", "WSD"}:
            raise ValueError("schedule kind must be linear or WSD")
        if self.cooldown_style not in {"linear", "cosine"}:
            raise ValueError("cooldown style must be linear or cosine")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if not 0 <= self.cooldown_fraction < 1:
            raise ValueError("cooldown_fraction must be in [0, 1)")
        if not 0 <= self.final_lr_fraction <= 1:
            raise ValueError("final_lr_fraction must be in [0, 1]")

    @classmethod
    def cosine(cls, train_iters: int, warmup: int) -> Schedule:
        """Linear warmup over `warmup` steps, then cosine decay to 0.1x the peak over the remaining steps.

        cooldown_fraction = (T - W) / T puts the start of the cooldown at W; when W / T is not a dyadic fraction,
        floating-point truncation can move it by one, and (T - W - 0.5) / T is used instead.
        """
        if not 0 < warmup < train_iters:
            raise ValueError(f"warmup {warmup} must be in (0, {train_iters})")
        fraction = (train_iters - warmup) / train_iters
        if int(train_iters * (1 - fraction)) != warmup:
            fraction = (train_iters - warmup - 0.5) / train_iters
        if int(train_iters * (1 - fraction)) != warmup:
            raise ValueError(f"warmup {warmup} / {train_iters}: the cooldown would not start at the warmup end")
        return cls(kind="WSD", warmup_steps=warmup, cooldown_fraction=fraction, cooldown_style="cosine")

    @classmethod
    def named(cls, name: str, train_iters: int, warmup: int = 0) -> Schedule:
        """"linear": the recipe default, no warmup; "cosine": `warmup` steps of linear warmup, then cosine decay."""
        if name == "linear":
            if warmup:
                raise ValueError("the linear schedule has no warmup; use the cosine schedule")
            return cls()
        if name == "cosine":
            return cls.cosine(train_iters, warmup)
        raise ValueError(f"unknown schedule {name!r}; choose linear or cosine")

    def cooldown_iters(self, train_iters: int) -> int:
        return train_iters - int(train_iters * (1 - self.cooldown_fraction))

    def lr_scale(self, step: float, total_steps: int) -> float:
        """Learning-rate multiplier after `step` zero-based optimizer updates."""
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        warmup = self.warmup_steps
        start = int(total_steps * (1 - self.cooldown_fraction))
        if self.kind == "linear" and warmup >= total_steps:
            raise ValueError("warmup must leave at least one scheduled update")
        if self.kind == "WSD" and warmup > start:
            raise ValueError("warmup overlaps the cooldown")
        cooldown = self.kind == "WSD" and self.cooldown_fraction > 0 and step >= start
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        if self.kind == "linear":
            progress = min(max((step - warmup) / max(total_steps - 1 - warmup, 1), 0), 1)
            return 1 - (1 - self.final_lr_fraction) * progress
        if not cooldown:
            return 1.0
        progress = (
            1.0
            if start == total_steps - 1
            else min(max((step - start) / (total_steps - 1 - start), 0), 1)
        )
        decay = 0.5 * (1 + math.cos(math.pi * progress)) if self.cooldown_style == "cosine" else 1 - progress
        return self.final_lr_fraction + (1 - self.final_lr_fraction) * decay
