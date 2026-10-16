"""MuonH for matrices and AdamW for everything else.

Adapted from nanochat/optim.py of kschwethelm/looped-lm-scaling (MIT, Copyright (c) 2026 Kristian Schwethelm),
which builds on the Muon optimizer of KellerJordan/modded-nanogpt (MIT) and on karpathy/nanochat (MIT); see
NOTICE. The orthogonalization uses the five Polar Express iterations of Amsel et al. (2025, arXiv:2505.16932).
MuonH keeps no second moment: every update and every weight is rescaled to the Frobenius norm of the matrix at
initialization (a fixed-radius sphere), so weight decay is unnecessary and must be zero.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from looptf.recipe import MUONH_MOMENTUM

POLAR_EXPRESS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def muonh_update_(weight, grad, momentum, radius, lr):
    """Nesterov momentum, Polar Express orthogonalization in BF16, and a step on the fixed-radius sphere."""
    mu = grad.new_tensor(MUONH_MOMENTUM)
    momentum.lerp_(grad, 1 - mu)
    g = grad.lerp(momentum, mu)
    x = g.bfloat16()
    x = x / (x.norm() * 1.02 + 1e-6)
    for a, b, c in POLAR_EXPRESS:
        if x.shape[-2] > x.shape[-1]:
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
        else:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    x = x * (radius.to(x.dtype) / x.norm().clamp_min(1e-10))
    weight.sub_(torch.as_tensor(lr, device=x.device, dtype=x.dtype) * x)
    weight.mul_(radius.to(weight.dtype) / weight.norm().clamp_min(1e-10))


class MuonHAdamW(Optimizer):
    """Parameter groups carry a role: "matrix" groups use MuonH, the others AdamW with the group's betas/eps."""

    def __init__(self, groups):
        super().__init__(groups, defaults={})
        for group in self.param_groups:
            if group["weight_decay"] != 0:
                raise ValueError("MuonH groups require zero weight decay")

    def load_state_dict(self, state_dict):
        groups = state_dict.get("param_groups", [])
        if len(groups) != len(self.param_groups) or any(
            any(old.get(key) != new.get(key) for key in ("role", "_base_lr", "param_names"))
            for old, new in zip(groups, self.param_groups)
        ):
            raise ValueError("checkpoint optimizer groups or peak learning rates differ from this run")
        return super().load_state_dict(state_dict)

    def initialize_state(self):
        """Also called by Megatron before it loads a checkpoint."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if state:
                    continue
                if group["role"] == "matrix":
                    if p.ndim != 2:
                        raise ValueError("MuonH requires a 2D matrix")
                    radius = p.detach().norm().item()
                    if radius == 0:
                        raise ValueError("MuonH cannot train a zero-initialized matrix")
                    state["radius"] = radius
                    state["momentum_buffer"] = torch.zeros_like(p)
                else:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.initialize_state()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise ValueError("MuonH and AdamW require dense gradients")
                state = self.state[p]
                if group["role"] == "matrix":
                    weight = p.detach()
                    muonh_update_(weight, grad, state["momentum_buffer"], weight.new_tensor(state["radius"]),
                                  group["lr"])
                else:
                    beta1, beta2 = group["betas"]
                    state["step"] += 1
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.lerp_(grad, 1 - beta1)
                    v.lerp_(grad.square(), 1 - beta2)
                    denom = (v / (1 - beta2 ** state["step"])).sqrt() + group["eps"]
                    p.addcdiv_(m, denom, value=-group["lr"] / (1 - beta1 ** state["step"]))
        return loss
