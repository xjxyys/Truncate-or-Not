"""MuonH + AdamW and the learning-rate schedule in place of Megatron's optimizer and scheduler.

Megatron's mixed-precision wrapper keeps owning the FP32 main parameters and gradients, the gradient-norm
computation and the checkpoint layout; only the inner optimizer and the per-step learning rates are replaced.
"""

from __future__ import annotations

import json
from typing import Any

from looptf.optim import build_optimizer
from looptf.recipe import ADAM_BETAS, FINAL_LR_FRACTION, LearningRateTransfer, Schedule

from .arguments import schedule


class ReportUnclippedGradNorm:
    """Report the gradient norm although the recipe does not clip.

    Without clipping Megatron's BF16 wrapper returns a placeholder; the norm is measured after the main gradients
    are copied and unscaled, before the update, with Megatron's handling of parallel groups and shared parameters.
    """

    def prepare_grads(self):
        found_inf = super().prepare_grads()
        self._unclipped_grad_norm = None
        if not found_inf and self.config.clip_grad <= 0.0:
            self._unclipped_grad_norm = self.get_grad_norm()
        return found_inf

    def step(self):
        success, grad_norm, num_zeros = super().step()
        if success and self.config.clip_grad <= 0.0:
            grad_norm = self._unclipped_grad_norm
        return success, grad_norm, num_zeros


class KeepRoleOrder:
    @staticmethod
    def _filter_and_reorder_param_groups(current_groups, state_dict_groups):
        """Megatron matches groups by learning rate and weight decay, which cannot tell the norm and other roles
        apart (same peak); the groups are built deterministically, so a resume must find them in the same order."""
        if [g.get("role") for g in current_groups] != [g.get("role") for g in state_dict_groups]:
            raise ValueError("the checkpoint's optimizer parameter roles differ from this run")
        return state_dict_groups


class RecipeScheduler:
    """Sets every group's learning rate after each update; Megatron counts progress in sequences."""

    def __init__(self, optimizer: Any, plan: Schedule, train_iters: int, global_batch_size: int) -> None:
        self.optimizer = optimizer
        self.schedule = plan
        self.total_steps = train_iters
        self.batch_size = global_batch_size
        self.num_steps = 0
        self.step(0)

    def step(self, increment: int) -> None:
        self.num_steps += increment
        scale = self.schedule.lr_scale(self.num_steps / self.batch_size, self.total_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = group["_base_lr"] * scale
            group["weight_decay"] = 0.0

    def _signature(self) -> dict[str, Any]:
        return {"schedule": json.loads(json.dumps(self.schedule.__dict__)), "total_steps": self.total_steps,
                "batch_size": self.batch_size}

    def state_dict(self) -> dict[str, Any]:
        return {**self._signature(), "num_steps": self.num_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if {key: state.get(key) for key in self._signature()} != self._signature():
            raise ValueError("the checkpoint's schedule, length or batch differs from this run")
        self.num_steps = 0
        self.step(state["num_steps"])


def install_optimizer(args: Any):
    """Replace Megatron's optimizer and scheduler factories; returns a function that restores them."""
    import torch
    from megatron.core import parallel_state
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
    from megatron.training import training

    original_optimizer = training.get_megatron_optimizer
    original_scheduler = training.get_optimizer_param_scheduler

    class BF16Optimizer(ReportUnclippedGradNorm, KeepRoleOrder, Float16OptimizerWithFloat16Params):
        pass

    class FP32RecipeOptimizer(ReportUnclippedGradNorm, KeepRoleOrder, FP32Optimizer):
        pass

    def get_megatron_optimizer(config, model_chunks, **kwargs):
        batch_tokens = args.global_batch_size * args.seq_length
        transfer = LearningRateTransfer(args.hidden_size, args.train_iters * batch_tokens, batch_tokens)
        optimizer = build_optimizer(torch.nn.ModuleList(model_chunks), args.base_lrs, transfer, ADAM_BETAS)
        for group in optimizer.param_groups:
            group.update(default_config=group["role"] == "matrix", max_lr=group["_base_lr"],
                         min_lr=FINAL_LR_FRACTION * group["_base_lr"], lr_mult=1.0, wd_mult=1.0,
                         is_expert_parallel=False, is_decoupled_lr=False)

        def init_state(inner, config=None):
            inner.initialize_state()

        if config.bf16:
            wrapped = BF16Optimizer(optimizer, config, None, init_state)
        else:
            wrapped = FP32RecipeOptimizer(optimizer, config, init_state)
        wrapped.tp_group = parallel_state.get_tensor_model_parallel_group()
        wrapped.expert_tp_group = wrapped.tp_group
        return wrapped

    def get_optimizer_param_scheduler(optimizer):
        return RecipeScheduler(optimizer, schedule(args), args.train_iters, args.global_batch_size)

    training.get_megatron_optimizer = get_megatron_optimizer
    training.get_optimizer_param_scheduler = get_optimizer_param_scheduler

    def restore() -> None:
        training.get_megatron_optimizer = original_optimizer
        training.get_optimizer_param_scheduler = original_scheduler

    return restore
