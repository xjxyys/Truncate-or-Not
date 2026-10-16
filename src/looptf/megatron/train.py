"""Train a looped transformer with Megatron-LM.

    torchrun --nproc-per-node=8 -m looptf.megatron.train --megatron-lm-path ../Megatron-LM \
        --hidden-size 768 --num-attention-heads 6 --prelude-layers 2 --recurrent-layers 4 --coda-layers 2 \
        --loops 4 --backprop-loops 2 --micro-batch-size 4 --global-batch-size 128 --train-iters 4096 ...

Every Megatron option that the model or the recipe fixes is set by the trainer (looptf/megatron/arguments.py);
scripts/train.py builds the full command for the runs of the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any

import torch

from looptf.data.online import close_online_dataloaders, dataset_provider

from .compat import (
    ensure_megatron,
    make_wandb_checkpoint_callback_nonfatal,
    stream_wandb_file_hashes,
    use_resolved_args_in_ft_setup,
)

BATCH_KEYS = ("attention_mask", "cu_seqlens", "cu_seqlens_padded", "hybrid_cp_group", "labels", "local_cp_size",
              "loss_mask", "max_seqlen", "position_ids", "tokens")


def get_batch(data_iterator: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Megatron's standard GPT batch path: tokens, labels and loss mask on this rank's GPU."""
    from megatron.core import mpu
    from megatron.core.parallel_state import get_context_parallel_group, get_hybrid_data_context_parallel_groups
    from megatron.core.utils import (
        flatten_batch_for_packed_sequences,
        get_batch_on_this_cp_rank,
        get_batch_on_this_tp_rank,
    )
    from megatron.training import get_args

    args = get_args()
    batch: dict[str, Any] = {}
    if mpu.get_tensor_model_parallel_rank() == 0:
        batch = next(data_iterator)
        for key in BATCH_KEYS:
            value = batch.get(key)
            batch[key] = value.cuda(non_blocking=True) if value is not None else None
    batch = get_batch_on_this_tp_rank(
        batch,
        broadcast_src_rank=mpu.get_tensor_model_parallel_src_rank(),
        broadcast_group=mpu.get_tensor_model_parallel_group(),
        has_cu_seqlens=False,
        is_hybrid_cp=False,
        create_attention_mask_in_dataloader=args.create_attention_mask_in_dataloader,
        cp_size=args.context_parallel_size,
        tp_rank=mpu.get_tensor_model_parallel_rank(),
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        mtp_on_this_rank=False,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        is_pipeline_first_stage=mpu.is_pipeline_first_stage(),
        is_pipeline_last_stage=mpu.is_pipeline_last_stage(),
    )
    batch = flatten_batch_for_packed_sequences(batch)
    batch = get_batch_on_this_cp_rank(
        batch,
        is_hybrid_cp=False,
        cp_group=get_context_parallel_group(),
        hybrid_cp_group_func=get_hybrid_data_context_parallel_groups,
        use_per_sequence_balancing=False,
    )
    return batch["tokens"], batch["labels"], batch["loss_mask"]


def loss_func(loss_mask: torch.Tensor, loss_per_token: torch.Tensor):
    """Megatron's loss contract: the summed loss, its token count, and the values to report."""
    mask = loss_mask.float()
    num_tokens = mask.sum().detach().to(torch.int)
    loss = (loss_per_token.float() * mask).sum()
    return loss, num_tokens, {"lm loss": torch.cat((loss.detach().view(1), num_tokens.view(1)))}


def forward_step(data_iterator: Any, model: Any):
    from .evaluation import record_batch

    tokens, labels, loss_mask = get_batch(data_iterator)
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    config = inner.looped_config
    # Training backpropagates the last B of the T loops; evaluation runs all T without gradients.
    backprop_loops = config.backprop_loops if inner.training else 0
    record_batch(tokens, loss_mask, config.loops)
    loss_per_token = model(tokens, labels=labels, num_steps=config.loops, backprop_steps=backprop_loops)
    return loss_per_token, partial(loss_func, loss_mask)


def _write_receipt(args: Any, name: str, payload: dict[str, Any]) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    print(json.dumps({name: payload}), flush=True)
    if args.save:
        path = Path(args.save).parent / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    preliminary = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    preliminary.add_argument("--megatron-lm-path", default=os.environ.get("MEGATRON_LM_PATH"))
    ensure_megatron(preliminary.parse_known_args(argv)[0].megatron_lm_path)

    from megatron.core.enums import ModelType
    from megatron.training import pretrain, set_startup_timestamps
    from megatron.training.argument_utils import pretrain_cfg_container_from_args
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.global_vars import get_wandb_writer

    from .arguments import add_arguments, parser_defaults, recipe_metadata, resolve_recipe, validate_args
    from .evaluation import evaluation_metadata, install_evaluation_hooks
    from .model import build_model_config
    from .optimizer import install_optimizer

    program_start = time.time()
    saved_argv = sys.argv
    sys.argv = [sys.argv[0], *argv]
    try:
        defaults = parser_defaults(argv)
        args = parse_and_validate_args(extra_args_provider=lambda parser: add_arguments(parser, defaults))
        validate_args(args, defaults)
        resolve_recipe(args)
        recipe = recipe_metadata(args)
        _write_receipt(args, "training_recipe", recipe)
        _write_receipt(args, "evaluation", evaluation_metadata(args))
        writer = get_wandb_writer()
        if writer is not None:
            writer.config.update({"training_recipe": recipe}, allow_val_change=True)
        model_config = build_model_config(args)
        with ExitStack() as hooks:
            hooks.callback(use_resolved_args_in_ft_setup(args))
            if args.wandb_project:
                hooks.callback(stream_wandb_file_hashes())
                make_wandb_checkpoint_callback_nonfatal()
            hooks.callback(install_optimizer(args))
            hooks.callback(install_evaluation_hooks(args))
            config = pretrain_cfg_container_from_args(args, model_config)
            set_startup_timestamps(program_start=program_start, main_entry=time.time())
            pretrain(config, dataset_provider, ModelType.encoder_or_decoder, forward_step,
                     get_embedding_ranks=lambda pipeline_ranks: [pipeline_ranks[0]])
    finally:
        try:
            close_online_dataloaders()
        finally:
            sys.argv = saved_argv


if __name__ == "__main__":
    main()
