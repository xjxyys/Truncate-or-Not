"""Trainer options on top of Megatron-LM's command line.

Megatron parses and validates the command line. The trainer adds the loop, recipe, data and evaluation options and
sets every Megatron option that the model or the recipe fixes as a parser default, so a run passes only its shape,
loops, batch, length, learning-rate multiplier, schedule and paths. Options derived from others (layer count, FFN
width, query groups, schedule settings, peak learning rates) are computed by the trainer and must not be passed.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any

from looptf.data.online import add_online_data_arguments, validate_online_data_args
from looptf.data.tokenizer import validate_tokenizer
from looptf.model.config import NORM_EPS, VOCAB_SIZE
from looptf.recipe import (
    ADAM_BETAS,
    ADAM_EPS,
    FINAL_LR_FRACTION,
    LR_MULTIPLIERS,
    MUONH_MOMENTUM,
    SEQUENCE_LENGTH,
    LearningRateTransfer,
    Schedule,
    base_lrs,
)

from .evaluation import add_evaluation_arguments, validate_evaluation_args

# Megatron options fixed by the model (operators, vocabulary, sequence length, data path) and by the recipe.
FIXED = {
    "transformer_impl": "local",
    "gradient_accumulation_fusion": False,
    "bias_dropout_fusion": False,
    "apply_rope_fusion": False,
    "normalization": "RMSNorm",
    "layernorm_epsilon": NORM_EPS,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
    "position_embedding_type": "rope",
    "rotary_base": 10000,
    "add_bias_linear": False,
    "untie_embeddings_and_output_weights": True,
    "seq_length": SEQUENCE_LENGTH,
    "max_position_embeddings": SEQUENCE_LENGTH,
    "tokenizer_type": "HuggingFaceTokenizer",
    "vocab_size": VOCAB_SIZE,
    "make_vocab_size_divisible_by": 64,
    "create_attention_mask_in_dataloader": False,
    "dataloader_type": "external",
    "num_workers": 0,
    "optimizer": "muonh",
    "adam_beta1": ADAM_BETAS[0],
    "adam_beta2": ADAM_BETAS[1],
    "adam_eps": ADAM_EPS,
    "muon_momentum": MUONH_MOMENTUM,
    "muon_nesterov": True,
    "weight_decay": 0.0,
    "min_lr": 0.0,
    "clip_grad": 0.0,
    "bf16": True,
    "ckpt_format": "torch",
}
# Defaults that a run may change.
DEFAULTS = {"seed": 42}

UNSUPPORTED = (
    "attention_output_gate", "group_query_attention", "qk_layernorm", "qk_l2_norm", "swiglu", "add_qkv_bias",
    "window_size", "spec", "rope_scaling", "use_rope_scaling", "rotary_interleaved", "seq_len_interpolation_factor",
    "rotary_seq_len_interpolation_factor", "apply_residual_connection_post_layernorm", "apply_layernorm_1p",
    "apply_query_key_layer_scaling", "squared_relu", "squared_relu_with_offset", "quick_geglu", "openai_gelu",
    "use_te_activation_func", "onnx_safe", "softmax_scale", "attn_logit_softcapping", "logit_dtype",
    "fp32_residual_connection", "multi_latent_attention", "num_experts", "mtp_num_layers", "use_flash_attn",
    "use_fused_rotary_pos_emb", "enable_mhc_connections", "use_legacy_models", "use_mup", "fp16", "fp8", "fp4",
    "reset_position_ids", "reset_attention_mask", "eod_mask_loss", "recompute_granularity", "sequence_parallel",
    "sequence_packing_scheduler", "use_varlen_dataset", "sft", "use_distributed_optimizer",
    "use_layer_wise_distributed_optimizer", "use_torch_fsdp2", "use_megatron_fsdp", "optimizer_cpu_offload",
    "optimizer_cuda_graph", "use_precision_aware_optimizer", "rampup_batch_size", "step_batch_size_schedule",
    "train_samples", "lr_warmup_samples", "lr_warmup_fraction", "lr_wsd_decay_samples", "decoupled_lr",
    "decoupled_min_lr", "use_checkpoint_opt_param_scheduler", "override_opt_param_scheduler",
    "use_checkpoint_args", "pretrained_checkpoint", "finetune",
)


def schedule(args: Any) -> Schedule:
    return Schedule.named(args.lr_schedule, args.train_iters, args.lr_warmup_iters)


def _derived(argv: Sequence[str]) -> dict[str, Any]:
    """Megatron options that follow from the run's shape and schedule; they must be set before Megatron's
    validation, which runs before the trainer sees the parsed arguments."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name, default in (("--prelude-layers", 2), ("--recurrent-layers", 4), ("--coda-layers", 2)):
        parser.add_argument(name, type=int, default=default)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--train-iters", type=int)
    parser.add_argument("--lr-multiplier", default="1")
    parser.add_argument("--lr-schedule", default="linear")
    parser.add_argument("--lr-warmup-iters", type=int, default=0)
    known, _ = parser.parse_known_args(argv)
    derived = {
        "num_layers": known.prelude_layers + known.recurrent_layers + known.coda_layers,
        # The schedule scales the per-group peaks; Megatron's --lr only has to pass its own checks here.
        "lr": base_lrs(known.lr_multiplier)["matrix"],
    }
    if known.hidden_size:
        derived["ffn_hidden_size"] = 4 * known.hidden_size
    if known.num_attention_heads:
        derived["num_query_groups"] = known.num_attention_heads
    if known.train_iters:
        plan = Schedule.named(known.lr_schedule, known.train_iters, known.lr_warmup_iters)
        cosine = plan.kind == "WSD"
        derived.update(
            lr_decay_style="WSD" if cosine else "linear",
            lr_wsd_decay_style="cosine" if cosine else "linear",
            lr_wsd_decay_iters=plan.cooldown_iters(known.train_iters),
        )
    return derived


def parser_defaults(argv: Sequence[str]) -> dict[str, Any]:
    return {**FIXED, **DEFAULTS, **_derived(argv)}


def add_arguments(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> argparse.ArgumentParser:
    for action in parser._actions:
        if "--optimizer" in action.option_strings and action.choices is not None:
            action.choices = [*action.choices, "muonh"]
    group = parser.add_argument_group("looped transformer")
    group.add_argument("--megatron-lm-path", default=os.environ.get("MEGATRON_LM_PATH"))
    group.add_argument("--prelude-layers", type=int, default=2, help="Layers before the recurrent block.")
    group.add_argument("--recurrent-layers", type=int, default=4, help="Layers of the recurrent block.")
    group.add_argument("--coda-layers", type=int, default=2, help="Layers after the recurrent block.")
    group.add_argument("--loops", type=int, default=1, help="T: forward loops of the recurrent block.")
    group.add_argument("--backprop-loops", type=int, default=None,
                       help="B: number of final loops that are backpropagated (truncated BPTT if B < T; default T).")
    group.add_argument("--untie-loops", action="store_true",
                       help="Give every loop its own copy of the recurrent block, input injection and norm (B = T).")
    group.add_argument("--recompute-loops", action="store_true",
                       help="Recompute every backpropagated loop in the backward pass (activation checkpointing).")
    group = parser.add_argument_group("training recipe")
    group.add_argument("--lr-multiplier", choices=tuple(LR_MULTIPLIERS), default="1",
                       help="Multiplier on the base learning rates of every parameter role.")
    group.add_argument("--lr-schedule", choices=("linear", "cosine"), default="linear",
                       help="linear: decay to 0.1x the peak from the first step; cosine: --lr-warmup-iters steps "
                            "of linear warmup, then cosine decay to 0.1x the peak.")
    add_online_data_arguments(parser)
    add_evaluation_arguments(parser)
    parser.set_defaults(**defaults)
    return parser


def validate_args(args: Any, defaults: dict[str, Any]) -> None:
    errors = []
    for name, expected in defaults.items():
        if name not in DEFAULTS and getattr(args, name, None) != expected:
            errors.append(f"--{name.replace('_', '-')} is set by the trainer ({expected!r}); do not pass it")
    for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size",
                 "expert_model_parallel_size"):
        if getattr(args, name, 1) != 1:
            errors.append(f"{name} must be 1")
    errors += [f"--{name.replace('_', '-')} is not supported" for name in UNSUPPORTED if getattr(args, name, None)]
    if getattr(args, "rotary_percent", 1.0) != 1.0 or getattr(args, "softmax_type", "vanilla") != "vanilla":
        errors.append("the model uses full rotary embeddings and ordinary softmax attention")
    if getattr(args, "lr_decay_iters", None) not in (None, args.train_iters):
        errors.append("the schedule spans the whole run; do not pass --lr-decay-iters")

    hidden, heads = args.hidden_size, args.num_attention_heads
    if hidden % heads:
        errors.append("hidden size must be a multiple of the number of attention heads")
    else:
        head_dim = hidden // heads
        if not 8 <= head_dim <= 256 or head_dim % 8:
            errors.append(f"head width {head_dim} must be a multiple of 8 between 8 and 256")
        if args.kv_channels not in (None, head_dim):
            errors.append("do not pass --kv-channels; the head width is hidden size / attention heads")
    if getattr(args, "padded_vocab_size", None) not in (None, 32064):
        errors.append("the vocabulary is padded to 32,064 entries")
    if args.prelude_layers < 0 or args.coda_layers < 0 or args.recurrent_layers < 1:
        errors.append("need at least one recurrent layer and nonnegative prelude and coda layer counts")
    if args.backprop_loops is None:
        args.backprop_loops = args.loops
    if args.loops < 1 or not 1 <= args.backprop_loops <= args.loops:
        errors.append("need 1 <= --backprop-loops <= --loops")
    if args.untie_loops and (args.loops < 2 or args.backprop_loops != args.loops):
        errors.append("--untie-loops needs at least two loops, all backpropagated (B = T)")
    if not args.train_iters or args.train_iters < 1:
        errors.append("--train-iters must be positive")
    if errors:
        raise ValueError("invalid configuration:\n  " + "\n  ".join(errors))
    validate_online_data_args(args)
    validate_evaluation_args(args)
    validate_tokenizer(args.tokenizer_model)


def resolve_recipe(args: Any) -> None:
    """Peak learning rate of every parameter role (HyperP transfer to the run's width, tokens and batch)."""
    schedule(args)
    batch_tokens = args.global_batch_size * args.seq_length
    transfer = LearningRateTransfer(args.hidden_size, args.train_iters * batch_tokens, batch_tokens)
    args.base_lrs = base_lrs(args.lr_multiplier)
    args.peak_lrs = transfer.peak_lrs(args.base_lrs)
    args.lr, args.min_lr = args.peak_lrs["matrix"], FINAL_LR_FRACTION * args.peak_lrs["matrix"]


def recipe_metadata(args: Any) -> dict[str, Any]:
    from dataclasses import asdict

    batch_tokens = args.global_batch_size * args.seq_length
    return {
        "lr_multiplier": args.lr_multiplier,
        "base_lrs": args.base_lrs,
        "peak_lrs": args.peak_lrs,
        "width": args.hidden_size,
        "total_tokens": args.train_iters * batch_tokens,
        "batch_tokens": batch_tokens,
        "schedule": asdict(schedule(args)),
        "model": {
            "hidden_size": args.hidden_size,
            "num_attention_heads": args.num_attention_heads,
            "layers": [args.prelude_layers, args.recurrent_layers, args.coda_layers],
            "loops": args.loops,
            "backprop_loops": args.backprop_loops,
            "untie_loops": args.untie_loops,
        },
    }
