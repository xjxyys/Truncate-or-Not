"""Validation around Megatron's evaluator: a separate budget for the final evaluation, an optional evaluation at
step 0, and one JSON line per evaluation in <run>/evaluation_metrics.jsonl (also logged to TensorBoard / W&B).

Periodic evaluations read --eval-iters global batches of held-out sequences and the final one --final-eval-iters;
all of them continue the same held-out iterator.
"""

from __future__ import annotations

import inspect
import json
import math
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any


def add_evaluation_arguments(parser: Any) -> None:
    group = parser.add_argument_group("evaluation")
    group.add_argument("--eval-at-start", action="store_true",
                       help="Evaluate at step 0 before training a fresh run (skipped on resume).")
    group.add_argument("--final-eval-iters", type=int, default=None,
                       help="Global batches of the final evaluation (default: --eval-iters).")


def validate_evaluation_args(args: Any) -> None:
    if not isinstance(args.eval_iters, int) or args.eval_iters < 1:
        raise ValueError("--eval-iters must be positive")
    if args.final_eval_iters is not None and args.final_eval_iters < 1:
        raise ValueError("--final-eval-iters must be positive")
    if not args.eval_interval or args.eval_interval < 1:
        raise ValueError("--eval-interval must be positive; use a value above --train-iters to evaluate only at "
                         "the end")
    for name in ("full_validation", "multiple_validation_sets", "perform_rl_step", "skip_train"):
        if getattr(args, name, False):
            raise ValueError(f"--{name.replace('_', '-')} is not supported")


def evaluation_metadata(args: Any) -> dict[str, Any]:
    batch = args.eval_global_batch_size
    final_iters = args.final_eval_iters or args.eval_iters
    return {
        "eval_at_start": args.eval_at_start,
        "interval": args.eval_interval,
        "iters": args.eval_iters,
        "final_iters": final_iters,
        "global_batch_size": batch,
        "sequence_length": args.seq_length,
        "sequences": args.eval_iters * batch,
        "final_sequences": final_iters * batch,
    }


@dataclass
class _Evaluation:
    step: int
    is_final: bool
    write_to_tensorboard: bool
    counts: Any = None
    loops: int | None = None


_current: ContextVar[_Evaluation | None] = ContextVar("looptf_evaluation", default=None)


def record_batch(tokens: Any, loss_mask: Any, loops: int) -> None:
    """Count the sequences and tokens of one evaluation micro-batch (no-op outside evaluation)."""
    event = _current.get()
    if event is None:
        return
    import torch

    if event.loops not in (None, loops):
        raise RuntimeError("the loop count changed within one evaluation")
    event.loops = loops
    mask = loss_mask.detach()
    counts = torch.stack((mask.new_tensor(tokens.shape[0], dtype=torch.int64), mask.sum().to(torch.int64)))
    event.counts = counts if event.counts is None else event.counts + counts


def _payload(losses: dict[str, Any], event: _Evaluation) -> dict[str, Any]:
    import torch

    if event.counts is None:
        raise RuntimeError("evaluation finished without any recorded batch")
    counts = event.counts.clone()
    if torch.distributed.is_initialized():
        from megatron.core import mpu

        torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    sequences, tokens = counts.tolist()
    prefix = "eval/final_val" if event.is_final else "eval/val"
    loss = float(losses["lm loss"])
    try:
        ppl = math.exp(loss)
    except OverflowError:
        ppl = math.inf
    return {
        f"{prefix}/loss": loss,
        f"{prefix}/ppl": ppl,
        f"{prefix}/loops": event.loops,
        "eval/checkpoint_step": event.step,
        "eval/is_final": event.is_final,
        f"{prefix}/num_sequences": sequences,
        f"{prefix}/num_tokens": tokens,
    }


def install_evaluation_hooks(args: Any):
    """Wrap Megatron's evaluation functions; returns a function that restores them."""
    from megatron.training import training

    original = {name: getattr(training, name) for name in ("evaluate", "evaluate_and_print_results", "train")}
    report_signature = inspect.signature(original["evaluate_and_print_results"])
    train_signature = inspect.signature(original["train"])
    configured_writers = set()

    @wraps(original["evaluate"])
    def evaluate(*positional, **keywords):
        event = _current.get()
        if event is not None:
            event.counts, event.loops = None, None
        result = original["evaluate"](*positional, **keywords)
        losses, _, timelimit = result
        if event is None or timelimit:
            return result
        payload = _payload(losses, event)
        if args.save and training.is_last_rank():
            receipt = Path(args.save).parent / "evaluation_metrics.jsonl"
            receipt.parent.mkdir(parents=True, exist_ok=True)
            with receipt.open("a") as handle:
                handle.write(json.dumps(payload) + "\n")
        writer = training.get_tensorboard_writer()
        if writer is not None and event.write_to_tensorboard:
            for name, value in payload.items():
                writer.add_scalar(name, value, event.step)
        writer = training.get_wandb_writer()
        if writer is not None and training.is_last_rank():
            if id(writer) not in configured_writers:
                writer.define_metric("eval/checkpoint_step", hidden=True)
                writer.define_metric("eval/*", step_metric="eval/checkpoint_step", step_sync=False)
                configured_writers.add(id(writer))
            writer.log(payload, step=event.step)
        return result

    @wraps(original["evaluate_and_print_results"])
    def evaluate_and_print_results(*positional, **keywords):
        call = report_signature.bind(*positional, **keywords)
        call.apply_defaults()
        # Megatron labels its final evaluation "iteration N on validation set", periodic ones "iteration N".
        is_final = call.arguments["prefix"].endswith(" on validation set")
        event = _Evaluation(int(call.arguments["iteration"]), is_final, call.arguments["write_to_tensorboard"])
        token = _current.set(event)
        periodic_iters = args.eval_iters
        if is_final and args.final_eval_iters is not None:
            args.eval_iters = args.final_eval_iters
            training.print_rank_0(f"final evaluation: {args.eval_iters} global batches")
        try:
            return original["evaluate_and_print_results"](*positional, **keywords)
        finally:
            args.eval_iters = periodic_iters
            _current.reset(token)

    @wraps(original["train"])
    def train(*positional, **keywords):
        call = train_signature.bind(*positional, **keywords)
        call.apply_defaults()
        if args.iteration == 0 and args.do_valid:
            values = call.arguments
            training.print_rank_0("evaluation at step 0, before the first update")
            training.evaluate_and_print_results(
                "iteration 0", values["forward_step_func"], values["valid_data_iterator"], values["model"], 0,
                values["process_non_loss_data_func"], values["config"], verbose=True, write_to_tensorboard=True,
                non_loss_data_func=values.get("non_loss_data_func"), pg_collection=values.get("pg_collection"),
                p2p_communicator=values.get("p2p_communicator"),
            )
        return original["train"](*positional, **keywords)

    training.evaluate = evaluate
    training.evaluate_and_print_results = evaluate_and_print_results
    if args.eval_at_start:
        training.train = wraps(original["train"])(train)

    def restore() -> None:
        for name, value in original.items():
            setattr(training, name, value)

    return restore
