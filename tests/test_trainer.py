"""Trainer options, recipe resolution, the schedule driver and the loss contract (no Megatron checkout needed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from looptf.megatron import arguments
from looptf.megatron.arguments import parser_defaults, resolve_recipe, validate_args
from looptf.megatron.optimizer import RecipeScheduler
from looptf.megatron.train import loss_func
from looptf.recipe import LearningRateTransfer, Schedule, base_lrs

ARGV = ["--hidden-size", "768", "--num-attention-heads", "6", "--prelude-layers", "2", "--recurrent-layers", "4",
        "--coda-layers", "2", "--train-iters", "20", "--lr-multiplier", "0.9170", "--lr-schedule", "cosine",
        "--lr-warmup-iters", "2"]


def parsed(defaults, **overrides):
    """What Megatron's parser returns for ARGV with the trainer's defaults."""
    values = dict(
        defaults, tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1,
        expert_model_parallel_size=1, hidden_size=768, num_attention_heads=6, kv_channels=128, prelude_layers=2,
        recurrent_layers=4, coda_layers=2, loops=4, backprop_loops=2, untie_loops=False, train_iters=20,
        lr_warmup_iters=2, lr_schedule="cosine", lr_multiplier="0.9170", global_batch_size=16,
        online_validation_fraction=0.001, online_tokenizer_batch_size=256, online_shuffle_buffer_records=128,
        save=None, load=None, dataloader_save=None, eval_iters=2, final_eval_iters=4, eval_interval=10,
        tokenizer_model="unused",
    )
    return SimpleNamespace(**(values | overrides))


@pytest.fixture(autouse=True)
def no_tokenizer_check(monkeypatch):
    monkeypatch.setattr(arguments, "validate_tokenizer", lambda path: None)


def test_derived_megatron_options():
    defaults = parser_defaults(ARGV)
    assert (defaults["num_layers"], defaults["ffn_hidden_size"], defaults["num_query_groups"]) == (8, 3072, 6)
    assert defaults["lr"] == base_lrs("0.9170")["matrix"]
    assert (defaults["lr_decay_style"], defaults["lr_wsd_decay_style"], defaults["lr_wsd_decay_iters"]) == (
        "WSD", "cosine", 18)
    linear = parser_defaults(ARGV[:-4])
    assert (linear["lr_decay_style"], linear["lr_wsd_decay_style"], linear["lr_wsd_decay_iters"]) == (
        "linear", "linear", 0)
    assert defaults["optimizer"] == "muonh" and defaults["transformer_impl"] == "local"


def test_valid_configuration_and_recipe_resolution():
    defaults = parser_defaults(ARGV)
    args = parsed(defaults)
    validate_args(args, defaults)
    resolve_recipe(args)
    peaks = LearningRateTransfer(768, 20 * 16 * 2048, 16 * 2048).peak_lrs(base_lrs("0.9170"))
    assert args.peak_lrs == peaks
    assert (args.lr, args.min_lr) == (peaks["matrix"], 0.1 * peaks["matrix"])


@pytest.mark.parametrize("overrides,message", [
    ({"normalization": "LayerNorm"}, "set by the trainer"),
    ({"num_layers": 9}, "set by the trainer"),
    ({"lr": 0.01}, "set by the trainer"),
    ({"backprop_loops": 5}, "backprop-loops"),
    ({"untie_loops": True}, "untie-loops"),
    ({"hidden_size": 780, "ffn_hidden_size": 3072}, "head width"),
    ({"swiglu": True}, "not supported"),
    ({"tensor_model_parallel_size": 2}, "must be 1"),
])
def test_invalid_configurations_are_rejected(overrides, message):
    defaults = parser_defaults(ARGV)
    with pytest.raises(ValueError, match=message):
        validate_args(parsed(defaults, **overrides), defaults)


def test_scheduler_sets_the_learning_rate_after_every_update_and_resumes():
    groups = [{"params": [torch.nn.Parameter(torch.ones(1))], "_base_lr": 0.04, "lr": 0.04},
              {"params": [torch.nn.Parameter(torch.ones(1))], "_base_lr": 0.002, "lr": 0.002}]
    optimizer = torch.optim.SGD(groups, lr=0.1)
    plan = Schedule.named("cosine", 20, 2)
    scheduler = RecipeScheduler(optimizer, plan, train_iters=20, global_batch_size=16)
    assert optimizer.param_groups[0]["lr"] == 0.04 * plan.lr_scale(0, 20)
    for step in range(1, 8):
        scheduler.step(increment=16)
        assert optimizer.param_groups[1]["lr"] == 0.002 * plan.lr_scale(step, 20)
    state = scheduler.state_dict()
    resumed = RecipeScheduler(optimizer, plan, train_iters=20, global_batch_size=16)
    resumed.load_state_dict(state)
    assert resumed.num_steps == 7 * 16 and optimizer.param_groups[0]["lr"] == 0.04 * plan.lr_scale(7, 20)
    with pytest.raises(ValueError, match="differs"):
        RecipeScheduler(optimizer, plan, train_iters=40, global_batch_size=16).load_state_dict(state)


def test_loss_contract():
    loss_per_token = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    loss, tokens, report = loss_func(torch.tensor([[1.0, 0.0], [1.0, 1.0]]), loss_per_token)
    assert loss.item() == 8.0 and tokens.item() == 3
    assert torch.equal(report["lm loss"], torch.tensor([8.0, 3.0]))
    loss.backward()
    assert torch.equal(loss_per_token.grad, torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
