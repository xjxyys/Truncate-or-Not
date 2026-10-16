"""Learning-rate transfer, schedules and the scaling-suite settings."""

from __future__ import annotations

import math

import pytest

from looptf import scaling
from looptf.model import parameter_counts
from looptf.recipe import (
    BASE_LRS,
    LR_MULTIPLIERS,
    REFERENCE_BATCH_TOKENS,
    REFERENCE_TOKENS,
    LearningRateTransfer,
    Schedule,
    base_lrs,
)


def test_reference_point_keeps_the_base_learning_rates():
    peaks = LearningRateTransfer(640, REFERENCE_TOKENS, REFERENCE_BATCH_TOKENS).peak_lrs()
    assert peaks == pytest.approx(BASE_LRS, rel=1e-12)
    # 10 tokens per non-embedding parameter of the non-looped 20-layer model at width 640.
    assert REFERENCE_TOKENS == 10 * parameter_counts(640, 20)["non_embedding"]


def test_transfer_exponents():
    base = LearningRateTransfer(640, REFERENCE_TOKENS).peak_lrs()
    wide = LearningRateTransfer(2560, REFERENCE_TOKENS).peak_lrs()
    long = LearningRateTransfer(640, 4 * REFERENCE_TOKENS).peak_lrs()
    big_batch = LearningRateTransfer(640, REFERENCE_TOKENS, 4 * REFERENCE_BATCH_TOKENS).peak_lrs()
    assert wide["matrix"] == base["matrix"] and wide["embedding"] == pytest.approx(base["embedding"] / 2)
    assert long["matrix"] == pytest.approx(base["matrix"] * 4**-0.32) and long["norm"] == base["norm"]
    assert big_batch["unembedding"] == pytest.approx(base["unembedding"] * 4**0.558)


def test_learning_rate_multipliers_are_powers_of_two_to_the_eighth():
    for label, rates in LR_MULTIPLIERS.items():
        k = round(8 * math.log2(float(label)))
        assert float(label) == pytest.approx(2 ** (k / 8), abs=5e-5)
        for role, rate in rates.items():
            # The table holds the values rounded to 10 decimal places.
            assert rate == pytest.approx(BASE_LRS[role] * 2 ** (k / 8), rel=0, abs=5e-11), (label, role)
    with pytest.raises(ValueError, match="unsupported"):
        base_lrs("3")


def test_linear_schedule_decays_from_the_peak_to_a_tenth():
    schedule = Schedule.named("linear", 100)
    assert schedule.lr_scale(0, 100) == 1.0
    assert schedule.lr_scale(99, 100) == pytest.approx(0.1)
    assert schedule.cooldown_iters(100) == 0
    with pytest.raises(ValueError, match="no warmup"):
        Schedule.named("linear", 100, 5)


@pytest.mark.parametrize("steps,warmup", [(20, 2), (16384, 328), (11920, 238), (2736, 55)])
def test_cosine_schedule_starts_its_decay_at_the_end_of_warmup(steps, warmup):
    schedule = Schedule.named("cosine", steps, warmup)
    scales = [schedule.lr_scale(step, steps) for step in range(steps)]
    assert scales[:warmup] == pytest.approx([(i + 1) / warmup for i in range(warmup)])
    assert scales[warmup] == 1.0 and scales[-1] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(scales[warmup:], scales[warmup + 1:]))
    assert steps - schedule.cooldown_iters(steps) == warmup


def test_suite_settings_of_the_readme_example():
    batch = scaling.global_batch(4)
    steps = scaling.train_iters(4, batch)
    evaluation = scaling.evaluation(steps, batch)
    assert (batch, steps, scaling.warmup_iters(steps), scaling.lr_multiplier(4)) == (128, 16384, 328, "0.9170")
    assert evaluation == scaling.Evaluation(interval=2048, global_batch_size=128, iters=32, final_iters=540)


def test_suite_settings_for_batches_that_do_not_divide_the_evaluation_window():
    batch = scaling.global_batch(2)
    assert batch == 88
    evaluation = scaling.evaluation(scaling.train_iters(2, batch), batch)
    assert (evaluation.global_batch_size, evaluation.iters, evaluation.final_iters) == (64, 64, 1080)
    assert scaling.lr_multiplier(1 / 16) == "1.1892" and scaling.lr_multiplier(1) == "1"
    assert [scaling.global_batch(b) for b in scaling.BUDGETS] == list(scaling.BATCH_LADDER)
