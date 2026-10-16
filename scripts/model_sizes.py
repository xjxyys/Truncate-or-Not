#!/usr/bin/env python3
"""Print the model sizes, the training FLOPs per token, the optimizer's share of the compute and the per-budget
training settings of the scaling grid (definitions in src/looptf/scaling.py)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from looptf import scaling  # noqa: E402
from looptf.model.config import PADDED_VOCAB_SIZE  # noqa: E402

FLOPS_SETTINGS = ((1, 1), (2, 2), (4, 1), (4, 4), (8, 2), (8, 8))
OPTIMIZER_SETTINGS = ((1, 1), (2, 2), (4, 4), (8, 8))


def main():
    print("| size | width | heads | layers P/R/C | N | N_once | N_rec | N_rec/N | total incl. embeddings |")
    print("|---|---|---|---|---|---|---|---|---|")
    for size, (d, p, r, c) in scaling.SIZES.items():
        once, rec = scaling.parameter_counts(d, p, r, c)
        n = once + rec
        label = size + ("" if size in scaling.MAIN_SIZES else " (pair)")
        print(f"| {label} | {d} | {d // 128} | {p}/{r}/{c} | {n / 1e6:.1f}M | {once / 1e6:.1f}M | {rec / 1e6:.1f}M | "
              f"{rec / n:.3f} | {(n + 2 * PADDED_VOCAB_SIZE * d) / 1e6:.1f}M |")

    print("\nTraining FLOPs per token (forward + backward), in units of 10^9:")
    print("| size | " + " | ".join(f"({t},{b})" for t, b in FLOPS_SETTINGS) + " |")
    print("|---|" + "---|" * len(FLOPS_SETTINGS))
    for size in scaling.MAIN_SIZES:
        d, p, r, c = scaling.SIZES[size]
        row = [scaling.flops_per_token(d, p + c, r, t, b) / 1e9 for t, b in FLOPS_SETTINGS]
        print(f"| {size} | " + " | ".join(f"{x:.3f}" for x in row) + " |")

    print("\nOptimizer compute as a share of the forward + backward compute, with the batch rule 64 sqrt(D / D0):")
    print("| size | D / D0 | " + " | ".join(f"({t},{b})" for t, b in OPTIMIZER_SETTINGS) + " |")
    print("|---|---|" + "---|" * len(OPTIMIZER_SETTINGS))
    for size in scaling.MAIN_SIZES:
        d, p, r, c = scaling.SIZES[size]
        for budget in (1, 4, 16, 32):
            tokens = budget * scaling.D0
            steps = tokens / (scaling.SEQUENCE_LENGTH * 64 * math.sqrt(budget))
            row = [scaling.optimizer_flops_per_step(d, p + c, r) * steps / (scaling.flops_per_token(d, p + c, r, t, b)
                                                                             * tokens) for t, b in OPTIMIZER_SETTINGS]
            print(f"| {size} | {budget} | " + " | ".join(f"{x:.2%}" for x in row) + " |")

    print("\nTraining settings per token budget:")
    print("| D / D0 | batch | steps | warmup | eval interval | eval batch | eval iters | final eval iters "
          "| LR multiplier |")
    print("|---|---|---|---|---|---|---|---|---|")
    for budget in scaling.BUDGETS:
        batch = scaling.global_batch(budget)
        steps = scaling.train_iters(budget, batch)
        ev = scaling.evaluation(steps, batch)
        print(f"| {budget:g} | {batch} | {steps} | {scaling.warmup_iters(steps)} | {ev.interval} | "
              f"{ev.global_batch_size} | {ev.iters} | {ev.final_iters} | {scaling.lr_multiplier(budget)} |")


if __name__ == "__main__":
    main()
