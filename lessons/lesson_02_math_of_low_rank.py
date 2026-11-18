#!/usr/bin/env python3
"""
Lesson 2 - The Math of Low-Rank

The core insight: a weight update dW can be approximated as B @ A where
A in R^{r x d} and B in R^{d x r}. For small r, this captures most of the
update at a fraction of the parameters.
"""

import numpy as np


def analyze_low_rank_approximation(d=768, true_rank=6, noise_std=0.02, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((d, true_rank)) @ rng.standard_normal((true_rank, d))
    dW = signal + noise_std * rng.standard_normal((d, d))
    sv = np.linalg.svd(dW, compute_uv=False)
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    return sv, energy


def main():
    print("=" * 70)
    print("LESSON 2: The Math of Low-Rank Adaptation")
    print("=" * 70)
    print()
    print("dW = B @ A")
    print()
    print("Instead of storing a full d x d update matrix (d^2 parameters),")
    print("LoRA decomposes it as B @ A:")
    print("  - A in R^{r x d}  -- squeeze to rank r")
    print("  - B in R^{d x r}  -- expand back to d")
    print("  - Cost: 2*d*r parameters (vs d^2 for the full matrix)")
    print()

    d = 768
    sv, energy = analyze_low_rank_approximation(d)

    print(f"A {d}x{d} update = {d*d:,} numbers if stored in full")
    print()
    print(f"{'Rank r':>8s} {'Energy Captured':>18s} {'LoRA params':>14s} {'Compression':>14s}")
    print("-" * 58)

    for r in [1, 2, 4, 8, 16, 32, 64, d]:
        params = 2 * d * r if r < d else d * d
        compress = (d * d) / params
        tag = " <- full dW" if r == d else ""
        print(f"{r:>8d} {energy[r-1]*100:>16.1f}% {params:>14,d} {compress:>12.1f}x{tag}")

    print()
    print("+" + "-" * 59 + "+")
    print("|  At rank 8: ~100% of a realistic update captured,       |")
    print(f"|  for {2*d*8:,} params -- {(d*d)/(2*d*8):.0f}x smaller than {d*d:,}.      |")
    print("|                                                          |")
    print("|  Rank r is the ONE dial between capacity and cost.       |")
    print("|  LoRA scales output by alpha/r to decouple rank from     |")
    print("|  adapter strength.                                       |")
    print("+" + "-" * 59 + "+")

    print()
    print("Singular value spectrum (top 20):")
    print("-" * 40)
    max_sv = sv[0]
    for i in range(min(20, len(sv))):
        bar_len = int(50 * sv[i] / max_sv)
        print(f"  sv_{i+1:>2d} = {sv[i]:>10.2f}  {'#' * bar_len}")

    print()
    print("-> Sharp drop after rank 6 confirms the low-rank hypothesis.")


if __name__ == "__main__":
    main()
