#!/usr/bin/env python3
"""
Lesson 3 - Build LoRALinear from Scratch

The core adapter module: a frozen nn.Linear + a trainable B @ A detour.
Innovation: Multiple initialization strategies with comparison.
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lora_scratch.core import LoRALinear


def main():
    print("=" * 70)
    print("LESSON 3: Build LoRALinear from Scratch")
    print("=" * 70)
    print()

    torch.manual_seed(42)
    base = nn.Linear(768, 2304)
    x = torch.randn(4, 768)

    print("Testing LoRALinear with different init strategies:")
    print("-" * 55)

    for strategy in ["kaiming", "gaussian", "xavier"]:
        torch.manual_seed(42)
        base_copy = nn.Linear(768, 2304)
        base_copy.load_state_dict(base.state_dict())

        lora = LoRALinear(base_copy, r=8, alpha=16, init_strategy=strategy)

        with torch.no_grad():
            is_noop = torch.allclose(lora(x), base(x))

        a_norm = lora.A.data.norm().item()
        trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in lora.parameters() if not p.requires_grad)

        print(f"\n  Strategy: {strategy}")
        print(f"    At init, LoRA(x) == base(x): {is_noop}  (B=0 -> no-op)")
        print(f"    ||A|| = {a_norm:.4f}")
        print(f"    Trainable: {trainable:,}  |  Frozen: {frozen:,}  |  "
              f"{100*trainable/(trainable+frozen):.1f}% trainable")

    print()
    print("-" * 55)
    print("Demonstrating the detour activation:")

    torch.manual_seed(42)
    base_demo = nn.Linear(768, 2304)
    lora_demo = LoRALinear(base_demo, r=8, alpha=16)

    with torch.no_grad():
        before = lora_demo(x)
        lora_demo.B.data += 0.1 * torch.randn_like(lora_demo.B)
        after = lora_demo(x)

    diff = (after - before).abs().mean().item()
    print(f"  Before nudging B: output matches base")
    print(f"  After nudging B:  mean output difference = {diff:.6f}")
    print(f"  Detour is now LIVE")

    print()
    print("-" * 55)
    print("Effective weight update (B @ A) * scale:")
    delta = lora_demo.effective_delta_w()
    print(f"  Shape: {tuple(delta.shape)}")
    print(f"  Frobenius norm: {delta.norm():.4f}")
    print(f"  Rank: {torch.linalg.matrix_rank(delta).item()}")

    print()
    print("+" + "-" * 59 + "+")
    print("|  LoRALinear: 10 lines of code, 3 key decisions:         |")
    print("|    1. Freeze the base (never trained)                   |")
    print("|    2. A random, B zero (safe no-op start)               |")
    print("|    3. Scale by alpha/r (rank-independent strength)      |")
    print("+" + "-" * 59 + "+")


if __name__ == "__main__":
    main()
