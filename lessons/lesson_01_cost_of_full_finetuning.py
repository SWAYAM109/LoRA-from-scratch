#!/usr/bin/env python3
"""
Lesson 1 - Why Not Just Fine-Tune Everything?

Shows the memory wall: full fine-tuning in fp32 costs ~16 bytes per parameter
(weight + gradient + 2 Adam states). At 7B+ this exceeds single-GPU memory.
LoRA's bet: freeze the base, train a sliver, fit on one GPU.
"""


def full_ft_memory_gb(n_params):
    """fp32 training: weight + gradient + Adam mean + Adam variance = 16 bytes/param."""
    return n_params * 16 / 1e9


def lora_train_memory_gb(n_params, trainable_frac=0.0018):
    """LoRA: frozen base in fp16 (2B/param) + adapter trains in fp32 (16B/param)."""
    base = n_params * 2 / 1e9
    adapter = n_params * trainable_frac * 16 / 1e9
    return base + adapter


def main():
    print("=" * 70)
    print("LESSON 1: The Cost of Full Fine-Tuning")
    print("=" * 70)
    print()
    print("When you fine-tune, the optimizer keeps for EVERY parameter:")
    print("  - the weight itself          (4 bytes in fp32)")
    print("  - the gradient               (4 bytes)")
    print("  - Adam running mean          (4 bytes)")
    print("  - Adam running variance      (4 bytes)")
    print("  " + "-" * 41)
    print("  Total: 16 bytes per parameter")
    print()

    models = [
        ("distilgpt2", 82e6),
        ("GPT-2 Medium", 345e6),
        ("Llama-2-7B", 7e9),
        ("Llama-2-13B", 13e9),
        ("Llama-2-70B", 70e9),
    ]

    print(f"{'Model':<18s} {'Full FT (fp32)':>16s} {'LoRA':>12s} {'Savings':>10s}")
    print("-" * 60)
    for name, n in models:
        full = full_ft_memory_gb(n)
        lora = lora_train_memory_gb(n)
        savings = full / lora
        print(f"{name:<18s} {full:>13,.1f} GB {lora:>9,.1f} GB {savings:>8,.1f}x")

    print()
    print("+" + "-" * 59 + "+")
    print("|  A single high-end GPU (A100/H100) has 80 GB of VRAM.   |")
    print("|  Full FT of 7B in fp32 (~112 GB) does NOT fit.          |")
    print("|  LoRA (~14 GB) fits comfortably.                        |")
    print("|  And each new task is a few MB adapter, not 112 GB.     |")
    print("+" + "-" * 59 + "+")


if __name__ == "__main__":
    main()
