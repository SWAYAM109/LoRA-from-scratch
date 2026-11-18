#!/usr/bin/env python3
"""
Lesson 4 - Inject LoRA into GPT-2

Freeze all 82M parameters, wrap the attention Q/K/V projections
in every block, and verify: 0.18% trainable, output unchanged.
Innovation: Multi-target injection -- also target c_proj for comparison.
"""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoModelForCausalLM
from lora_scratch.injection import add_lora, TARGET_PRESETS
from lora_scratch.analysis import count_parameters


def main():
    print("=" * 70)
    print("LESSON 4: Inject LoRA into GPT-2")
    print("=" * 70)
    print()

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    c_attn = model.transformer.h[0].attn.c_attn
    print(f"c_attn type: {type(c_attn).__name__}")
    print(f"c_attn weight shape: {tuple(c_attn.weight.shape)}  (Conv1D: [d_in, d_out])")
    print()

    ids = torch.tensor([[15496, 11, 995]])  # "Hello, world"
    with torch.no_grad():
        before = model(ids).logits

    print("Comparing injection targets:")
    print("-" * 65)

    for preset_name in TARGET_PRESETS:
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_pretrained("distilgpt2")

        with torch.no_grad():
            ref = m(ids).logits

        add_lora(m, r=8, alpha=16, targets=preset_name)

        with torch.no_grad():
            after = m(ids).logits

        identical = torch.allclose(ref, after, atol=1e-6)
        stats = count_parameters(m)

        targets = TARGET_PRESETS[preset_name]
        print(f"\n  Preset: '{preset_name}'")
        print(f"  Targets: {targets}")
        print(f"  Output unchanged at injection: {identical}")
        print(f"  Trainable: {stats['trainable']:>10,} / {stats['total']:,}")
        print(f"  -> {stats['pct_trainable']:.2f}% of model trains "
              f"({stats['total'] // stats['trainable']}x fewer weights)")

    print()
    print("-" * 65)
    print("Trainable parameters (attention_qkv preset):")
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained("distilgpt2")
    add_lora(m, r=8, alpha=16)
    for n, p in m.named_parameters():
        if p.requires_grad:
            print(f"  {n:45s} {str(tuple(p.shape)):>15s}")

    print()
    print("+" + "-" * 63 + "+")
    print("|  The frozen 82M still run on every forward pass.              |")
    print("|  LoRA doesn't make the model faster -- it removes gradient    |")
    print("|  and optimizer state for 99.8% of the model.                  |")
    print("+" + "-" * 63 + "+")


if __name__ == "__main__":
    main()
