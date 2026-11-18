#!/usr/bin/env python3
"""
Analyze a trained LoRA adapter: spectral analysis, rank ablation,
and dynamic rank pruning experiments.

Usage:
    python scripts/analyze.py --adapter outputs/lora-training/lora-adapter.pt
"""

import argparse
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from lora_scratch.injection import add_lora
from lora_scratch.merge import load_adapter
from lora_scratch.data import setup_tokenizer
from lora_scratch.analysis import (
    count_parameters,
    rank_ablation_study,
    adapter_spectral_analysis,
    training_memory_estimate,
)
from lora_scratch.core import LoRALinear


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LoRA adapter")
    parser.add_argument("--adapter", type=str, default=None, help="Path to adapter file")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("LoRA Adapter Analysis")
    print("=" * 70)

    # 1. Rank Ablation
    print("\n1. RANK ABLATION")
    print("-" * 65)
    results = rank_ablation_study(d=768, true_rank=6, noise_std=0.02)
    print(f"{'Rank':>6s} {'Energy':>10s} {'Params':>12s} {'Compression':>14s}")
    for r in results:
        print(f"{r['rank']:>6d} {r['energy_pct']:>9.1f}% {r['lora_params']:>12,d} {r['compression_ratio']:>12.1f}x")

    # 2. Memory Estimates
    print("\n2. MEMORY ESTIMATES")
    print("-" * 65)
    print(f"{'Model':<18s} {'Full FT':>10s} {'LoRA':>10s} {'QLoRA':>10s}")
    for name, n in [("distilgpt2", 82e6), ("Llama-7B", 7e9), ("Llama-70B", 70e9)]:
        full = training_memory_estimate(n, "full_ft")
        lora = training_memory_estimate(n, "lora")
        qlora = training_memory_estimate(n, "qlora")
        print(f"{name:<18s} {full:>8.1f}GB {lora:>8.1f}GB {qlora:>8.1f}GB")

    # 3. Spectral Analysis
    if args.adapter and os.path.exists(args.adapter):
        print(f"\n3. SPECTRAL ANALYSIS")
        print("-" * 65)

        ds = load_dataset("coastalcph/lex_glue", "ledgar")
        names = ds["train"].features["label"].names
        top_labels = [l for l, _ in Counter(ds["train"]["label"]).most_common(6)]

        tok = AutoTokenizer.from_pretrained("distilgpt2")
        setup_tokenizer(tok, names, top_labels)

        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        model.resize_token_embeddings(len(tok))
        add_lora(model)
        load_adapter(model, args.adapter, n_labels=len(top_labels))

        spectral = adapter_spectral_analysis(model)
        print(f"{'Layer':<45s} {'Eff. Rank':>10s} {'Max sv':>10s} {'Min sv':>10s}")
        for s in spectral:
            layer_short = s["layer"].split("transformer.")[-1] if "transformer" in s["layer"] else s["layer"]
            print(f"{layer_short:<45s} {s['effective_rank']:>10d} {s['max_sv']:>10.4f} {s['min_sv']:>10.4f}")

        # 4. Dynamic Rank Pruning
        print(f"\n4. DYNAMIC RANK PRUNING")
        print("-" * 65)
        orig_params = sum(p.numel() for m in model.modules() if isinstance(m, LoRALinear)
                         for p in [m.A, m.B])
        print(f"Original adapter: rank 8, {orig_params:,} parameters")

        for keep_rank in [6, 4, 2]:
            # Reload fresh for each pruning level
            model2 = AutoModelForCausalLM.from_pretrained("distilgpt2")
            model2.resize_token_embeddings(len(tok))
            add_lora(model2)
            load_adapter(model2, args.adapter, n_labels=len(top_labels))
            for m in model2.modules():
                if isinstance(m, LoRALinear):
                    m.prune_rank(keep_rank)
            pruned_params = sum(p.numel() for m in model2.modules() if isinstance(m, LoRALinear)
                               for p in [m.A, m.B])
            reduction = 100 * (1 - pruned_params / orig_params)
            print(f"Pruned to rank {keep_rank}: {pruned_params:,} params ({reduction:.0f}% reduction)")
    else:
        print("\n3. SPECTRAL ANALYSIS - Skipped (no --adapter provided)")
        print("   Run: python scripts/analyze.py --adapter outputs/lora-training/lora-adapter.pt")


if __name__ == "__main__":
    main()
