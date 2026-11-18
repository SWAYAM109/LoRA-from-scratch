#!/usr/bin/env python3
"""
Lesson 8 - QLoRA: Quantize the Frozen Base

The final trick: crush the frozen base to 4 bits per weight.
The adapter stays in full precision where learning happens.
Innovation: Side-by-side comparison of Absmax vs NF4 quantization.
"""

import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_scratch.injection import add_lora
from lora_scratch.merge import load_adapter
from lora_scratch.data import setup_tokenizer
from lora_scratch.quantization import quantize_model_weights


def build_tagger(tok, top_labels, adapter_path):
    """Build a fresh model with LoRA adapter loaded."""
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tok))
    add_lora(model)
    load_adapter(model, adapter_path, n_labels=len(top_labels))
    return model.eval()


def evaluate(model, tok, val, label_map, label_ids, trues):
    """Run evaluation and return accuracy."""
    @torch.no_grad()
    def classify(text):
        logits = model(**tok(text.strip() + " ", return_tensors="pt")).logits[0, -1]
        return label_map[max(label_ids, key=lambda t: logits[t].item())]

    preds = [classify(e["text"]) for e in val]
    return sum(p == t for p, t in zip(preds, trues)) / len(trues)


def main():
    print("=" * 70)
    print("LESSON 8: QLoRA -- Quantize the Frozen Base")
    print("=" * 70)
    print()

    # Setup
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    names = ds["train"].features["label"].names
    top_labels = [l for l, _ in Counter(ds["train"]["label"]).most_common(6)]
    top_set = set(top_labels)

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    label_map = setup_tokenizer(tok, names, top_labels)
    label_ids = list(label_map.keys())

    val = ds["validation"].filter(lambda e: e["label"] in top_set).shuffle(seed=2).select(range(120))
    trues = [names[e["label"]] for e in val]

    adapter_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "lora-training", "lora-adapter.pt")

    # 1. Baseline: fp32
    print("1. fp32 frozen base + adapter...")
    model_fp32 = build_tagger(tok, top_labels, adapter_path)
    acc_fp32 = evaluate(model_fp32, tok, val, label_map, label_ids, trues)
    print(f"   Accuracy: {acc_fp32:.1%}")
    del model_fp32

    # 2. Absmax 4-bit quantization
    print("\n2. 4-bit ABSMAX quantization + adapter...")
    model_absmax = build_tagger(tok, top_labels, adapter_path)
    n_q, err = quantize_model_weights(model_absmax, scheme="absmax")
    acc_absmax = evaluate(model_absmax, tok, val, label_map, label_ids, trues)
    print(f"   Quantized {n_q:,} weights | Mean relative error: {100*err:.1f}%")
    print(f"   Accuracy: {acc_absmax:.1%}")
    del model_absmax

    # 3. NF4 quantization (Innovation)
    print("\n3. 4-bit NF4 quantization + adapter (INNOVATION)...")
    model_nf4 = build_tagger(tok, top_labels, adapter_path)
    n_q, err_nf4 = quantize_model_weights(model_nf4, scheme="nf4")
    acc_nf4 = evaluate(model_nf4, tok, val, label_map, label_ids, trues)
    print(f"   Quantized {n_q:,} weights | Mean relative error: {100*err_nf4:.1f}%")
    print(f"   Accuracy: {acc_nf4:.1%}")
    del model_nf4

    # 4. Summary table
    print()
    print("-" * 65)
    print(f"{'Method':<30s} {'Accuracy':>10s} {'Quant Error':>14s}")
    print("-" * 65)
    print(f"{'fp32 base + adapter':<30s} {acc_fp32:>9.1%} {'N/A':>14s}")
    print(f"{'4-bit absmax + adapter':<30s} {acc_absmax:>9.1%} {f'{100*err:.1f}%':>14s}")
    print(f"{'4-bit NF4 + adapter':<30s} {acc_nf4:>9.1%} {f'{100*err_nf4:.1f}%':>14s}")
    print("-" * 65)

    # 5. Memory math for large models
    print()
    print("Memory to FINE-TUNE a 70B model:")
    print("-" * 55)
    P = 70e9
    methods = [
        ("Full fine-tune (fp32)", P * 16 / 1e9),
        ("LoRA (fp16 frozen base)", (P * 2 + P * 0.0018 * 16) / 1e9),
        ("QLoRA (4-bit frozen base)", (P * 0.5 + P * 0.0018 * 16) / 1e9),
        ("Single 80GB GPU", 80),
    ]
    for name, gb in methods:
        marker = " <-- fits!" if gb <= 80 and "GPU" not in name else ""
        print(f"  {name:<28s} {gb:>7,.0f} GB{marker}")

    print()
    print("+" + "-" * 63 + "+")
    print("|  KEY INSIGHT: Quantization damages the base.                |")
    print("|  The adapter heals it -- but only if it can SEE the         |")
    print("|  damage while it learns.                                    |")
    print("|                                                             |")
    print("|  Real QLoRA: quantize FIRST, then train the adapter.        |")
    print("|  NF4 narrows the gap by placing quantization levels         |")
    print("|  where neural network weights actually live (normal dist).  |")
    print("+" + "-" * 63 + "+")


if __name__ == "__main__":
    main()
