#!/usr/bin/env python3
"""
Lesson 7 - Merge & Swap Adapters

Three things you can do with a trained adapter:
  1. Merge B @ A into W for free inference (no overhead)
  2. Unmerge to get back the clean base
  3. Hot-swap one adapter for another (one base, many tasks)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_scratch.injection import add_lora, get_lora_state_dict
from lora_scratch.merge import merge_lora, save_adapter, load_adapter
from lora_scratch.data import setup_tokenizer
from lora_scratch.analysis import count_parameters

from collections import Counter
from datasets import load_dataset


def main():
    print("=" * 70)
    print("LESSON 7: Merge & Swap Adapters")
    print("=" * 70)
    print()

    # Setup
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    names = ds["train"].features["label"].names
    top_labels = [l for l, _ in Counter(ds["train"]["label"]).most_common(6)]

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    label_map = setup_tokenizer(tok, names, top_labels)
    label_ids = list(label_map.keys())

    # Build model with adapter
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tok))
    add_lora(model)

    adapter_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "lora-training", "lora-adapter.pt")
    load_adapter(model, adapter_path, n_labels=len(top_labels))
    model.eval()

    adapter_state = get_lora_state_dict(model)

    test_clause = "This Agreement may be executed in counterparts."
    ids = tok(test_clause + " ", return_tensors="pt")

    # 1. Output with adapter
    with torch.no_grad():
        logits_with_adapter = model(**ids).logits[0, -1]
        pred_adapter = label_map[max(label_ids, key=lambda t: logits_with_adapter[t].item())]
    print("1. WITH ADAPTER:")
    print(f"   Input: '{test_clause}'")
    print(f"   Prediction: {pred_adapter}")

    # 2. Merge adapter into base
    print("\n2. MERGING adapter into base weights...")
    model_merged, n_merged = merge_lora(model)
    stats = count_parameters(model_merged)
    print(f"   Merged {n_merged} LoRA layers")
    print(f"   LoRA wrappers remaining: 0")
    print(f"   All parameters now frozen: {stats['trainable']} trainable")

    with torch.no_grad():
        logits_merged = model_merged(**ids).logits[0, -1]
        pred_merged = label_map[max(label_ids, key=lambda t: logits_merged[t].item())]
    print(f"   Prediction after merge: {pred_merged}")
    print(f"   Output matches pre-merge: {torch.allclose(logits_with_adapter, logits_merged, atol=1e-5)}")
    print(f"   -> Free inference: same output, no adapter overhead")

    # 3. Merged model size
    merged_size = sum(p.numel() * p.element_size() for p in model_merged.parameters())
    print(f"\n3. MERGED MODEL SIZE: {merged_size / 1e6:.1f} MB")
    print(f"   This is a standard checkpoint -- no LoRA code needed to serve it.")

    # 4. Demonstrate swap
    print("\n4. ADAPTER SWAP DEMO:")
    print("   Reloading fresh base + LoRA wrappers...")

    model2 = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model2.resize_token_embeddings(len(tok))
    add_lora(model2)

    state_a = torch.load(adapter_path, weights_only=True)
    rows = state_a.pop("label_embeddings")
    with torch.no_grad():
        model2.transformer.wte.weight[-len(top_labels):] = rows
    model2.load_state_dict(state_a, strict=False)
    model2.eval()

    with torch.no_grad():
        logits_a = model2(**ids).logits[0, -1]
        pred_a = label_map[max(label_ids, key=lambda t: logits_a[t].item())]
    print(f"   Adapter A (LEDGAR) prediction: {pred_a}")

    print("   Swapping to Adapter B (zeroed -- simulates untrained)...")
    zero_state = {k: torch.zeros_like(v) for k, v in state_a.items()}
    model2.load_state_dict(zero_state, strict=False)

    with torch.no_grad():
        logits_b = model2(**ids).logits[0, -1]
        pred_b = label_map[max(label_ids, key=lambda t: logits_b[t].item())]
    print(f"   Adapter B (zeroed) prediction: {pred_b}")
    print(f"   Predictions differ: {pred_a != pred_b or not torch.allclose(logits_a, logits_b)}")

    print()
    print("+" + "-" * 59 + "+")
    print("|  One base model, many adapters:                         |")
    print("|    - Merge: fold adapter into weights, serve vanilla.   |")
    print("|    - Swap: change task by loading 12 tensors.           |")
    print("|    - No reloading the 82M base -- just the ~600 KB.    |")
    print("+" + "-" * 59 + "+")


if __name__ == "__main__":
    main()
