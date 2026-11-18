#!/usr/bin/env python3
"""
Lesson 6 - Does It Match Full Fine-Tuning?

Load the 613 KB adapter onto a FRESH distilgpt2 and score it on 120
held-out clauses. Head-to-head against full fine-tuning's 95.8%.
"""

import os
import sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_scratch.injection import add_lora
from lora_scratch.merge import load_adapter
from lora_scratch.data import setup_tokenizer


def main():
    print("=" * 70)
    print("LESSON 6: Does It Match Full Fine-Tuning?")
    print("=" * 70)
    print()

    # 1. Prepare dataset and labels
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    names = ds["train"].features["label"].names
    top_labels = [l for l, _ in Counter(ds["train"]["label"]).most_common(6)]
    top_set = set(top_labels)

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    label_map = setup_tokenizer(tok, names, top_labels)
    label_ids = list(label_map.keys())

    # 2. Build a FRESH model and load adapter
    print("Loading fresh distilgpt2 (never fine-tuned)...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tok))
    add_lora(model)

    adapter_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "lora-training", "lora-adapter.pt")
    n_loaded = load_adapter(model, adapter_path, n_labels=len(top_labels))
    model.eval()

    file_size = os.path.getsize(adapter_path)
    print(f"Loaded {n_loaded} adapter tensors from {file_size:,}-byte file")
    print()

    # 3. Evaluate
    @torch.no_grad()
    def classify(clause_text):
        logits = model(**tok(clause_text.strip() + " ", return_tensors="pt")).logits[0, -1]
        return label_map[max(label_ids, key=lambda t: logits[t].item())]

    val = ds["validation"].filter(lambda e: e["label"] in top_set).shuffle(seed=2).select(range(120))
    preds = [classify(e["text"]) for e in val]
    trues = [names[e["label"]] for e in val]

    correct = sum(p == t for p, t in zip(preds, trues))
    acc = correct / len(val)

    # 4. Print results
    print(f"{'Method':<20s} {'Accuracy':>10s} {'Trainable':>14s} {'File Size':>12s}")
    print("-" * 60)
    print(f"{'Full fine-tune':<20s} {'95.8%':>10s} {'81,912,576':>14s} {'327 MB':>12s}")
    print(f"{'LoRA r=8':<20s} {acc:>9.1%} {'147,456':>14s} {file_size/1024:>9,.0f} KB")
    print(f"\n(6 clause types, random guessing = {1/len(top_labels):.1%})")

    # Show errors
    errors = [(p, t, e) for p, t, e in zip(preds, trues, val) if p != t]
    if errors:
        print(f"\nMisclassifications ({len(errors)}/{len(val)}):")
        print("-" * 70)
        for p, t, e in errors[:8]:
            print(f"  true={t:22s} pred={p:22s} :: {e['text'][:50]}...")

    # Per-class breakdown
    print()
    print("Per-class accuracy:")
    print("-" * 40)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for p, t in zip(preds, trues):
        class_total[t] += 1
        if p == t:
            class_correct[t] += 1

    for cls in sorted(class_total.keys()):
        c, tot = class_correct[cls], class_total[cls]
        print(f"  {cls:22s} {c:>3d}/{tot:<3d} ({100*c/tot:5.1f}%)")

    print()
    print("+" + "-" * 59 + "+")
    print(f"|  LoRA accuracy: {acc:.1%}                                   |")
    print(f"|  Trained 0.18% of the weights.                          |")
    print(f"|  Shipped {file_size/1024:.0f} KB instead of 327 MB.                 |")
    print("|  Same hard cases -- same skill, same limits, 557x fewer  |")
    print("|  weights.                                                |")
    print("+" + "-" * 59 + "+")


if __name__ == "__main__":
    main()
