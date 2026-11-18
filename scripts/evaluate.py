#!/usr/bin/env python3
"""
Evaluate a trained LoRA adapter on LEDGAR validation set.

Usage:
    python scripts/evaluate.py --adapter outputs/lora-r8-attention_qkv/lora-adapter.pt
    python scripts/evaluate.py --adapter outputs/lora-training/lora-adapter.pt --targets attention_qkv
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter")
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter file")
    parser.add_argument("--rank", "-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", "-a", type=float, default=16, help="LoRA alpha")
    parser.add_argument("--targets", type=str, default="attention_qkv", help="Injection targets")
    parser.add_argument("--n-eval", type=int, default=120, help="Number of eval examples")
    parser.add_argument("--seed", type=int, default=2, help="Eval shuffle seed")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LoRA Adapter Evaluation")
    print("=" * 60)
    print(f"  Adapter: {args.adapter}")
    print(f"  Rank: {args.rank}  |  Targets: {args.targets}")
    print()

    # Setup
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    names = ds["train"].features["label"].names
    top_labels = [l for l, _ in Counter(ds["train"]["label"]).most_common(6)]
    top_set = set(top_labels)

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    label_map = setup_tokenizer(tok, names, top_labels)
    label_ids = list(label_map.keys())

    # Build model
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tok))
    add_lora(model, r=args.rank, alpha=args.alpha, targets=args.targets)
    load_adapter(model, args.adapter, n_labels=len(top_labels))
    model.eval()

    # Evaluate
    val = ds["validation"].filter(lambda e: e["label"] in top_set).shuffle(seed=args.seed).select(range(args.n_eval))
    trues = [names[e["label"]] for e in val]

    @torch.no_grad()
    def classify(text):
        logits = model(**tok(text.strip() + " ", return_tensors="pt")).logits[0, -1]
        return label_map[max(label_ids, key=lambda t: logits[t].item())]

    preds = [classify(e["text"]) for e in val]
    correct = sum(p == t for p, t in zip(preds, trues))
    acc = correct / len(val)

    # Results
    print(f"Accuracy: {acc:.1%} ({correct}/{len(val)})")
    print(f"File size: {os.path.getsize(args.adapter)/1024:.1f} KB")
    print()

    # Per-class breakdown
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for p, t in zip(preds, trues):
        class_total[t] += 1
        if p == t:
            class_correct[t] += 1

    print("Per-class:")
    print("-" * 50)
    for cls in sorted(class_total.keys()):
        c, tot = class_correct[cls], class_total[cls]
        pct = 100 * c / tot
        bar_len = int(20 * c / tot)
        print(f"  {cls:22s} {c:>3d}/{tot:<3d} ({pct:5.1f}%)")

    # Errors
    errors = [(p, t, e) for p, t, e in zip(preds, trues, val) if p != t]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for p, t, e in errors[:10]:
            print(f"  true={t:22s} pred={p:22s} :: {e['text'][:45]}...")


if __name__ == "__main__":
    main()
