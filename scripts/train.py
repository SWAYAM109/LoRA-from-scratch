#!/usr/bin/env python3
"""
Train a LoRA adapter on the LEDGAR clause-tagging task.

Usage:
    python scripts/train.py --rank 8 --alpha 16 --epochs 6 --lr 1e-3
    python scripts/train.py --rank 16 --targets attention_full
    python scripts/train.py --init-strategy xavier
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from lora_scratch.injection import add_lora, TARGET_PRESETS
from lora_scratch.analysis import count_parameters
from lora_scratch.data import load_ledgar_task, setup_tokenizer, encode_clause
from lora_scratch.merge import save_adapter


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapter on LEDGAR")
    parser.add_argument("--rank", "-r", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--alpha", "-a", type=float, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument("--dropout", type=float, default=0.05, help="Adapter dropout (default: 0.05)")
    parser.add_argument("--targets", type=str, default="attention_qkv",
                        choices=list(TARGET_PRESETS.keys()),
                        help="Which layers to adapt (default: attention_qkv)")
    parser.add_argument("--init-strategy", type=str, default="kaiming",
                        choices=["kaiming", "gaussian", "xavier"],
                        help="A matrix initialization (default: kaiming)")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs (default: 6)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--max-length", type=int, default=96, help="Max sequence length (default: 96)")
    parser.add_argument("--per-class", type=int, default=100, help="Training examples per class (default: 100)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LoRA Adapter Training")
    print("=" * 60)
    print(f"  Rank: {args.rank}  |  Alpha: {args.alpha}  |  Targets: {args.targets}")
    print(f"  Init: {args.init_strategy}  |  LR: {args.lr}  |  Epochs: {args.epochs}")
    print()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(__file__), "..", "outputs",
            f"lora-r{args.rank}-{args.targets}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Load task
    print("Loading LEDGAR dataset...")
    task = load_ledgar_task(per_class=args.per_class, top_k=6, seed=args.seed)
    names = task["names"]
    top_labels = task["top_labels"]

    # Setup tokenizer
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    label_map = setup_tokenizer(tok, names, top_labels)
    print(f"Labels: {[f'<{names[i]}>' for i in top_labels]}")

    # Build model
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tok))
    add_lora(model, r=args.rank, alpha=args.alpha, dropout=args.dropout,
             targets=args.targets, init_strategy=args.init_strategy)

    stats = count_parameters(model)
    print(f"\nTrainable: {stats['trainable']:,} / {stats['total']:,} = {stats['pct_trainable']:.2f}%")

    # Tokenize
    def encode_fn(example):
        return encode_clause(example, tok, names, max_length=args.max_length)

    tokenized = task["train"].map(encode_fn, remove_columns=task["train"].column_names)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        report_to="none",
        save_strategy="no",
        disable_tqdm=True,
        warmup_ratio=0.05,
    )

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 50)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )
    result = trainer.train()

    print("-" * 50)
    print(f"Done in {result.metrics['train_runtime']:.1f}s | Final loss: {result.metrics['train_loss']:.4f}")

    # Save adapter
    adapter_path = os.path.join(args.output_dir, "lora-adapter.pt")
    extra = {"label_embeddings": model.transformer.wte.weight.data[-len(top_labels):].clone()}
    save_adapter(model, adapter_path, extra_state=extra)

    file_size = os.path.getsize(adapter_path)
    print(f"\nAdapter saved: {adapter_path} ({file_size/1024:.1f} KB)")

    # Save config
    config = {
        "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
        "targets": args.targets, "init_strategy": args.init_strategy,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "trainable_params": stats["trainable"], "total_params": stats["total"],
        "final_loss": result.metrics["train_loss"],
        "runtime_seconds": result.metrics["train_runtime"],
        "log_history": trainer.state.log_history,
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Config saved: {config_path}")


if __name__ == "__main__":
    main()
