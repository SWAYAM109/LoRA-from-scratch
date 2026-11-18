#!/usr/bin/env python3
"""
Lesson 5 - Train Just the Adapter

The fine-tune itself: same LEDGAR clause-tagging task, but the optimizer
sees only 147,456 numbers (A and B matrices across 6 layers).
This is the compute-heavy lesson (~15 minutes on CPU).
Innovation: Tracks per-epoch metrics and saves training curves.
"""

import os
import sys
import json

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from lora_scratch.injection import add_lora
from lora_scratch.analysis import count_parameters
from lora_scratch.data import load_ledgar_task, setup_tokenizer, encode_clause
from lora_scratch.merge import save_adapter


def main():
    print("=" * 70)
    print("LESSON 5: Train Just the Adapter")
    print("=" * 70)
    print()

    # 1. Load the task
    print("Loading LEDGAR dataset...")
    task = load_ledgar_task(per_class=100, top_k=6, seed=0)
    train_data = task["train"]
    names = task["names"]
    top_labels = task["top_labels"]

    # 2. Setup tokenizer
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    label_map = setup_tokenizer(tok, names, top_labels)
    print(f"Added {len(top_labels)} label tokens: {[f'<{names[i]}>' for i in top_labels]}")

    # 3. Build model with LoRA
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.resize_token_embeddings(len(tok))
    add_lora(model, r=8, alpha=16, dropout=0.05)

    stats = count_parameters(model)
    print(f"\nTrainable: {stats['trainable']:,} / {stats['total']:,} = {stats['pct_trainable']:.2f}%")
    print()

    # 4. Tokenize training data
    def encode_fn(example):
        return encode_clause(example, tok, names, max_length=96)

    tokenized = train_data.map(encode_fn, remove_columns=train_data.column_names)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 5. Train
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "lora-training")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=6,
        per_device_train_batch_size=8,
        learning_rate=1e-3,
        logging_steps=50,
        report_to="none",
        save_strategy="no",
        disable_tqdm=True,
        warmup_ratio=0.05,
    )

    print("Training... (this takes ~15 minutes on CPU)")
    print("-" * 55)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )
    result = trainer.train()

    print("-" * 55)
    print(f"Training complete in {result.metrics['train_runtime']:.1f}s")
    print(f"Final loss: {result.metrics['train_loss']:.4f}")

    # 6. Save adapter + label embeddings
    adapter_path = os.path.join(output_dir, "lora-adapter.pt")
    extra = {"label_embeddings": model.transformer.wte.weight.data[-len(top_labels):].clone()}
    state = save_adapter(model, adapter_path, extra_state=extra)

    file_size = os.path.getsize(adapter_path)
    print(f"\nSaved adapter to: {adapter_path}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"  Contains: {len(state) - 1} adapter tensors + {len(top_labels)} label embedding rows")

    # Save training log
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "metrics": result.metrics,
            "log_history": trainer.state.log_history,
            "config": {
                "r": 8, "alpha": 16, "dropout": 0.05,
                "lr": 1e-3, "epochs": 6, "batch_size": 8,
                "trainable_params": stats["trainable"],
                "total_params": stats["total"],
            }
        }, f, indent=2, default=str)
    print(f"  Training log: {log_path}")

    print()
    print("+" + "-" * 59 + "+")
    print(f"|  147K numbers learned the task.                         |")
    print(f"|  Adapter: {file_size/1024:.0f} KB vs 327 MB for full fine-tune.        |")
    print("|  But a low loss is not a result.                         |")
    print("|  Next: does it match full fine-tuning's 95.8%?           |")
    print("+" + "-" * 59 + "+")


if __name__ == "__main__":
    main()
