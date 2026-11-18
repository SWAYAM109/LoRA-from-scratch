"""
Data utilities for the LEDGAR clause-tagging task.

Handles loading, balancing, tokenizing, and label-masking — the same
pipeline as Legal-Clause GPT, packaged for reuse.
"""

import random
from collections import Counter, defaultdict


def load_ledgar_task(per_class: int = 100, top_k: int = 6, seed: int = 0):
    """
    Load and prepare the LEDGAR clause-tagging task.
    
    Returns the top-k most common clause types, balanced train/val subsets,
    and the full dataset reference.
    
    Args:
        per_class: Number of examples per class for training.
        top_k: Number of clause types to use.
        seed: Random seed for reproducibility.
    
    Returns:
        dict with keys: 'train', 'val', 'names', 'top_labels', 'ds'
    """
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    names = ds["train"].features["label"].names
    top_labels = [label_id for label_id, _ in Counter(ds["train"]["label"]).most_common(top_k)]
    top_set = set(top_labels)

    train = _balanced_subset(ds["train"], top_set, per_class, seed)
    val = ds["validation"].filter(lambda e: e["label"] in top_set).shuffle(seed=seed + 2).select(range(per_class * top_k // 5))

    return {
        "train": train,
        "val": val,
        "names": names,
        "top_labels": top_labels,
        "ds": ds,
    }


def _balanced_subset(split, top_set, per_class, seed):
    """Pick exactly per_class examples for each label in top_set."""
    picked = defaultdict(list)
    for i, lab in enumerate(split["label"]):
        if lab in top_set and len(picked[lab]) < per_class:
            picked[lab].append(i)
        if all(len(picked[l]) >= per_class for l in top_set):
            break
    flat = [i for l in sorted(top_set) for i in picked[l]]
    random.Random(seed).shuffle(flat)
    return split.select(flat)


def setup_tokenizer(tokenizer, names, top_labels):
    """
    Add clause-type tokens to the tokenizer and return label mappings.
    
    Args:
        tokenizer: HuggingFace tokenizer.
        names: List of all label names.
        top_labels: List of label indices to add as special tokens.
    
    Returns:
        dict mapping label token ids to label names.
    """
    tokenizer.pad_token = tokenizer.eos_token
    new_tokens = [f"<{names[i]}>" for i in top_labels]
    tokenizer.add_tokens(new_tokens)
    return {tokenizer.convert_tokens_to_ids(f"<{names[i]}>"): names[i] for i in top_labels}


def encode_clause(example, tokenizer, names, max_length=96):
    """
    Encode a single clause for training.
    
    Format: "{clause_text} <LabelToken>"
    Labels are masked to -100 everywhere except the label token position,
    so the model is graded on a single question: given this clause, which type?
    """
    label_str = f"<{names[example['label']]}>"
    enc = tokenizer(
        example["text"].strip() + " " + label_str,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    label_tid = tokenizer.convert_tokens_to_ids(label_str)
    labels = [-100] * len(enc["input_ids"])
    if label_tid in enc["input_ids"]:
        labels[enc["input_ids"].index(label_tid)] = label_tid

    enc["labels"] = labels
    return enc
