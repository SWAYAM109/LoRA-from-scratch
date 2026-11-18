"""
Injection — freeze a pretrained model and bolt LoRA adapters onto chosen layers.

Supports targeting:
  - c_attn (Q/K/V fused projection) — the standard LoRA target
  - c_proj (attention output projection) — adds expressivity
  - mlp.c_fc / mlp.c_proj (feed-forward) — rarely needed but available

Innovation: multi-target injection with per-target rank configuration.
"""

import torch.nn as nn
from .core import LoRALinear


# Predefined target sets
TARGET_PRESETS = {
    "attention_qkv": ["attn.c_attn"],
    "attention_full": ["attn.c_attn", "attn.c_proj"],
    "attention_and_mlp": ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
}


def add_lora(
    model,
    r: int = 8,
    alpha: float = 16,
    dropout: float = 0.05,
    targets: str | list[str] = "attention_qkv",
    init_strategy: str = "kaiming",
):
    """
    Freeze the entire pretrained model, then wrap specified layers with LoRALinear.

    Args:
        model: A HuggingFace GPT-2 style causal LM.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout for the adapter path.
        targets: Which layers to adapt. Either a preset name (str) from
                 TARGET_PRESETS or an explicit list of submodule suffixes.
        init_strategy: Initialization for A matrices ('kaiming', 'gaussian', 'xavier').

    Returns:
        The model (modified in-place) with LoRA adapters injected.
    """
    # Resolve target list
    if isinstance(targets, str):
        if targets in TARGET_PRESETS:
            target_suffixes = TARGET_PRESETS[targets]
        else:
            target_suffixes = [targets]
    else:
        target_suffixes = list(targets)

    # Step 1: freeze EVERYTHING
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: wrap target layers in each transformer block
    injected = 0
    for block in model.transformer.h:
        for suffix in target_suffixes:
            # Navigate to the target submodule
            parts = suffix.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            original = getattr(parent, attr_name)

            # Wrap it
            lora_layer = LoRALinear(
                original, r=r, alpha=alpha, dropout=dropout, init_strategy=init_strategy
            )
            setattr(parent, attr_name, lora_layer)
            injected += 1

    if injected == 0:
        raise RuntimeError(
            f"No layers matched targets {target_suffixes}. "
            f"Check that the model has transformer.h[i].{target_suffixes[0]}."
        )

    return model


def remove_lora(model):
    """
    Remove all LoRA wrappers, restoring the original frozen base layers.
    
    Useful for reverting to a clean base before injecting a different adapter.
    """
    removed = 0
    for block in model.transformer.h:
        for name, module in list(block.named_modules()):
            if isinstance(module, LoRALinear):
                parts = name.split(".")
                parent = block
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], module.base)
                removed += 1
    return model, removed


def get_lora_parameters(model):
    """Return only the LoRA A and B parameters (the trainable ones)."""
    return [p for n, p in model.named_parameters() if p.requires_grad]


def get_lora_state_dict(model):
    """Return a state dict containing only LoRA adapter weights."""
    return {k: v for k, v in model.state_dict().items() if ".A" in k or ".B" in k}
