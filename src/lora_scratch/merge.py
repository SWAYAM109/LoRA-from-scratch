"""
Merge, unmerge, and swap LoRA adapters.

- merge: fold B @ A * scale into the base weight for free inference (no adapter overhead)
- unmerge: subtract it back out for continued training or swapping
- swap: hot-swap one adapter for another on a single frozen base
"""

import torch
import torch.nn as nn
from .core import LoRALinear, _is_conv1d


def merge_lora(model):
    """
    Fold every LoRA adapter into its base weight: W' = W + B @ A * (alpha/r).
    
    After merging, the model runs at original speed with no adapter overhead.
    The LoRALinear wrappers are replaced with plain base layers.
    
    Returns:
        The model with adapters merged and LoRA wrappers removed.
    """
    merged = 0
    for block in model.transformer.h:
        for name, module in list(block.named_modules()):
            if isinstance(module, LoRALinear):
                _merge_single(module)
                # Replace wrapper with the now-modified base
                parts = name.split(".")
                parent = block
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], module.base)
                merged += 1
    return model, merged


def _merge_single(lora_module: LoRALinear):
    """Merge adapter weights into the base weight tensor."""
    base = lora_module.base
    # Guard: cannot merge into a quantized base (from lora-forge)
    if not hasattr(base, "weight") or not isinstance(base.weight, torch.Tensor):
        raise TypeError(
            "Cannot merge LoRA into a base layer without a floating-point weight. "
            "If the base is quantized, dequantize first."
        )
    if not base.weight.is_floating_point():
        raise TypeError(
            "Cannot merge a LoRA adapter into a non-floating-point weight. "
            "Merge is only valid on fp16/fp32 base layers."
        )
    delta = lora_module.effective_delta_w()  # [d_out, d_in]
    with torch.no_grad():
        if lora_module.fan_in_fan_out:
            base.weight.data += delta.T.to(base.weight.dtype)
        else:
            base.weight.data += delta.to(base.weight.dtype)


def unmerge_lora(model):
    """
    Subtract merged adapter from base weights, restoring original pretrained values.
    
    Only works if the LoRALinear wrappers are still in place (i.e., merge was done
    in-place on the weight but the wrapper was kept).
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta = module.effective_delta_w()
            with torch.no_grad():
                if module._is_conv1d:
                    module.base.weight.data -= delta.T
                else:
                    module.base.weight.data -= delta


def swap_adapter(model, new_state_dict: dict):
    """
    Hot-swap one adapter for another on the same frozen base.
    
    1. Zero out the current A and B.
    2. Load the new adapter's A and B from new_state_dict.
    
    The base weights are never touched — only the adapter tensors change.
    
    Args:
        model: Model with LoRALinear wrappers in place.
        new_state_dict: Dict mapping parameter names to new A/B tensors.
    """
    # Zero current adapter
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.data.zero_()
            module.B.data.zero_()

    # Load new adapter
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    loaded = [k for k in new_state_dict if k not in unexpected]
    return loaded, unexpected


def save_adapter(model, path: str, extra_state: dict = None):
    """
    Save only the adapter weights (and optionally extra state like label embeddings).
    
    Args:
        model: Model with LoRA adapters.
        path: File path to save to.
        extra_state: Additional tensors to include (e.g., label embeddings).
    """
    state = {k: v for k, v in model.state_dict().items() if ".A" in k or ".B" in k}
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)
    return state


def load_adapter(model, path: str, label_embedding_key: str = "label_embeddings", n_labels: int = 6):
    """
    Load an adapter onto a model with LoRA wrappers already in place.
    
    Also handles restoring label embeddings if present.
    
    Args:
        model: Model with LoRALinear wrappers.
        path: Path to the saved adapter file.
        label_embedding_key: Key for label embeddings in the saved state.
        n_labels: Number of label embedding rows.
    
    Returns:
        Number of adapter tensors loaded.
    """
    state = torch.load(path, weights_only=True)

    # Restore label embeddings if present
    if label_embedding_key in state:
        rows = state.pop(label_embedding_key)
        with torch.no_grad():
            model.transformer.wte.weight[-n_labels:] = rows

    missing, unexpected = model.load_state_dict(state, strict=False)
    return len(state)
