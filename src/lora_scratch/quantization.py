"""
Quantization — 4-bit weight compression for the frozen base.

Two schemes implemented:
  1. Block-wise Absmax: simple 16-level symmetric quantization (baseline)
  2. NF4 (Normal Float 4-bit): levels placed at normal-distribution quantiles
     (INNOVATION — the scheme from the QLoRA paper, implemented from scratch)

NF4 exploits the empirical fact that pretrained weights are approximately
normally distributed: its 16 levels are the expected values of 16 equal-probability
bins under N(0,1), which minimizes expected quantization error for Gaussian data.
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================
# NF4 lookup table — the 16 levels from the QLoRA paper
# These are the expected values of 16 equal-probability bins
# under a standard normal distribution, normalized to [-1, 1].
# ============================================================
def _compute_nf4_levels() -> torch.Tensor:
    """
    Compute the 16 NF4 quantization levels.
    
    For a standard normal, divide the CDF into 2^4 = 16 equal-probability bins.
    The representative value for each bin is E[X | X in bin], normalized to [-1, 1].
    
    One level is fixed at 0.0 for exact zero representation (important for
    sparse weights and zero-initialized parameters).
    """
    from scipy.stats import norm

    n_levels = 16
    # 15 non-zero levels + 1 zero level
    # Compute quantiles for 15 levels (excluding zero)
    n_positive = 8  # 8 positive levels
    n_negative = 7  # 7 negative levels (the 16th slot is zero)

    # Negative levels: quantiles of the negative half
    neg_quantiles = np.linspace(0, 0.5, n_negative + 1)
    neg_levels = []
    for i in range(n_negative):
        # Expected value in the bin [q_i, q_{i+1}]
        lo, hi = norm.ppf(neg_quantiles[i] + 1e-10), norm.ppf(neg_quantiles[i + 1])
        # E[X | lo < X < hi] using truncated normal formula
        level = (norm.pdf(lo) - norm.pdf(hi)) / (norm.cdf(hi) - norm.cdf(lo) + 1e-10)
        neg_levels.append(-abs(level))

    # Positive levels: mirror of negative
    pos_quantiles = np.linspace(0.5, 1.0, n_positive + 1)
    pos_levels = []
    for i in range(n_positive):
        lo, hi = norm.ppf(pos_quantiles[i]), norm.ppf(pos_quantiles[i + 1] - 1e-10)
        level = (norm.pdf(lo) - norm.pdf(hi)) / (norm.cdf(hi) - norm.cdf(lo) + 1e-10)
        pos_levels.append(abs(level))

    levels = sorted(neg_levels) + [0.0] + sorted(pos_levels)
    # Normalize to [-1, 1]
    levels = np.array(levels)
    levels = levels / np.max(np.abs(levels))
    return torch.tensor(levels, dtype=torch.float32)


# Pre-compute once at import time
try:
    NF4_LEVELS = _compute_nf4_levels()
except ImportError:
    # Fallback if scipy not available — use pre-computed values
    NF4_LEVELS = torch.tensor([
        -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000
    ])


# ============================================================
# Scheme 1: Block-wise Absmax (baseline)
# ============================================================

def quantize_absmax_4bit(weight: torch.Tensor, block_size: int = 64):
    """
    Block-wise absmax 4-bit quantization.
    
    Chops the weight into blocks, finds the max absolute value per block,
    and rounds each weight to one of 16 symmetric levels.
    
    Args:
        weight: The tensor to quantize.
        block_size: Number of elements per quantization block.
    
    Returns:
        Tuple of (quantized_int8, scales, padding, original_shape).
    """
    flat = weight.detach().flatten().float()
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])

    blocks = flat.view(-1, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)

    # Quantize to [-8, 7] (4-bit signed range)
    q = torch.round(blocks / scales * 7).clamp(-8, 7).to(torch.int8)

    return q, scales, pad, weight.shape


def dequantize_absmax(q, scales, pad, shape):
    """Dequantize absmax 4-bit back to float."""
    out = (q.float() / 7 * scales).flatten()
    if pad:
        out = out[: out.numel() - pad]
    return out.view(shape)


# ============================================================
# Scheme 2: NF4 (Normal Float 4-bit) — INNOVATION
# ============================================================

def quantize_nf4(weight: torch.Tensor, block_size: int = 64):
    """
    NF4 quantization: map each weight to the nearest NF4 level after block normalization.
    
    This exploits the empirical observation that pretrained neural network weights
    are approximately normally distributed. The NF4 levels are optimal for this case.
    
    Args:
        weight: Tensor to quantize.
        block_size: Block size for per-block scaling.
    
    Returns:
        Tuple of (quantized_indices_uint8, scales, padding, original_shape, nf4_levels).
    """
    levels = NF4_LEVELS.to(weight.device)

    flat = weight.detach().flatten().float()
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])

    blocks = flat.view(-1, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)

    # Normalize to [-1, 1] per block
    normalized = blocks / scales

    # Find nearest NF4 level for each weight (vectorized)
    # Shape: [n_blocks, block_size, 1] vs [1, 1, 16]
    distances = (normalized.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)  # [n_blocks, block_size]

    return indices, scales, pad, weight.shape, levels


def dequantize_nf4(indices, scales, pad, shape, levels):
    """Dequantize NF4 back to float using the level lookup table."""
    levels = levels.to(scales.device)
    # Look up the float value for each index
    dequantized = levels[indices.long()] * scales  # [n_blocks, block_size]
    out = dequantized.flatten()
    if pad:
        out = out[: out.numel() - pad]
    return out.view(shape)


# ============================================================
# Apply quantization to a model
# ============================================================

def quantize_model_weights(model, scheme: str = "absmax", block_size: int = 64):
    """
    Quantize all frozen base weights in-place (dequantize and store back).
    
    This simulates what QLoRA does: the base is stored at reduced precision,
    and dequantized on-the-fly during forward passes. Here we do a one-shot
    dequantize for simplicity (the precision loss is the same).
    
    Args:
        model: Model with or without LoRA wrappers.
        scheme: 'absmax' or 'nf4'.
        block_size: Quantization block size.
    
    Returns:
        Tuple of (n_quantized_params, mean_relative_error).
    """
    from .core import LoRALinear
    from transformers.pytorch_utils import Conv1D

    n_quantized = 0
    errors = []

    with torch.no_grad():
        for module in model.modules():
            # Quantize frozen base weights only
            target = None
            if isinstance(module, LoRALinear):
                target = module.base
            elif isinstance(module, (nn.Linear, Conv1D)):
                target = module

            if target is None:
                continue

            w = target.weight.data
            if scheme == "nf4":
                dq = dequantize_nf4(*quantize_nf4(w, block_size))
            else:
                dq = dequantize_absmax(*quantize_absmax_4bit(w, block_size))

            # Measure error
            rel_err = (w - dq).abs().mean().item() / (w.abs().mean().item() + 1e-8)
            errors.append(rel_err)

            target.weight.data = dq
            n_quantized += w.numel()

    mean_error = sum(errors) / len(errors) if errors else 0.0
    return n_quantized, mean_error
