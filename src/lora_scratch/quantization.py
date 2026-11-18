"""
Quantization — 4-bit weight compression for the frozen base.

Two schemes:
  1. Block-wise Absmax with real nibble packing (from lora-forge):
     - Signed 4-bit values packed two-per-byte into uint8
     - On forward: dequantize on the fly (true ~0.5B/param storage)

  2. NF4 (Normal Float 4-bit) — levels placed at normal-distribution
     quantiles. Recovers more accuracy than absmax for Gaussian weights.

Both schemes store one fp32 scale per block of 64 weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------------------------------------------------ #
#  NF4 lookup table (computed once at import)
# ------------------------------------------------------------------ #

def _compute_nf4_levels() -> torch.Tensor:
    """16 levels placed at normal-distribution quantile midpoints, normalised to [-1,1]."""
    try:
        from scipy.stats import norm
        n_neg, n_pos = 7, 8
        neg_q = np.linspace(0, 0.5, n_neg + 1)
        pos_q = np.linspace(0.5, 1.0, n_pos + 1)
        levels = []
        for lo, hi in zip(neg_q[:-1], neg_q[1:]):
            lo_v, hi_v = norm.ppf(lo + 1e-9), norm.ppf(hi)
            mid = (norm.pdf(lo_v) - norm.pdf(hi_v)) / max(norm.cdf(hi_v) - norm.cdf(lo_v), 1e-9)
            levels.append(-abs(mid))
        levels.append(0.0)
        for lo, hi in zip(pos_q[:-1], pos_q[1:]):
            lo_v, hi_v = norm.ppf(lo), norm.ppf(hi - 1e-9)
            mid = (norm.pdf(lo_v) - norm.pdf(hi_v)) / max(norm.cdf(hi_v) - norm.cdf(lo_v), 1e-9)
            levels.append(abs(mid))
        arr = np.array(sorted(levels))
        arr /= max(abs(arr.max()), abs(arr.min()))
        return torch.tensor(arr, dtype=torch.float32)
    except ImportError:
        # Pre-computed fallback
        return torch.tensor([
            -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0000,
             0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000,
        ])


NF4_LEVELS = _compute_nf4_levels()


# ------------------------------------------------------------------ #
#  Scheme 1: Block-wise Absmax with real nibble packing
# ------------------------------------------------------------------ #

def quantize_absmax_4bit(weight: torch.Tensor, block_size: int = 64):
    """
    Pack weight into 4-bit nibbles (two values per uint8 byte).

    Signed symmetric quantization: each value mapped to [-7, 7],
    stored as nibble offset [1, 15] to avoid the zero nibble as sentinel.

    Returns:
        packed   : uint8 tensor, shape [ceil(numel/2)]
        scales   : fp32 per-block scale, shape [n_blocks]
        pad      : number of zero-padding elements added
        shape    : original weight shape
    """
    flat = weight.detach().to(torch.float32).contiguous().view(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))

    blocks = flat.view(-1, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0  # per-block scale
    quantized = torch.round(blocks / scales).clamp(-7, 7).to(torch.int16).view(-1)

    # Encode: shift [-7,7] -> [1,15] so nibble 0 is never a data value
    encoded = (quantized + 8).to(torch.uint8)
    if encoded.numel() % 2:
        encoded = F.pad(encoded, (0, 1))

    # Pack two 4-bit values per byte
    packed = encoded[0::2] | (encoded[1::2] << 4)
    return packed.contiguous(), scales.view(-1).contiguous(), pad, tuple(weight.shape)


def dequantize_absmax(
    packed: torch.Tensor,
    scales: torch.Tensor,
    pad: int,
    shape: tuple,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack nibbles and rescale back to float."""
    low  = packed & 0x0F
    high = (packed >> 4) & 0x0F
    encoded = torch.stack((low, high), dim=1).reshape(-1).to(torch.int16)
    signed = (encoded - 8).to(torch.float32)

    padded_count = math.ceil(math.prod(shape) / block_size) * block_size
    signed = signed[:padded_count].view(-1, block_size)
    restored = (signed * scales.to(torch.float32).view(-1, 1)).reshape(-1)
    if pad:
        restored = restored[:-pad]
    return restored.view(shape).to(dtype=dtype)


# ------------------------------------------------------------------ #
#  Scheme 2: NF4 — Normal Float 4-bit (Innovation)
# ------------------------------------------------------------------ #

def quantize_nf4(weight: torch.Tensor, block_size: int = 64):
    """
    Map each weight to the nearest of 16 NF4 levels after per-block normalisation.
    NF4 levels are optimal for normally-distributed weights (as pretrained LLMs have).
    Returns indices as uint8 (values 0-15).
    """
    levels = NF4_LEVELS.to(weight.device)
    flat = weight.detach().float().contiguous().view(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))

    blocks = flat.view(-1, block_size)
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    normalised = blocks / scales  # each block normalised to [-1, 1]

    # Nearest NF4 level (vectorised)
    distances = (normalised.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)
    return indices.contiguous(), scales.view(-1).contiguous(), pad, tuple(weight.shape), levels


def dequantize_nf4(
    indices: torch.Tensor,
    scales: torch.Tensor,
    pad: int,
    shape: tuple,
    levels: torch.Tensor,
) -> torch.Tensor:
    """Look up NF4 levels and rescale."""
    levels = levels.to(scales.device)
    dequantised = levels[indices.long()] * scales.view(-1, 1)
    out = dequantised.reshape(-1)
    if pad:
        out = out[:-pad]
    return out.view(shape)


# ------------------------------------------------------------------ #
#  QuantizedProjection — real 4-bit frozen layer (from lora-forge)
# ------------------------------------------------------------------ #

class QuantizedProjection(nn.Module):
    """
    Frozen projection whose weight is stored in packed block-wise 4-bit form.
    Dequantizes on every forward pass. Keeps ~0.5 bytes per weight (vs 4 in fp32).
    """

    def __init__(self, base: nn.Module, block_size: int = 64, scheme: str = "absmax") -> None:
        super().__init__()
        if not hasattr(base, "weight") or base.weight.ndim != 2:
            raise TypeError("QuantizedProjection expects a module with a 2D weight.")
        if scheme not in ("absmax", "nf4"):
            raise ValueError(f"scheme must be 'absmax' or 'nf4', got '{scheme}'")

        self.fan_in_fan_out = type(base).__name__ == "Conv1D" or bool(
            getattr(base, "fan_in_fan_out", False)
        )
        self.block_size = block_size
        self.scheme = scheme
        self.weight_shape = tuple(base.weight.shape)
        self._weight_dtype = base.weight.dtype

        if self.fan_in_fan_out:
            self.in_features, self.out_features = base.weight.shape
        else:
            self.out_features, self.in_features = base.weight.shape

        if scheme == "absmax":
            packed, scales, pad, shape = quantize_absmax_4bit(base.weight, block_size)
            self.register_buffer("packed_weight", packed)
            self.register_buffer("scales", scales)
            self.pad = int(pad)
            self.original_shape = shape
            self._nf4_levels = None
        else:  # nf4
            indices, scales, pad, shape, nf4_levels = quantize_nf4(base.weight, block_size)
            self.register_buffer("packed_weight", indices)
            self.register_buffer("scales", scales)
            self.pad = int(pad)
            self.original_shape = shape
            self._nf4_levels = nf4_levels

        bias = getattr(base, "bias", None)
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

    def dequantized_weight(self) -> torch.Tensor:
        if self.scheme == "absmax":
            return dequantize_absmax(
                self.packed_weight, self.scales, self.pad,
                self.original_shape, self.block_size, self._weight_dtype,
            )
        else:
            return dequantize_nf4(
                self.packed_weight, self.scales, self.pad,
                self.original_shape, self._nf4_levels,
            ).to(self._weight_dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight()
        if self.fan_in_fan_out:
            weight = weight.transpose(0, 1)
        return F.linear(inputs, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"scheme={self.scheme}, block_size={self.block_size}"
        )


# ------------------------------------------------------------------ #
#  Apply quantization to a full model
# ------------------------------------------------------------------ #

def quantize_model_weights(
    model: nn.Module,
    scheme: str = "absmax",
    block_size: int = 64,
) -> tuple[int, float]:
    """
    Replace every frozen Conv1D projection with a QuantizedProjection.
    LoRA A and B parameters are untouched (they stay in fp32).

    Returns (n_quantized_params, mean_relative_error).
    """
    from .core import LoRALinear

    replacements = []
    for name, module in list(model.named_modules()):
        if not name:
            continue
        # Target Conv1D layers that are either bare or wrapped in LoRALinear
        if type(module).__name__ == "Conv1D":
            replacements.append(name)
        elif isinstance(module, LoRALinear) and type(module.base).__name__ == "Conv1D":
            replacements.append(name + ".base")

    if not replacements:
        # Fallback: quantize all Linear layers (for plain nn.Linear models)
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and not any(
                isinstance(module, LoRALinear) for _ in [None]
            ):
                replacements.append(name)

    n_quantized = 0
    errors = []

    for dotted in replacements:
        parts = dotted.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]
        child = getattr(parent, child_name)

        if not hasattr(child, "weight"):
            continue

        # Measure error before replacing
        orig_w = child.weight.data.float()
        quant_proj = QuantizedProjection(child, block_size=block_size, scheme=scheme)
        dq_w = quant_proj.dequantized_weight().float()
        rel_err = (orig_w - dq_w).abs().mean().item() / (orig_w.abs().mean().item() + 1e-8)
        errors.append(rel_err)
        n_quantized += orig_w.numel()

        setattr(parent, child_name, quant_proj)

    mean_error = sum(errors) / len(errors) if errors else 0.0
    return n_quantized, mean_error
