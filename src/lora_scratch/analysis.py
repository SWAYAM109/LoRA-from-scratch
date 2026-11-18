"""
Analysis and visualization utilities.

Provides tools for:
  - Parameter counting and memory estimation
  - Rank ablation studies (INNOVATION: automated sweep across ranks)
  - Adapter spectral analysis (INNOVATION: SVD-based insight into what the adapter learned)
"""

import torch
import numpy as np


def count_parameters(model):
    """
    Count and categorize model parameters.
    
    Returns:
        dict with 'total', 'trainable', 'frozen', 'pct_trainable', and 'by_layer' breakdown.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    by_layer = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            by_layer[name] = p.numel()

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "pct_trainable": 100 * trainable / total if total > 0 else 0,
        "by_layer": by_layer,
    }


def training_memory_estimate(n_params, method="full_ft", lora_frac=0.0018):
    """
    Estimate training memory in GB.
    
    Args:
        n_params: Total number of model parameters.
        method: One of 'full_ft', 'lora', 'qlora'.
        lora_frac: Fraction of parameters that are trainable (for LoRA/QLoRA).
    
    Returns:
        Estimated memory in GB.
    """
    GB = 1e9
    if method == "full_ft":
        # weight + gradient + Adam mean + Adam variance = 16 bytes/param (fp32)
        return n_params * 16 / GB
    elif method == "lora":
        # Frozen base in fp16 (2B/param) + adapter trains in fp32 (16B/param)
        base_mem = n_params * 2 / GB
        adapter_mem = n_params * lora_frac * 16 / GB
        return base_mem + adapter_mem
    elif method == "qlora":
        # Frozen base in 4-bit (~0.5B/param) + adapter trains in fp32
        base_mem = n_params * 0.5 / GB
        adapter_mem = n_params * lora_frac * 16 / GB
        return base_mem + adapter_mem
    else:
        raise ValueError(f"Unknown method: {method}")


def rank_ablation_study(d=768, true_rank=6, noise_std=0.02, ranks=None, seed=0):
    """
    INNOVATION: Demonstrate how much of a weight update is captured at each rank.
    
    Generates a synthetic low-rank + noise update matrix and measures how much
    energy (Frobenius norm) is captured by rank-r approximations.
    
    Args:
        d: Dimension of the square weight matrix.
        true_rank: True rank of the signal component.
        noise_std: Standard deviation of the noise component.
        ranks: List of ranks to test. Defaults to [1, 2, 4, 8, 16, 32, 64].
        seed: Random seed.
    
    Returns:
        List of dicts with 'rank', 'energy_pct', 'lora_params', 'compression_ratio'.
    """
    if ranks is None:
        ranks = [1, 2, 4, 8, 16, 32, 64]

    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((d, true_rank)) @ rng.standard_normal((true_rank, d))
    dW = signal + noise_std * rng.standard_normal((d, d))

    sv = np.linalg.svd(dW, compute_uv=False)
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)

    results = []
    for r in ranks:
        idx = min(r, len(energy)) - 1
        results.append({
            "rank": r,
            "energy_pct": float(energy[idx] * 100),
            "lora_params": 2 * d * r,
            "full_params": d * d,
            "compression_ratio": (d * d) / (2 * d * r),
        })
    return results


def adapter_spectral_analysis(model):
    """
    INNOVATION: Analyze the singular value spectrum of each adapter's B @ A.
    
    Reveals how many of the r rank directions each adapter actually uses.
    If most singular values are near zero, the adapter is over-parameterized
    and could be pruned to a lower rank without loss.
    
    Args:
        model: Model with LoRALinear wrappers.
    
    Returns:
        List of dicts with layer name, singular values, and effective rank.
    """
    from .core import LoRALinear

    results = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta = module.B @ module.A  # [d_out, d_in]
                sv = torch.linalg.svdvals(delta).cpu().numpy()
                # Effective rank: number of SVs > 1% of the largest
                threshold = sv[0] * 0.01 if sv[0] > 0 else 0
                effective_rank = int(np.sum(sv > threshold))

            results.append({
                "layer": name,
                "singular_values": sv.tolist(),
                "effective_rank": effective_rank,
                "max_sv": float(sv[0]),
                "min_sv": float(sv[-1]),
                "sv_ratio": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
            })
    return results
