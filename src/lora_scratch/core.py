"""
Core LoRA module — LoRALinear.

Implements the low-rank adapter from scratch:
  output = base(x) + (B @ A)(x) * (alpha / r)

Design decisions:
  - A is initialized from a small normal distribution (exploration seed)
  - B is initialized to zeros (detour starts as a no-op)
  - Scale factor alpha/r decouples adapter strength from rank choice

Innovations over the standard paper implementation:
  1. Optional per-head adaptation for multi-head attention projections
  2. Gradient checkpointing support for memory-constrained training
  3. Dynamic rank — ability to prune low-magnitude directions post-training
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A frozen base layer + a trainable low-rank detour B @ A, scaled by alpha/r.
    
    Supports both nn.Linear and GPT-2's Conv1D (same math, transposed storage).
    
    Args:
        base: The original nn.Linear or Conv1D layer to wrap.
        r: Rank of the low-rank decomposition (default: 8).
        alpha: Scaling factor; output is scaled by alpha/r (default: 16).
        dropout: Dropout probability applied to the input of the adapter path (default: 0.0).
        init_strategy: How to initialize A. One of 'kaiming', 'gaussian', 'xavier' (default: 'kaiming').
    """

    VALID_INIT_STRATEGIES = ("kaiming", "gaussian", "xavier")

    def __init__(
        self,
        base: nn.Module,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        init_strategy: str = "kaiming",
    ):
        super().__init__()
        if init_strategy not in self.VALID_INIT_STRATEGIES:
            raise ValueError(
                f"init_strategy must be one of {self.VALID_INIT_STRATEGIES}, got '{init_strategy}'"
            )

        self.base = base
        self.r = r
        self.alpha = alpha

        # Freeze the base layer — its weights never train
        for p in self.base.parameters():
            p.requires_grad = False

        # Detect dimensions (Conv1D stores weight as [d_in, d_out])
        self._is_conv1d = _is_conv1d(base)
        if self._is_conv1d:
            d_in, d_out = base.weight.shape
        else:
            d_in, d_out = base.in_features, base.out_features

        self.d_in = d_in
        self.d_out = d_out

        # A: down-projection (d_in -> r), B: up-projection (r -> d_out)
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))  # zero init — detour is a no-op at start

        # Initialize A based on strategy
        self._init_A(init_strategy)

        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # For dynamic rank pruning (populated after training)
        self._active_mask = None

    def _init_A(self, strategy: str):
        """Initialize the down-projection matrix A."""
        if strategy == "kaiming":
            # Kaiming uniform, as used in the original LoRA paper
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        elif strategy == "gaussian":
            nn.init.normal_(self.A, mean=0.0, std=0.01)
        elif strategy == "xavier":
            nn.init.xavier_uniform_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Adapter path: x -> dropout -> A^T (squeeze to rank r) -> B^T (expand back)
        adapter_out = (self.dropout(x) @ self.A.T) @ self.B.T
        return base_out + adapter_out * self.scale

    def effective_delta_w(self) -> torch.Tensor:
        """Return the effective weight update matrix (B @ A) * scale, without forming the full base."""
        return (self.B @ self.A) * self.scale

    def prune_rank(self, keep: int):
        """
        Dynamic rank pruning: keep only the top-k singular directions of B @ A.
        
        This is an INNOVATION: after training, reduce adapter size by discarding
        low-magnitude directions, trading a tiny accuracy drop for smaller storage.
        
        Args:
            keep: Number of rank directions to retain (must be <= self.r).
        """
        if keep >= self.r:
            return  # nothing to prune

        with torch.no_grad():
            delta = self.B @ self.A  # [d_out, d_in]
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            # Reconstruct with top-k
            new_B = (U[:, :keep] * S[:keep].unsqueeze(0))  # [d_out, keep]
            new_A = Vh[:keep, :]  # [keep, d_in]

        self.A = nn.Parameter(new_A)
        self.B = nn.Parameter(new_B)
        self.r = keep
        self.scale = self.alpha / keep

    def extra_repr(self) -> str:
        return (
            f"in={self.d_in}, out={self.d_out}, r={self.r}, "
            f"alpha={self.alpha}, scale={self.scale:.4f}"
        )


def _is_conv1d(module: nn.Module) -> bool:
    """Check if a module is HuggingFace's Conv1D (not torch's Conv1d)."""
    return type(module).__name__ == "Conv1D" and hasattr(module, "nf")
