"""
AdapterConfig — serialisable dataclass that fully describes a LoRA adapter.

Inspired by lora-forge's config system. Saving the config alongside the
adapter weights means anyone can reconstruct the exact setup needed to
load and use the adapter without reading the training code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AdapterConfig:
    """All hyperparameters needed to reproduce a LoRA adapter.

    Save alongside adapter weights so the adapter is self-describing.
    """

    # Model
    base_model: str = "distilgpt2"
    task: str = "ledgar-classification"

    # LoRA
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    init_strategy: str = "kaiming"           # kaiming | gaussian | xavier
    target_modules: list[str] = field(default_factory=lambda: ["attn.c_attn"])

    # Quantization
    quantization: str = "none"               # none | absmax | nf4
    quant_block_size: int = 64

    # Data
    max_length: int = 96
    top_k_labels: int = 6
    label_names: list[str] = field(default_factory=list)
    label_tokens: list[str] = field(default_factory=list)

    # Training
    seed: int = 0
    epochs: int = 6
    learning_rate: float = 1e-3
    batch_size: int = 8
    warmup_ratio: float = 0.05

    # Metadata
    final_loss: float | None = None
    accuracy: float | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    adapter_size_bytes: int | None = None
    notes: str = ""

    # ------------------------------------------------------------------ #
    #  Serialisation
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdapterConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in allowed})

    @classmethod
    def load(cls, directory: str | Path) -> "AdapterConfig":
        path = Path(directory) / "adapter_config.json"
        if not path.exists():
            raise FileNotFoundError(f"Adapter config not found: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "adapter_config.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    def summary(self) -> str:
        """Human-readable one-liner."""
        pct = (
            f"{100 * self.trainable_params / self.total_params:.2f}%"
            if self.trainable_params and self.total_params
            else "?"
        )
        return (
            f"LoRA r={self.rank} alpha={self.alpha} targets={self.target_modules} "
            f"quant={self.quantization}  |  trainable={pct}  "
            f"acc={self.accuracy:.1%}" if self.accuracy else
            f"LoRA r={self.rank} alpha={self.alpha} targets={self.target_modules} "
            f"quant={self.quantization}  |  trainable={pct}"
        )
