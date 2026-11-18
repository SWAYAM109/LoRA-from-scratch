"""
LoRA from Scratch — Parameter-Efficient Fine-Tuning of LLMs

A from-scratch implementation of LoRA (Low-Rank Adaptation) and QLoRA,
built entirely in PyTorch without the peft library. Demonstrates how to
freeze a pretrained model, inject low-rank adapters, train them, and
deploy efficiently.

Author: SWAYAM109
"""

__version__ = "1.0.0"

from .core import LoRALinear
from .config import AdapterConfig
from .injection import add_lora, remove_lora
from .merge import merge_lora, unmerge_lora, swap_adapter
from .quantization import (
    quantize_absmax_4bit, dequantize_absmax,
    quantize_nf4, dequantize_nf4,
    QuantizedProjection, quantize_model_weights,
)
from .data import load_ledgar_task, encode_clause
from .analysis import (
    count_parameters,
    training_memory_estimate,
    rank_ablation_study,
    adapter_spectral_analysis,
)
