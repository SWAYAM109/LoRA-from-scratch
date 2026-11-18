#!/usr/bin/env python3
"""
Unit tests for the LoRA from Scratch implementation.

Run: python -m pytest tests/test_lora.py -v
Or:  python tests/test_lora.py
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lora_scratch.core import LoRALinear
from lora_scratch.analysis import count_parameters, rank_ablation_study, training_memory_estimate


class TestLoRALinear(unittest.TestCase):
    """Tests for the core LoRA module."""

    def setUp(self):
        torch.manual_seed(42)
        self.base = nn.Linear(64, 128)
        self.x = torch.randn(4, 64)

    def test_noop_at_init(self):
        """B=0 means the adapter starts as a no-op."""
        lora = LoRALinear(self.base, r=4, alpha=8)
        with torch.no_grad():
            base_out = self.base(self.x)
            lora_out = lora(self.x)
        self.assertTrue(torch.allclose(base_out, lora_out, atol=1e-6),
                        "LoRA should produce identical output at init (B=0)")

    def test_detour_activates(self):
        """After nudging B, adapter output should differ from base."""
        lora = LoRALinear(self.base, r=4, alpha=8)
        with torch.no_grad():
            lora.B.data += 0.1 * torch.randn_like(lora.B)
            base_out = self.base(self.x)
            lora_out = lora(self.x)
        self.assertFalse(torch.allclose(base_out, lora_out),
                         "After nudging B, LoRA output should differ")

    def test_base_frozen(self):
        """Base layer parameters should not require gradients."""
        lora = LoRALinear(self.base, r=4, alpha=8)
        for p in lora.base.parameters():
            self.assertFalse(p.requires_grad, "Base params should be frozen")

    def test_adapter_trainable(self):
        """A and B should require gradients."""
        lora = LoRALinear(self.base, r=4, alpha=8)
        self.assertTrue(lora.A.requires_grad, "A should be trainable")
        self.assertTrue(lora.B.requires_grad, "B should be trainable")

    def test_b_starts_zero(self):
        """B should be initialized to zeros."""
        lora = LoRALinear(self.base, r=4, alpha=8)
        self.assertTrue(torch.all(lora.B == 0), "B should start at zero")

    def test_a_nonzero(self):
        """A should be initialized to non-zero values."""
        lora = LoRALinear(self.base, r=4, alpha=8)
        self.assertFalse(torch.all(lora.A == 0), "A should not start at zero")

    def test_parameter_count(self):
        """Check trainable params = r * d_in + d_out * r."""
        r = 4
        lora = LoRALinear(self.base, r=r, alpha=8)
        trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        expected = r * 64 + 128 * r
        self.assertEqual(trainable, expected)

    def test_scaling(self):
        """Scale factor should be alpha / r."""
        lora = LoRALinear(self.base, r=4, alpha=16)
        self.assertAlmostEqual(lora.scale, 4.0)

    def test_different_ranks(self):
        """Different ranks should produce different parameter counts."""
        counts = []
        for r in [2, 4, 8, 16]:
            torch.manual_seed(42)
            base = nn.Linear(64, 128)
            lora = LoRALinear(base, r=r)
            trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
            counts.append(trainable)
        self.assertEqual(len(set(counts)), len(counts))

    def test_init_strategies(self):
        """All init strategies should produce valid A matrices."""
        for strategy in ["kaiming", "gaussian", "xavier"]:
            torch.manual_seed(42)
            base = nn.Linear(64, 128)
            lora = LoRALinear(base, r=4, init_strategy=strategy)
            self.assertFalse(torch.any(torch.isnan(lora.A)))
            self.assertFalse(torch.any(torch.isinf(lora.A)))

    def test_invalid_init_strategy(self):
        """Invalid strategy should raise ValueError."""
        with self.assertRaises(ValueError):
            LoRALinear(self.base, r=4, init_strategy="invalid")

    def test_effective_delta_w(self):
        """effective_delta_w should return (B @ A) * scale."""
        lora = LoRALinear(self.base, r=4, alpha=16)
        with torch.no_grad():
            lora.B.data = torch.randn_like(lora.B)
        delta = lora.effective_delta_w()
        expected = (lora.B @ lora.A) * lora.scale
        self.assertTrue(torch.allclose(delta, expected))

    def test_prune_rank(self):
        """Pruning should reduce rank and parameter count."""
        lora = LoRALinear(self.base, r=8, alpha=16)
        with torch.no_grad():
            lora.B.data = torch.randn_like(lora.B) * 0.1
        original_params = sum(p.numel() for p in [lora.A, lora.B])
        lora.prune_rank(4)
        pruned_params = sum(p.numel() for p in [lora.A, lora.B])
        self.assertLess(pruned_params, original_params)
        self.assertEqual(lora.r, 4)
        self.assertEqual(lora.A.shape[0], 4)
        self.assertEqual(lora.B.shape[1], 4)


class TestAnalysis(unittest.TestCase):
    """Tests for analysis utilities."""

    def test_rank_ablation(self):
        """Rank ablation should return increasing energy with rank."""
        results = rank_ablation_study(d=64, true_rank=4, ranks=[1, 2, 4, 8])
        energies = [r["energy_pct"] for r in results]
        self.assertEqual(energies, sorted(energies))
        self.assertGreater(energies[-1], 99)

    def test_memory_estimate_ordering(self):
        """QLoRA < LoRA < Full FT in memory."""
        n = 7e9
        full = training_memory_estimate(n, "full_ft")
        lora = training_memory_estimate(n, "lora")
        qlora = training_memory_estimate(n, "qlora")
        self.assertGreater(full, lora)
        self.assertGreater(lora, qlora)

    def test_memory_estimate_invalid(self):
        """Invalid method should raise ValueError."""
        with self.assertRaises(ValueError):
            training_memory_estimate(1e6, "invalid_method")


class TestQuantization(unittest.TestCase):
    """Tests for quantization schemes."""

    def test_absmax_roundtrip(self):
        """Absmax quantize->dequantize should approximately recover input."""
        from lora_scratch.quantization import quantize_absmax_4bit, dequantize_absmax
        w = torch.randn(128, 128)
        dq = dequantize_absmax(*quantize_absmax_4bit(w))
        rel_error = (w - dq).abs().mean() / w.abs().mean()
        self.assertLess(rel_error, 0.3)

    def test_absmax_shape_preserved(self):
        """Dequantized tensor should have same shape as input."""
        from lora_scratch.quantization import quantize_absmax_4bit, dequantize_absmax
        w = torch.randn(64, 128)
        dq = dequantize_absmax(*quantize_absmax_4bit(w))
        self.assertEqual(dq.shape, w.shape)

    def test_nf4_roundtrip(self):
        """NF4 quantize->dequantize should approximately recover input."""
        from lora_scratch.quantization import quantize_nf4, dequantize_nf4
        w = torch.randn(128, 128)
        dq = dequantize_nf4(*quantize_nf4(w))
        rel_error = (w - dq).abs().mean() / w.abs().mean()
        self.assertLess(rel_error, 0.25)

    def test_nf4_beats_absmax_for_normal(self):
        """NF4 should have lower error than absmax for normal-distributed weights."""
        from lora_scratch.quantization import (
            quantize_absmax_4bit, dequantize_absmax,
            quantize_nf4, dequantize_nf4,
        )
        torch.manual_seed(42)
        w = torch.randn(256, 256)

        dq_abs = dequantize_absmax(*quantize_absmax_4bit(w))
        err_abs = (w - dq_abs).abs().mean() / w.abs().mean()

        dq_nf4 = dequantize_nf4(*quantize_nf4(w))
        err_nf4 = (w - dq_nf4).abs().mean() / w.abs().mean()

        self.assertLessEqual(err_nf4, err_abs + 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
