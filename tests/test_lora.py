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


class TestAdapterConfig(unittest.TestCase):
    """Tests for AdapterConfig serialisation."""

    def test_defaults(self):
        from lora_scratch.config import AdapterConfig
        cfg = AdapterConfig()
        self.assertEqual(cfg.rank, 8)
        self.assertEqual(cfg.alpha, 16.0)
        self.assertEqual(cfg.quantization, "none")

    def test_save_load_roundtrip(self):
        import tempfile, os
        from lora_scratch.config import AdapterConfig
        cfg = AdapterConfig(rank=16, alpha=32.0, accuracy=0.958, notes="test run")
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.save(tmpdir)
            loaded = AdapterConfig.load(tmpdir)
        self.assertEqual(loaded.rank, 16)
        self.assertAlmostEqual(loaded.alpha, 32.0)
        self.assertAlmostEqual(loaded.accuracy, 0.958)
        self.assertEqual(loaded.notes, "test run")

    def test_from_dict_ignores_unknown_keys(self):
        from lora_scratch.config import AdapterConfig
        cfg = AdapterConfig.from_dict({"rank": 4, "unknown_key": "ignored"})
        self.assertEqual(cfg.rank, 4)

    def test_load_missing_file_raises(self):
        from lora_scratch.config import AdapterConfig
        with self.assertRaises(FileNotFoundError):
            AdapterConfig.load("/nonexistent/path")


class TestQuantizedProjection(unittest.TestCase):
    """Tests for real nibble-packed QuantizedProjection."""

    def test_absmax_forward_shape(self):
        from lora_scratch.quantization import QuantizedProjection
        base = nn.Linear(64, 128)
        qp = QuantizedProjection(base, block_size=64, scheme="absmax")
        x = torch.randn(4, 64)
        out = qp(x)
        self.assertEqual(out.shape, (4, 128))

    def test_nf4_forward_shape(self):
        from lora_scratch.quantization import QuantizedProjection
        base = nn.Linear(64, 128)
        qp = QuantizedProjection(base, block_size=64, scheme="nf4")
        x = torch.randn(4, 64)
        out = qp(x)
        self.assertEqual(out.shape, (4, 128))

    def test_packed_storage_smaller(self):
        """packed_weight should use fewer bytes than fp32 weight."""
        from lora_scratch.quantization import QuantizedProjection
        base = nn.Linear(256, 512)
        qp = QuantizedProjection(base, scheme="absmax")
        packed_bytes = qp.packed_weight.numel() * qp.packed_weight.element_size()
        fp32_bytes = base.weight.numel() * 4
        self.assertLess(packed_bytes, fp32_bytes // 2)

    def test_invalid_scheme_raises(self):
        from lora_scratch.quantization import QuantizedProjection
        with self.assertRaises(ValueError):
            QuantizedProjection(nn.Linear(32, 64), scheme="invalid")

    def test_merge_into_quantized_raises(self):
        """Merging a LoRA adapter into a quantized base should raise TypeError."""
        from lora_scratch.quantization import QuantizedProjection
        from lora_scratch.core import LoRALinear
        from lora_scratch.merge import _merge_single
        qp = QuantizedProjection(nn.Linear(32, 64), scheme="absmax")
        lora = LoRALinear(nn.Linear(32, 64), r=4)
        # Manually point lora.base to the quantized projection
        lora.base = qp
        with self.assertRaises(TypeError):
            _merge_single(lora)

    def test_lora_rank_validation(self):
        """Rank < 1 should raise ValueError."""
        with self.assertRaises(ValueError):
            LoRALinear(nn.Linear(32, 64), r=0)

    def test_lora_alpha_validation(self):
        """Alpha <= 0 should raise ValueError."""
        with self.assertRaises(ValueError):
            LoRALinear(nn.Linear(32, 64), alpha=-1)

    def test_lora_dropout_validation(self):
        """Dropout >= 1 should raise ValueError."""
        with self.assertRaises(ValueError):
            LoRALinear(nn.Linear(32, 64), dropout=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
