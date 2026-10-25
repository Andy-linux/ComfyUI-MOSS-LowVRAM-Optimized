"""CPU semantic checks; full UI/GPU validation remains an integration gate."""
import importlib.util
from pathlib import Path
import sys
import unittest

VENDOR = Path(__file__).resolve().parents[1] / "runtime" / "vendor"
sys.path[:0] = [str(VENDOR), str(VENDOR / "kohaku_src")]
import paired_session

try:
    import torch
except ImportError:
    torch = None


class CompatibilityTests(unittest.TestCase):
    def test_old_config_defaults_and_text_keys(self):
        c = paired_session.SessionConfig("old.safetensors")
        self.assertEqual((c.te_mode, c.compute_dtype), ("auto", "fp16"))
        for key in ("layers.0.weight", "language_model.layers.0.weight",
                    "model.language_model.layers.0.weight", "model.layers.0.weight"):
            self.assertEqual(paired_session._key(key), "layers.0.weight")
        self.assertIsNone(paired_session._key("model.visual.layers.0.weight"))
        with self.assertRaises(KeyError):
            paired_session._key("unrecognized.weight")


@unittest.skipIf(torch is None, "torch is not installed in this review environment")
class NumericalTests(unittest.TestCase):
    def test_a8_rotated_matches_independent_integer_reference(self):
        from te_precision import a8_linear
        from kohakumoss_kernels.hadamard import group_hadamard_eager
        torch.manual_seed(123)
        x = torch.randn(3, 2, 64).half()
        q = torch.randint(-127, 128, (35, 64), dtype=torch.int8)
        s = torch.rand(35).float() / 30
        bias = torch.randn(35).half()
        # Independent CPU integer matmul: no floating dot rounding and no A16
        # fallback. Nonaligned M/N exercise padding and multidimensional restore.
        xr = group_hadamard_eager(x, 16).reshape(-1, 64).float()
        sx = xr.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
        aq = (xr / sx).round().clamp(-127, 127).to(torch.int32)
        want = ((aq @ q.int().t()).float() * sx * s + bias.float()).half().reshape(3, 2, 35)
        got = a8_linear(x, q, s, bias, 16)
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_zero_tokens_and_base_rebind_preserve_storage(self):
        from te_precision import a8_linear, configure_base, TEA8Weight
        from kohakumoss.quant.weights import QuantWeight
        q = torch.randint(-127, 128, (32, 64), dtype=torch.int8)
        scale = torch.rand(32)
        w = QuantWeight(q, scale, "int8", 16)
        before = (w.q.data_ptr(), w.scale.data_ptr(), w.rotation)
        configure_base(w, "w8a8")
        self.assertIsInstance(w, TEA8Weight)
        w.bind(torch.float16, "cpu")
        self.assertEqual(before, (w.q.data_ptr(), w.scale.data_ptr(), w.rotation))
        x = torch.zeros(1, 64, dtype=torch.float16)
        torch.testing.assert_close(w.linear(x, None), torch.zeros(1, 32).half(), rtol=0, atol=0)
        # Explicit W8A16 and auto retain the exact preexisting object and binder.
        original = QuantWeight(q, scale, "int8", 16)
        self.assertIs(configure_base(original, "w8a16"), original)

    def test_packed_and_decode_once_keep_a8_after_rebind(self):
        from te_precision import configure_base, TEA8Weight
        from kohakumoss.quant.w4a8 import W4A8Weight
        from kohakumoss.quant.bind import rebind_quant
        torch.manual_seed(9)
        q = torch.randint(-128, 128, (32, 32), dtype=torch.int8)
        rel = torch.ones(32, 4, dtype=torch.float8_e4m3fn)
        scale = torch.rand(32) / 20
        packed = configure_base(W4A8Weight(q, rel, scale, None, 16, 16), "w4a8")
        decoded = packed.to_int8()
        self.assertIsInstance(decoded, TEA8Weight)
        for base in (packed, decoded):
            rebind_quant(base, "cpu", torch.float16, autograd=False)
        x = torch.randn(2, 64).half()
        torch.testing.assert_close(packed.linear(x, None), decoded.linear(x, None), rtol=0, atol=0)
        self.assertEqual((packed._te_a8_calls, decoded._te_a8_calls), (1, 1))


if __name__ == "__main__":
    unittest.main()
