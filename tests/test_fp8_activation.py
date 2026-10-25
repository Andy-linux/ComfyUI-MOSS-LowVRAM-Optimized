"""Shared row storage activation selection; CPU reference is not GPU qualification."""

import gc
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.session import RuntimeConfig


class ActivationConfig(unittest.TestCase):
    def test_default_and_config_identity(self):
        config = RuntimeConfig(candidate="base_fp8")
        self.assertEqual(config.fp8_activation_mode, "checkpoint")
        self.assertNotEqual(config, RuntimeConfig(candidate="base_fp8", fp8_activation_mode="16bit"))
        self.assertNotEqual(config, RuntimeConfig(candidate="base_fp8", fp8_activation_mode="fp8"))

    def test_bad_mode_rejected_without_importing_tensor_runtime(self):
        from runtime.fp8_support import select_fp8_activation

        with self.assertRaisesRegex(ValueError, "fp8_activation_mode"):
            select_fp8_activation(None, mode="auto", checkpoint={}, dtype=None)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required; tensor checks not executed")
class SharedFP8Activation(unittest.TestCase):
    def setUp(self):
        import torch
        from runtime.checkpoints import bootstrap_vendor

        bootstrap_vendor()
        from kohakumoss.model.linear import MossLinear
        from kohakumoss.quant.compute import ComputeWeight
        from kohakumoss.quant.weights import QuantWeight
        from runtime.fp8_support import select_fp8_activation

        self.torch = torch
        self.Linear, self.Compute, self.Quant = MossLinear, ComputeWeight, QuantWeight
        self.select = select_fp8_activation
        self.info = {"format": "fp8_compute", "recipe": "row"}

    def model(self, count=2):
        torch = self.torch
        model = torch.nn.ModuleList()
        for i in range(count):
            lin = self.Linear(128, 128, bias=False, device="meta", dtype=torch.bfloat16)
            del lin.weight
            lin.__dict__["_held"] = torch.empty(0, dtype=torch.bfloat16)
            q = (torch.arange(128 * 128).reshape(128, 128) % 23 - 11).float().to(torch.float8_e4m3fn)
            scale = torch.linspace(0.001, 0.1, 128, dtype=torch.float32)
            lin.set_base(self.Compute.from_stored("row", q, scale))
            model.append(lin)
        return model

    def test_a16_replaces_registered_base_with_identical_storage_and_values(self):
        torch = self.torch
        model = self.model()
        originals = [(lin.base.q, lin.base.scale) for lin in model]
        snapshots = [(q.view(torch.uint8).clone(), scale.clone()) for q, scale in originals]
        old_bases = [weakref.ref(lin.base) for lin in model]
        # Any quantization or materialization attempt is a regression.
        with (patch("kohakumoss.quant.weights.quantize", side_effect=AssertionError("requantization")),
              patch("kohakumoss.quant.compute.quantize_pair", side_effect=AssertionError("requantization")),
              patch.object(self.Compute, "weight", side_effect=AssertionError("dense copy")),
              patch.object(self.Compute, "materialize", side_effect=AssertionError("dense copy"))):
            receipt = self.select(model, mode="16bit", checkpoint=self.info, dtype=torch.bfloat16)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in old_bases))
        for lin, (q, scale), (q_bytes, scale_values) in zip(model, originals, snapshots):
            self.assertIsInstance(lin.base, self.Quant)
            self.assertIs(lin._modules["base"], lin.base)
            self.assertIs(lin.base.q, q)
            self.assertIs(lin.base.scale, scale)
            self.assertEqual(lin.base.rotation, 0)
            self.assertTrue(torch.equal(q.view(torch.uint8), q_bytes))
            self.assertTrue(torch.equal(scale, scale_values))
            self.assertEqual(set(dict(lin.base.named_buffers())), {"q", "scale"})
            self.assertEqual(list(lin.parameters()), [])
        self.assertEqual(receipt["effective_mode"], "16bit")
        self.assertFalse(receipt["requantized"])
        self.assertFalse(receipt["persistent_dense_copy"])

    def test_a16_reference_from_exact_same_codes_and_scales(self):
        torch = self.torch
        from runtime.fp8_support import bind_fp8_inference

        model = self.model(count=1)
        q, scale = model[0].base.q, model[0].base.scale
        self.select(model, mode="16bit", checkpoint=self.info, dtype=torch.bfloat16)
        bind_fp8_inference(model, dtype=torch.bfloat16, device="cpu", backend="eager")
        x = torch.linspace(-1, 1, 3 * 128).reshape(3, 128).to(torch.bfloat16)
        expected_weight = (q.float() * scale[:, None]).to(x.dtype)
        with torch.inference_mode():
            expected = torch.nn.functional.linear(x, expected_weight)
            actual = model[0](x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_validation_is_atomic_for_invalid_second_layer(self):
        torch = self.torch
        for invalid in ("scale_shape", "scale_dtype", "q_dtype", "q_shape", "recipe", "transpose", "aux"):
            with self.subTest(invalid=invalid):
                model = self.model()
                before = [lin.base for lin in model]
                base = model[1].base
                if invalid == "scale_shape":
                    base.scale = torch.ones(1)
                elif invalid == "scale_dtype":
                    base.scale = base.scale.to(torch.bfloat16)
                elif invalid == "q_dtype":
                    base.q = base.q.float()
                elif invalid == "q_shape":
                    base.q = base.q.reshape(-1)
                elif invalid == "recipe":
                    base.recipe = "tensor"
                elif invalid == "transpose":
                    base.q_t = base.q.t()
                else:
                    base.aux = torch.zeros(1)
                with self.assertRaises(ValueError):
                    self.select(model, mode="16bit", checkpoint=self.info, dtype=torch.bfloat16)
                for lin, original in zip(model, before):
                    self.assertIs(lin.base, original)

    def test_defaults_and_explicit_fp8_preserve_compute_base(self):
        torch = self.torch
        for mode in ("checkpoint", "fp8"):
            model = self.model()
            before = [lin.base for lin in model]
            receipt = self.select(model, mode=mode, checkpoint=self.info, dtype=torch.bfloat16)
            self.assertEqual(receipt["effective_mode"], "fp8")
            for lin, original in zip(model, before):
                self.assertIs(lin.base, original)

    def test_legacy_weight_only_default_and_fp8_rejection(self):
        torch = self.torch
        model = self.model()
        for lin in model:
            lin.set_base(self.Quant(lin.base.q, lin.base.scale, "fp8"))
        info = {"format": "fp8_weight_only", "recipe": None}
        before = [lin.base for lin in model]
        for mode in ("checkpoint", "16bit"):
            receipt = self.select(model, mode=mode, checkpoint=info, dtype=torch.bfloat16)
            self.assertEqual(receipt["effective_mode"], "16bit")
        with self.assertRaisesRegex(ValueError, "stored row compute"):
            self.select(model, mode="fp8", checkpoint=info, dtype=torch.bfloat16)
        for lin, original in zip(model, before):
            self.assertIs(lin.base, original)

    def test_other_recipes_keep_checkpoint_policy_but_reject_explicit_selection(self):
        torch = self.torch
        for recipe in ("tensor", "block", "mx"):
            model = self.model(count=1)
            base = model[0].base
            scale = torch.ones({"tensor": (1,), "block": (1, 1), "mx": (128, 4)}[recipe],
                               dtype=torch.uint8 if recipe == "mx" else torch.float32)
            # Construct directly to avoid MX swizzling, irrelevant to mode selection.
            model[0].set_base(self.Compute(recipe, {"q": base.q, "scale": scale}))
            info = {"format": "mxfp8" if recipe == "mx" else "fp8_compute", "recipe": recipe}
            before = model[0].base
            self.select(model, checkpoint=info, dtype=torch.bfloat16)
            for mode in ("fp8", "16bit"):
                with self.assertRaisesRegex(ValueError, "shared row"):
                    self.select(model, mode=mode, checkpoint=info, dtype=torch.bfloat16)
                self.assertIs(model[0].base, before)


if __name__ == "__main__":
    unittest.main()
