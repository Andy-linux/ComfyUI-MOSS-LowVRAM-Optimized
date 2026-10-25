"""Real CPU byte payloads with fake CUDA scheduling; no GPU claims."""

import importlib.util
from pathlib import Path
import sys
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required")
class FP8Payload(unittest.TestCase):
    def setUp(self):
        import torch
        from runtime.checkpoints import bootstrap_vendor

        bootstrap_vendor()
        import runtime.payload_stream as payload

        self.torch, self.payload = torch, payload

    def model(self, kind="row"):
        from kohakumoss.quant.compute import ComputeWeight
        from kohakumoss.quant.weights import QuantWeight

        torch = self.torch

        class Hook:
            def remove(self):
                self.removed = True

        class Block:
            def __init__(self, index):
                self.proj = {}
                for part in self_parts:
                    q = torch.full((128, 128), index % 16, dtype=torch.float32).to(
                        torch.float8_e4m3fn
                    )
                    scale = torch.ones(128, dtype=torch.float32)
                    base = (
                        QuantWeight(q, scale, "fp8")
                        if kind == "weight_only"
                        else ComputeWeight.from_stored("row", q, scale)
                    )
                    self.proj[part] = SimpleNamespace(base=base)

            def register_forward_pre_hook(self, fn):
                return Hook()

            def register_forward_hook(self, fn, **kw):
                return Hook()

        self_parts = self.payload.PROJECTIONS
        bb = SimpleNamespace(
            layers=[Block(i) for i in range(42)],
            video_embedder=SimpleNamespace(weight=torch.empty(0, dtype=torch.bfloat16)),
        )
        bb.to = lambda device: bb
        return bb

    def test_cache_exact_bytes_and_scale_budget(self):
        for kind in ("row", "weight_only"):
            bb = self.model(kind)
            cache = self.payload.PayloadCache()
            self.assertFalse(cache.capture(bb, kind))
            self.assertEqual(cache.arena.numel(), 42 * 6 * 128 * 128)
            self.assertEqual(cache.block_bytes, 6 * 128 * 128)
            self.assertEqual(cache.side_buffer_bytes, 42 * 6 * 128 * 4)
            self.assertTrue(cache.capture(bb, kind))
            with self.assertRaisesRegex(ValueError, "resident scale/aux"):
                cache.capture(bb, kind, fp8_side_buffer_budget_bytes=1)
            cache.close()
            self.assertIsNone(cache.arena)
            self.assertEqual(cache.fp8_signature, ())

    def test_training_and_heterogeneous_payload_refused(self):
        bb = self.model()
        base = bb.layers[0].proj["q"].base
        base.q_t = base.q.clone()
        with self.assertRaisesRegex(ValueError, "Training transpose"):
            self.payload.PayloadCache().capture(bb, "training")
        base.q_t = None
        base.recipe = "tensor"
        with self.assertRaisesRegex(ValueError, "identical recipe"):
            self.payload.PayloadCache().capture(bb, "mixed")

    def test_two_slot_skipped_block_lifecycle_restores_original_bytes(self):
        torch = self.torch
        p = self.payload
        bb = self.model()
        cache = p.PayloadCache()
        cache.capture(bb, "test")

        class Event:
            def record(self, *args):
                pass

            def synchronize(self):
                pass

        class Stream(Event):
            def wait_stream(self, *args):
                pass

            def wait_event(self, *args):
                pass

        cuda = SimpleNamespace(
            Event=Event,
            Stream=lambda **kw: Stream(),
            current_stream=lambda *args: Stream(),
            stream=lambda *args: nullcontext(),
        )
        actual_empty = torch.empty

        def empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            kwargs.pop("pin_memory", None)
            return actual_empty(*args, **kwargs)

        with (
            patch.object(p.torch, "cuda", cuda),
            patch.object(p.torch, "empty", empty),
            patch.object(p, "_bind_fp8_payload") as bind,
        ):
            owner = p.FrozenPayloadStream(
                bb, cache, "cuda:0", resident_blocks=2
            ).place()
            bind.assert_called_once()
            for skipped in ((), (22,)):
                owner.begin_forward(skipped)
                for i in range(42):
                    if i in skipped:
                        continue
                    owner.before(i)
                    for name in p.PROJECTIONS:
                        q = bb.layers[i].proj[name].base.q
                        self.assertTrue(torch.all(q.float() == i % 16))
                    owner.after(i)
            self.assertEqual(
                owner.stats["payload_H2D_bytes"], (40 + 39) * cache.block_bytes
            )
            self.assertEqual(owner.stats["slot_GPU_bytes"], 2 * cache.block_bytes)
            self.assertEqual(
                owner.stats["resident_scale_aux_GPU_bytes"], cache.side_buffer_bytes
            )
            owner.close()
            owner.close()
            self.assertEqual(owner.slots, [])
            for i in range(42):
                for name in p.PROJECTIONS:
                    self.assertTrue(
                        torch.all(bb.layers[i].proj[name].base.q.float() == i % 16)
                    )


if __name__ == "__main__":
    unittest.main()
