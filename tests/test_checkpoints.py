"""CPU header checks need only stdlib; tensor checks run when torch is available."""
import importlib.util
import gc
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.checkpoints import inspect_checkpoint


class CheckpointHeaders(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "test.safetensors"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, header, data=b""):
        raw = json.dumps(header).encode()
        self.path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)

    def info(self, dtype, shape):
        return {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}

    def fp8(self, recipe="row"):
        shape = {"row": [128], "tensor": [1], "block": [1, 1], "mx": [128, 4]}[recipe]
        return {"__metadata__": {"kohakumoss_fp8": json.dumps({"recipe": recipe, "layers": ["layers.0.proj.q"]})},
                "layers.0.proj.q.weight": self.info("F8_E4M3", [128, 128]),
                "layers.0.proj.q.weight_scale": self.info("U8" if recipe == "mx" else "F32", shape)}

    def test_all_stored_recipes(self):
        for recipe in ("row", "tensor", "block", "mx"):
            self.write(self.fp8(recipe))
            result = inspect_checkpoint(self.path)
            self.assertEqual(result["recipe"], recipe)
            self.assertEqual(result["format"], "mxfp8" if recipe == "mx" else "fp8_compute")

    def test_bad_scale_rejected(self):
        header = self.fp8(); header["layers.0.proj.q.weight_scale"]["shape"] = [1]
        self.write(header)
        with self.assertRaisesRegex(ValueError, "scale/shape"):
            inspect_checkpoint(self.path)

    def test_unrecorded_codes_rejected(self):
        header = self.fp8(); header["layers.0.proj.k.weight"] = self.info("F8_E4M3", [128,128])
        self.write(header)
        with self.assertRaisesRegex(ValueError, "every quantized"):
            inspect_checkpoint(self.path)

    def test_int8_fused_and_separate_records(self):
        record = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()
        for part in ("qkv", "q"):
            name = "layers.0.proj." + part
            self.write({name + ".weight": self.info("I8", [768 if part == "qkv" else 256,256]),
                        name + ".weight_scale": self.info("F32", [768 if part == "qkv" else 256,1]),
                        name + ".comfy_quant": {"dtype":"U8", "shape":[len(record)], "data_offsets":[0,len(record)]}}, record)
            result = inspect_checkpoint(self.path)
            self.assertEqual(result["format"], "int8_convrot")
            self.assertEqual(result["fused_qkv"], part == "qkv")
            self.assertEqual(result["quant_layers"][name]["convrot_groupsize"], 256)

    def test_missing_rotation_record_not_guessed(self):
        self.write({"layers.0.proj.q.weight": self.info("I8", [256,256])})
        with self.assertRaisesRegex(ValueError, "explicit comfy_quant"):
            inspect_checkpoint(self.path)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"), "torch/safetensors required")
class TensorRoundtrip(unittest.TestCase):
    def test_fp8_compute_and_weight_only_export_reload(self):
        import torch
        from safetensors.torch import save_file
        from safetensors import safe_open
        from runtime.checkpoints import bootstrap_vendor, load_checkpoint
        from runtime.fp8_support import export_fp8, bind_fp8_inference
        bootstrap_vendor()
        from kohakumoss.model.build import build_backbone
        from kohakumoss.types.arch import MossArchConfig
        cfg = MossArchConfig(dim=256, depth=1, heads=4, ffn_dim=256, text_dim=256,
                             text_blocks=1, latent_channels=16, freq_dim=32, negative_skip_layers=())
        original = build_backbone(cfg, device="cpu", dtype=torch.bfloat16, init=True)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "bf16.safetensors"
            save_file(dict(original.state_dict()),str(source))
            for precision in ("fp8_compute", "fp8_weight_only", "mxfp8"):
                output = Path(folder) / (precision + ".safetensors")
                receipt = export_fp8(source,output,variant="base",precision=precision)
                self.assertFalse(receipt["training_performed"])
                loaded, _ = load_checkpoint(output,variant="base",cfg=cfg)
                bind_fp8_inference(loaded,dtype=torch.bfloat16,device="cpu",backend="eager")
                base = loaded.layers[0].proj.q.base
                with safe_open(str(output),framework="pt",device="cpu") as weights:
                    self.assertTrue(torch.equal(base.q.view(torch.uint8),weights.get_tensor("layers.0.proj.q.weight").view(torch.uint8)))
                    self.assertTrue(torch.equal(base.scale,weights.get_tensor("layers.0.proj.q.weight_scale")))
                x = torch.randn(2,256,dtype=torch.bfloat16)
                self.assertEqual(tuple(base.linear(x,None).shape),(2,256))
                with self.assertRaises(FileExistsError):
                    export_fp8(source,output,variant="base",precision=precision)
                del loaded, base, weights
                gc.collect()  # release safetensors mmap owners before Windows unlink

    def test_dense_base_source_and_fused_quant_codes(self):
        import torch
        from safetensors.torch import save_file
        from runtime.checkpoints import bootstrap_vendor, load_checkpoint
        bootstrap_vendor()
        from kohakumoss.model.build import build_backbone
        from kohakumoss.types.arch import MossArchConfig
        cfg = MossArchConfig(dim=256, depth=1, heads=4, ffn_dim=256, text_dim=256,
                             text_blocks=1, latent_channels=16, freq_dim=32, negative_skip_layers=())
        model = build_backbone(cfg, device="cpu", dtype=torch.bfloat16, init=True)
        state = dict(model.state_dict())
        for part in ("q", "k", "v"):
            state.pop("layers.0.proj." + part + ".weight")
        name = "layers.0.proj.qkv"
        codes = torch.arange(768*256).remainder(255).sub(127).to(torch.int8).reshape(768,256)
        scales = torch.arange(1,769,dtype=torch.float32).reshape(-1,1) / 10000
        state[name + ".weight"] = codes
        state[name + ".weight_scale"] = scales
        record = json.dumps({"format":"int8_tensorwise","convrot":True,"convrot_groupsize":256}).encode()
        state[name + ".comfy_quant"] = torch.tensor(list(record),dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "base.safetensors"; save_file(state,str(path))
            loaded, receipt = load_checkpoint(path, variant="base", cfg=cfg)
            self.assertTrue(torch.equal(loaded.video_embedder.weight, state["video_embedder.weight"]))
            for index, part in enumerate(("q","k","v")):
                base = loaded.get_submodule("layers.0.proj."+part).base
                self.assertTrue(torch.equal(base.codes,codes.chunk(3,0)[index]))
                self.assertTrue(torch.equal(base.scales,scales.chunk(3,0)[index]))
            self.assertEqual(len(receipt["w8a8_layers"]),3)
            del loaded, base
            gc.collect()  # no live mmap-backed tensors when TemporaryDirectory exits


if __name__ == "__main__":
    unittest.main()
