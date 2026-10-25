"""CPU-only configuration contracts: storage, arithmetic and legacy stability."""
import ast
import importlib.util
import json
import sys
import types
from unittest.mock import patch
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moss_catalog_fixture", ROOT / "runtime/catalog.py")
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_are_explicit_sft_w5_and_w8a16_te(self):
        args = catalog.resolve_configuration({})
        self.assertEqual(args["model"], "SFT W5")
        self.assertEqual(args["profile"], "6 GB")
        self.assertEqual(args["text_encoder"], "INT8 ConvRot · W8A16")
        self.assertEqual(args["dit_checkpoint"], "q5_ridge_gptq.safetensors")
        self.assertEqual(args["te_checkpoint"], "moss-image-qwen35-int8-convrot-v2.safetensors")
        self.assertEqual(catalog.asset_ids_for_selection({}), ["sft_w5", "te_int8", "vae"])

    def test_integer_compute_is_real_a8_for_all_integer_storage(self):
        for family, storage, candidate in (("SFT", "Packed W5", "q5_ridge_gptq"), ("SFT", "Packed W6", "q6_direct"), ("SFT", "INT8 ConvRot", "w8_int8c"), ("Base", "INT8 ConvRot", "base_w8_int8c")):
            selection = {"model_family": family, "dit_storage": storage}
            self.assertEqual(catalog.environment_selection(selection)["candidate"], candidate)
            for invalid in ("W8A16", "FP8", "16-bit"):
                with self.assertRaises(ValueError):
                    catalog.resolve_configuration({**selection, "dit_compute": invalid})

    def test_fp8_explicit_activation_modes_do_not_change_asset(self):
        for family in ("Base", "SFT"):
            for compute, mode in (("FP8", "fp8"), ("16-bit", "16bit")):
                selection = {"model_family": family, "dit_storage": "Kohaku FP8 row", "dit_compute": compute}
                self.assertEqual(catalog.resolve_configuration(selection)["fp8_activation_mode"], mode)
                self.assertEqual(catalog.asset_ids_for_selection(selection)[0], family.lower() + "_fp8")

    def test_structurally_invalid_pairs_fail_before_model_load(self):
        for selection in ({"model_family": "Base"}, {"te_compute": "W4A8"}, {"te_storage": "Original float"}, {"model_family": "other"}, {"resident_blocks": True}):
            with self.assertRaises(ValueError):
                catalog.environment_selection(selection)

    def test_text_assets_and_compute_are_independent(self):
        selection = {"te_storage": "Comfy INT8 ConvRot", "te_compute": "W8A8"}
        result = catalog.resolve_configuration(selection)
        self.assertEqual(result["te_checkpoint"], "qwen3.5_9b_int8_convrot.safetensors")
        self.assertEqual(result["text_encoder"], "INT8 ConvRot · W8A8")
        self.assertEqual(catalog.asset_ids_for_selection(selection)[1], "te_int8_comfy")
        with self.assertRaises(ValueError):
            catalog.resolve_configuration({**selection, "te_checkpoint": "auto"})
        for compute, label in (("W4A16 (legacy)", "W4A8 GPTQ"), ("W4A8", "W4 GPTQ · W4A8")):
            self.assertEqual(catalog.resolve_configuration({"te_storage": "Comfy W4 GPTQ", "te_compute": compute})["text_encoder"], label)
        for compute in ("FP16", "BF16"):
            args = catalog.resolve_configuration({"te_storage": "Original float", "te_compute": compute})
            self.assertEqual(args["text_encoder"], "Original " + compute)
            self.assertEqual(args["te_checkpoint"], "MOSS-Image-Base/text_encoder")

    def test_recommended_checkpoint_from_different_variant_is_rejected(self):
        for selection in ({"dit_checkpoint": "q6_direct.safetensors"},
                          {"te_checkpoint": "qwen3.5_9b_int8_convrot.safetensors"},
                          {"vae_checkpoint": "flux2-vae_small.safetensors"}):
            with self.assertRaisesRegex(ValueError, "different catalog"):
                catalog.resolve_configuration(selection)
        self.assertEqual(catalog.resolve_configuration({"dit_checkpoint": "my-custom.safetensors"})["dit_checkpoint"], "my-custom.safetensors")

    def test_sampling_recommendation_does_not_enter_loader_kwargs(self):
        selected = {"model_family": "Base", "dit_storage": "INT8 ConvRot"}
        self.assertEqual(catalog.configuration_capabilities(selected)["recommended_sampling"]["steps"], 50)
        self.assertEqual(catalog.configuration_capabilities({})["recommended_sampling"]["steps"], 16)
        self.assertFalse({"seed", "steps", "positive", "negative"} & catalog.resolve_configuration(selected).keys())

    def test_catalog_checkpoints_are_available_without_local_files(self):
        dit = catalog.catalog_checkpoint_choices("diffusion_models")
        te = catalog.catalog_checkpoint_choices("text_encoders")
        vae = catalog.catalog_checkpoint_choices("vae")
        self.assertTrue({"q5_ridge_gptq.safetensors", "q6_direct.safetensors", "moss_image_base_kohaku_fp8_row.safetensors", "moss_image_sft_kohaku_fp8_row.safetensors", "moss-image-base-dit-int8-convrot.safetensors", "moss_image_sft_int8_convrot.safetensors"} <= set(dit))
        self.assertTrue({"moss-image-qwen35-int8-convrot-v2.safetensors", "qwen3.5_9b_int8_convrot.safetensors", "qwen3.5_9b_w4a8_gptq.safetensors", "MOSS-Image-Base/text_encoder"} <= set(te))
        self.assertFalse(any("model-0000" in name for name in te))
        self.assertEqual(set(vae), {"moss-image-vae.safetensors", "flux2-vae_small.safetensors"})
        self.assertNotIn("moss_image_base_int8_convrot.safetensors", dit)

    def test_new_node_delegates_and_uses_authoritative_policy(self):
        prefix = "moss_configuration_node_fixture"
        package = types.ModuleType(prefix)
        package.__path__ = [str(ROOT)]
        runtime = types.ModuleType(prefix + ".runtime")
        runtime.__path__ = [str(ROOT / "runtime")]
        nodes = types.ModuleType(prefix + ".nodes")
        class Loader:
            RETURN_TYPES = ("MOSS_SESSION",)
            def load_session(self, **kwargs):
                self.received = kwargs
                return ("session sentinel",)
        Loader.INPUT_TYPES = classmethod(lambda cls: {"required": {k: ([], {}) for k in ("qkv_backend", "ffn_tile", "host_arena")}, "optional": {}})
        nodes.MOSSOptimizedSessionLoader = Loader
        service = types.ModuleType(prefix + ".runtime.environment_service")
        seen = []
        service.check_selection = lambda selection: seen.append(selection) or {"evaluation": {"errors": [{"message": "GPU policy block"}]}}
        service.validation_message = lambda evaluation: evaluation["errors"][0]["message"]
        assets = types.ModuleType(prefix + ".runtime.assets")
        assets.checkpoint_choices = lambda category: ["auto"]
        setup = types.ModuleType(prefix + ".runtime.setup_service")
        setup.default_service = lambda: types.SimpleNamespace(preparation=types.SimpleNamespace(runtime_checkpoints=lambda values: {}))
        modules = {prefix + ".runtime.setup_service": setup, prefix + ".runtime.assets": assets, prefix: package, prefix + ".runtime": runtime, prefix + ".nodes": nodes, prefix + ".runtime.catalog": catalog, prefix + ".runtime.environment_service": service}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(prefix + ".configuration", ROOT / "configuration.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls = module.MOSSOptimizedConfiguration
            schema = cls.INPUT_TYPES()
            for field, category in (("dit_checkpoint", "diffusion_models"), ("te_checkpoint", "text_encoders"), ("vae_checkpoint", "vae")):
                self.assertTrue(set(catalog.catalog_checkpoint_choices(category)) <= set(schema["optional"][field][0]))
            instance = cls()
            self.assertEqual(instance.configure(), ("session sentinel",))
            self.assertEqual(instance.received, catalog.resolve_configuration({}))
            self.assertEqual(cls.VALIDATE_INPUTS(), "GPU policy block")
            self.assertEqual(seen[0], catalog.environment_selection({}))
            self.assertIn("Invalid MOSS configuration", cls.VALIDATE_INPUTS(dit_compute="W8A16"))
            self.assertEqual(len(seen), 1)  # Illegal arithmetic never reaches policy/model loading.

    def test_only_main_workflow_is_published_and_history_is_excluded(self):
        self.assertEqual([p.name for p in (ROOT / "example_workflows").glob("*.json")], ["00_moss_lowvram.json"])
        self.assertEqual(len(list((ROOT / "tests/historical_workflows").glob("*.json"))), 7)
        ignored = (ROOT / ".comfyignore").read_text().splitlines()
        self.assertTrue("tests/" in ignored or "/tests/" in ignored)

    def test_main_api_and_ui_workflows_agree_and_keep_old_workflow(self):
        api = json.loads((ROOT / "tests/integration_workflows/00_moss_lowvram.json").read_text())
        ui = json.loads((ROOT / "example_workflows/00_moss_lowvram.json").read_text())
        node = next(n for n in ui["nodes"] if n["id"] == 1)
        self.assertEqual(node["type"], "MOSSOptimizedConfiguration")
        self.assertEqual(len(node["widgets_values"]), 24)
        self.assertEqual(catalog.resolve_configuration(api["1"]["inputs"]), catalog.resolve_configuration({}))
        self.assertEqual(api["4"]["inputs"]["steps"], 16)
        self.assertEqual(api["3"]["inputs"]["width"], 1024)
        old = json.loads((ROOT / "tests/historical_workflows/06_sft_w5_lowvram_1024.json").read_text())
        old_loader = next(n for n in old["nodes"] if n["type"] == "MOSSOptimizedSessionLoader")
        self.assertEqual(len(old_loader["widgets_values"]), 20)
        tree = ast.parse((ROOT / "nodes.py").read_text())
        legacy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MOSSOptimizedSessionLoader")
        schema = next(n for n in legacy.body if isinstance(n, ast.FunctionDef) and n.name == "INPUT_TYPES")
        self.assertNotIn("fp8_activation_mode", ast.unparse(schema))


if __name__ == "__main__":
    unittest.main()
