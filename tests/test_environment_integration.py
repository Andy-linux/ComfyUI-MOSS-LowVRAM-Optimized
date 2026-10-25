"""No plugin registration, CUDA, server socket or torch required."""

import asyncio
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "moss_environment_integration_fixture"
for name, directory in ((PREFIX, ROOT), (PREFIX + ".runtime", ROOT / "runtime")):
    module = types.ModuleType(name)
    module.__path__ = [str(directory)]
    sys.modules[name] = module


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


env = load(PREFIX + ".runtime.environment", "runtime/environment.py")
service = load(
    PREFIX + ".runtime.environment_service", "runtime/environment_service.py"
)
assets = load(PREFIX + ".runtime.assets", "runtime/assets.py")
comfy = types.ModuleType("comfy")
samplers = types.ModuleType("comfy.samplers")
samplers.KSampler = types.SimpleNamespace(
    SAMPLERS=["euler"], SCHEDULERS=["sgm_uniform"]
)
comfy.samplers = samplers
# Import the actual module with only its external Comfy dependency stubbed.
# The isolated namespace package above skips plugin __init__/route registration.
with patch.dict(sys.modules, {"comfy": comfy, "comfy.samplers": samplers}):
    nodes_module = load(PREFIX + ".nodes", "nodes.py")
namespace = vars(nodes_module)


class ServiceTests(unittest.TestCase):
    def tearDown(self):
        service.current_environment.cache_clear()

    def test_cache_refresh_and_pure_evaluation(self):
        with (
            patch.object(service, "inspect_environment", return_value={}) as inspect,
            patch.object(service, "evaluate_selection", return_value={}) as evaluate,
        ):
            service.check_selection({})
            service.check_selection({})
            self.assertEqual(inspect.call_count, 1)
            self.assertEqual(evaluate.call_count, 2)
            service.check_selection({}, refresh=True)
            self.assertEqual(inspect.call_count, 2)

    def test_route_valid_bad_json_bad_shape(self):
        handlers = {}

        class Routes:
            def post(self, path):
                def decorator(fn):
                    handlers[path] = fn
                    return fn

                return decorator

        server = types.SimpleNamespace(
            PromptServer=types.SimpleNamespace(
                instance=types.SimpleNamespace(routes=Routes())
            )
        )
        web = types.SimpleNamespace(
            json_response=lambda data, status=200: (status, data)
        )

        class Request:
            def __init__(self, data):
                self.data = data

            async def json(self):
                if isinstance(self.data, Exception):
                    raise self.data
                return self.data

        result = {"environment": {}, "evaluation": {"errors": []}}
        with patch.dict(
            sys.modules, {"server": server, "aiohttp": types.SimpleNamespace(web=web)}
        ):
            service.register_routes(namespace["environment_selection"])
        handler = handlers["/moss_lowvram/environment"]

        async def inline_thread(fn, *args, **kwargs):
            # Avoid an executor/socket wakeup in restricted test containers;
            # production route still uses asyncio.to_thread.
            return fn(*args, **kwargs)

        with (
            patch.object(service, "check_selection", return_value=result) as check,
            patch.object(service.asyncio, "to_thread", new=inline_thread),
        ):
            status, data = asyncio.run(
                handler(Request({"selection": {}, "refresh": True}))
            )
            self.assertEqual(status, 200)
            self.assertEqual(data, result)
            self.assertTrue(check.call_args.kwargs["refresh"])
        for data in (
            [],
            {"selection": []},
            json.JSONDecodeError("bad", "", 0),
            {"selection": {"precision": "nonsense"}},
        ):
            with patch.object(
                service,
                "check_selection",
                side_effect=AssertionError("bad request inspected environment"),
            ):
                self.assertEqual(asyncio.run(handler(Request(data)))[0], 400)

    def test_preflight_blocks_before_session_creation(self):
        cls = namespace["MOSSOptimizedSessionLoader"]
        failure = {
            "evaluation": {"errors": [{"message": "missing fixture dependency"}]}
        }
        with patch.object(service, "check_selection", return_value=failure):
            result = cls.VALIDATE_INPUTS("SFT W8", "8 GB", "INT8 ConvRot", "auto")
            self.assertIn("missing fixture dependency", result)
            namespace["MossSession"] = lambda *a: self.fail(
                "constructed session after failed preflight"
            )
            with self.assertRaisesRegex(RuntimeError, "missing fixture dependency"):
                cls().load_session(
                    "SFT W8", "8 GB", "INT8 ConvRot", "auto", "auto", "auto", True
                )
        self.assertIs(service.validation_message({"errors": []}), True)

    def test_mapper_rejects_invalid_custom_validated_inputs(self):
        mapper = namespace["environment_selection"]
        for values in (
            [1],
            {"resident_blocks": 1.5},
            {"resident_blocks": True},
            {"resident_blocks": 41},
            {"precision": "fp32"},
            {"vae_precision": "fp16"},
            {"te_memory": "bogus"},
        ):
            with self.assertRaises((ValueError, TypeError)):
                mapper(values)

    def test_twenty_input_order_and_internal_ids(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda c: [], get_folder_paths=lambda c: []
        )
        with patch.dict(sys.modules, {"folder_paths": folder_paths}):
            schema = namespace["MOSSOptimizedSessionLoader"].INPUT_TYPES()
        fields = list(schema["required"]) + list(schema["optional"])
        self.assertEqual(
            fields,
            [
                "model",
                "profile",
                "text_encoder",
                "attention",
                "qkv_backend",
                "ffn_tile",
                "host_arena",
                "adapter_cache",
                "conditioning_memory",
                "precision",
                "attention_head_group",
                "vae_precision",
                "te_memory",
                "vae_admission",
                "resident_blocks",
                "payload_pinning",
                "te_resident_gib",
                "dit_checkpoint",
                "te_checkpoint",
                "vae_checkpoint",
            ],
        )
        self.assertEqual(
            list(nodes_module.NODE_CLASS_MAPPINGS),
            [
                "MOSSOptimizedSessionLoader",
                "MOSSOptimizedPairedTextEncode",
                "MOSSOptimizedSampler",
                "MOSSOptimizedVAEDecode",
                "MOSSOptimizedReleaseSession",
                "MOSSOptimizedControls",
            ],
        )


class HFDiscoveryTests(unittest.TestCase):
    def test_v2_alias_has_priority_but_old_aliases_remain(self):
        names = assets.ASSETS["te_int8"][1]
        self.assertEqual(names[0], "moss-image-qwen35-int8-convrot-v2.safetensors")
        self.assertIn("moss-image-qwen35-int8-convrot.safetensors", names)
        self.assertIn("qwen3.5_9b_int8_convrot.safetensors", names)

    def test_directory_choices_deduplicated_and_resolvable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            te = root / "MOSS-Image-Base/text_encoder"
            te.mkdir(parents=True)
            (te / "model.safetensors.index.json").write_text("{}")
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "model-00001-of-00004.safetensors").write_bytes(b"fixture")
            folders = types.SimpleNamespace(
                get_filename_list=lambda c: [
                    "single.safetensors",
                    "single.safetensors",
                ],
                get_folder_paths=lambda c: [str(root), str(root)],
                get_full_path=lambda c, n: None,
            )
            with (
                patch.dict(sys.modules, {"folder_paths": folders}),
                patch.object(assets, "local_overrides", return_value={}),
            ):
                choices = assets.checkpoint_choices("text_encoders")
                self.assertEqual(
                    choices,
                    ["auto", "MOSS-Image-Base/text_encoder", "single.safetensors"],
                )
                self.assertEqual(
                    assets.resolve_asset("te_original", "MOSS-Image-Base/text_encoder"),
                    te.resolve(),
                )
                self.assertEqual(assets.resolve_asset("te_original"), te.resolve())
                self.assertNotIn(
                    "MOSS-Image-Base/text_encoder", assets.checkpoint_choices("vae")
                )


if __name__ == "__main__":
    unittest.main()
