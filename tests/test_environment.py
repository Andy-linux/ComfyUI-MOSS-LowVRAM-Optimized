import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "environment_under_test",
    Path(__file__).resolve().parents[1] / "runtime/environment.py",
)
env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env)


def report(sm=89, platform="win32"):
    packages = {n: {"status": "available", "version": "1.0"} for n in env.PACKAGES}
    packages["torch"]["version"] = "2.10.0+cu130"
    packages["triton"]["version"] = "3.2.0.post21" if sm == 75 else "3.6.0"
    packages["comfy_kitchen"]["version"] = "0.2.37"
    packages["transformers"]["version"] = "5.20.1"
    packages["diffusers"]["version"] = "0.41.0"
    packages["safetensors"]["version"] = "0.8.0"
    packages["einops"]["version"] = "0.8.2"
    return {
        "platform": platform,
        "python_version": "3.12.13",
        "packages": packages,
        "cuda_available": True,
        "cuda_build": "13.0",
        "device": {"sm": sm},
        "torch_flash_compiled": False,
        "optional_dependencies": {},
    }


class PolicyTests(unittest.TestCase):
    def codes(self, r, selection):
        return {e["code"] for e in env.evaluate_selection(r, selection)["errors"]}

    def test_reference_stack_and_linux(self):
        self.assertFalse(env.evaluate_selection(report(), {})["errors"])
        self.assertIn(
            "reference runtime", env.evaluate_selection(report(), {})["qualification"]
        )
        linux = env.evaluate_selection(report(platform="linux"), {})
        self.assertEqual(linux["qualification"], "unvalidated platform/build")
        self.assertIn(
            "platform_build_unvalidated", {w["code"] for w in linux["warnings"]}
        )

    def test_python_requires_312(self):
        r = report()
        r["python_version"] = "3.11.14"
        self.assertIn("python_version", self.codes(r, {}))
        r["python_version"] = "3.12.0"
        self.assertNotIn("python_version", self.codes(r, {}))

    def test_dependency_minimums_and_transformers_upper_bound(self):
        for name, version in (
            ("diffusers", "0.40.9"),
            ("diffusers", "0.42.0"),
            ("transformers", "5.20.0"),
            ("transformers", "6.0.0"),
            ("safetensors", "0.7.9"),
            ("einops", "0.8.1"),
            ("comfy_kitchen", "0.2.36"),
        ):
            r = report()
            r["packages"][name]["version"] = version
            self.assertIn("dependency_version_" + name, self.codes(r, {}))

    def test_unknown_dependency_version_warns(self):
        for version in (None, "unknown", "5.20.1rc1"):
            r = report()
            r["packages"]["transformers"]["version"] = version
            result = env.evaluate_selection(r, {})
            self.assertFalse(result["errors"])
            self.assertEqual(result["status"], "warning")
            self.assertIn(
                "dependency_version_unknown_transformers",
                {w["code"] for w in result["warnings"]},
            )

    def test_triton_source_actions(self):
        for sm in (75, 89):
            result = env.evaluate_selection(report(sm), {})
            self.assertTrue(
                any(
                    "https://github.com/triton-lang/triton-windows" in a
                    for a in result["actions"]
                )
            )

    def test_sm75_triton_and_defaults(self):
        r = report(75)
        self.assertFalse(self.codes(r, {"candidate": "q5_ridge_gptq", "profile": "4g"}))
        r["packages"]["triton"]["version"] = "3.3.0"
        self.assertIn("sm75_triton", self.codes(r, {}))

    def test_sm75_bf16_fields(self):
        for selection in (
            {"precision": "bf16"},
            {"te_compute_dtype": "bf16"},
            {"vae_precision": "bf16"},
        ):
            self.assertIn("bf16_architecture", self.codes(report(75), selection))

    def test_te_modes_and_fp8(self):
        for mode in ("w4a8", "w8a8"):
            self.assertIn(
                "te_a8_architecture", self.codes(report(75), {"te_mode": mode})
            )
        for mode in ("w8a16", "original"):
            self.assertNotIn(
                "te_a8_architecture", self.codes(report(75), {"te_mode": mode})
            )
        self.assertIn(
            "fp8_architecture", self.codes(report(75), {"candidate": "base_fp8"})
        )

    def test_stream_guard_matches_runtime(self):
        for sm in (75, 86, 120):
            self.assertIn(
                "payload_stream_architecture",
                self.codes(report(sm), {"resident_blocks": 2}),
            )
            for candidate in ("q6_direct", "w8_int8c", "base_w8_int8c", "base_fp8"):
                self.assertIn(
                    "payload_stream_architecture",
                    self.codes(report(sm), {"candidate": candidate, "profile": "4g"}),
                )
        self.assertFalse(
            self.codes(report(), {"profile": "4g", "candidate": "base_fp8"})
        )

    def test_later_architectures_warning_and_features_error(self):
        for sm, code in ((120, "sm120_unvalidated"), (86, "sm86_future")):
            result = env.evaluate_selection(report(sm), {})
            self.assertFalse(result["errors"])
            self.assertIn(code, {w["code"] for w in result["warnings"]})
            self.assertIn(
                "attention_architecture", self.codes(report(sm), {"attention": "cudnn"})
            )
            self.assertIn(
                "flash_architecture", self.codes(report(sm), {"attention": "flash"})
            )
            self.assertIn(
                "te_partial_architecture",
                self.codes(report(sm), {"te_memory": "partial"}),
            )

    def test_optional_missing_and_explicit_flash(self):
        self.assertIn("flash_missing", self.codes(report(), {"attention": "flash"}))
        self.assertIn(
            "sage_dependency", self.codes(report(), {"attention": "sage_fp16"})
        )
        self.assertIn(
            "turing_flash_missing", self.codes(report(75), {"attention": "flash"})
        )
        r = report()
        r["torch_flash_compiled"] = True
        self.assertNotIn("flash_missing", self.codes(r, {"attention": "flash"}))

    def test_policy_pure_and_dependencies(self):
        r = report()
        before = copy.deepcopy(r)
        with patch.object(
            env.importlib,
            "import_module",
            side_effect=AssertionError("pure policy imported a module"),
        ):
            env.evaluate_selection(r, {})
        self.assertEqual(r, before)
        r["packages"]["comfy_kitchen"]["status"] = "incompatible"
        self.assertIn("dependency_comfy_kitchen", self.codes(r, {}))
        r = report()
        r["packages"]["diffusers"]["status"] = "missing"
        self.assertIn("dependency_diffusers", self.codes(r, {}))


class InspectionTests(unittest.TestCase):
    def test_missing_torch_returns_json_without_exception(self):
        def missing(name):
            e = ModuleNotFoundError("missing " + name)
            e.name = name
            raise e

        with patch.object(env.importlib, "import_module", side_effect=missing):
            r = env.inspect_environment(check_extensions=True)
        import json

        json.dumps(r)
        self.assertEqual(r["packages"]["torch"]["status"], "missing")
        self.assertFalse(r["gpu_workload_run"])

    def test_triton_windows_distribution_fallback(self):
        def version(name):
            if name == "triton-windows":
                return "3.2.0.post21"
            raise env.metadata.PackageNotFoundError(name)

        with (
            patch.object(env.metadata, "version", side_effect=version),
            patch.object(
                env.importlib.util,
                "find_spec",
                return_value=type("Spec", (), {"origin": "triton/__init__.py"})(),
            ),
        ):
            p = env._package("triton", ("triton", "triton-windows"))
        self.assertEqual(
            (p["distribution"], p["version"], p["status"]),
            ("triton-windows", "3.2.0.post21", "available"),
        )


if __name__ == "__main__":
    unittest.main()
