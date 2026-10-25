import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("install_dependencies_test", Path(__file__).resolve().parents[1] / "runtime/install_dependencies.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def report(sm=89, platform="win32", installed=False):
    version = "3.2.0.post21" if sm == 75 else "3.6.0.post26"
    return {"platform": platform, "machine": "AMD64" if platform == "win32" else "x86_64", "python_version": "3.12.10", "cuda_available": True, "cuda_build": "13.0", "device": {"sm": sm},
            "packages": {"torch": {"status": "available", "version": "2.10.0+cu130"}, "triton": {"status": "available" if installed else "missing", "version": version if installed else None, "distribution": "triton-windows" if installed else None}},
            "sage_installation": {"status": "conflict", "reason": "test fixture excludes optional downloads"},
            "install_prerequisites": {"python_headers": True, "python_import_library": True, "header_root": "/fake/python", "embedded_python": True, "c_compiler": True, "cuda_compiler": True, "compiler_override": None}}


class InstallPlanTests(unittest.TestCase):
    def test_plan_scope_is_not_runtime_readiness(self):
        facts = report(installed=True)
        facts["install_prerequisites"]["compiler_override"] = "existing-custom-compiler.exe"
        plan = installer.dependency_plan(facts)
        self.assertEqual(plan["status"], "unsupported")
        self.assertEqual(plan["scope"], "automatic_dependency_preparation")
        self.assertEqual(plan["runtime_readiness"], "see_environment_evaluation")
        self.assertNotIn("runtime_blocked", plan)

    def test_windows_pins(self):
        for sm, version in [(89, "3.6.0.post26"), (75, "3.2.0.post21")]:
            plan = installer.dependency_plan(report(sm))
            self.assertEqual(plan["status"], "action_required")
            self.assertEqual(plan["steps"][0]["requirement"], "triton-windows==" + version)
        self.assertEqual(installer.dependency_plan(report(75))["warnings"][0]["code"], "historical_sm75")

    def test_supported_installed_is_idempotent(self):
        plan = installer.dependency_plan(report(installed=True))
        self.assertEqual(plan["status"], "ready")
        with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run") as run:
            self.assertEqual(installer.execute_plan(plan)["exit_code"], 0)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][:2], [sys.executable, "-c"])

    def test_missing_portable_headers_are_pinned(self):
        facts = report()
        facts["install_prerequisites"]["python_headers"] = False
        plan = installer.dependency_plan(facts)
        self.assertEqual([s["kind"] for s in plan["steps"]], ["python_headers", "pip"])
        self.assertEqual(len(plan["steps"][0]["sha256"]), 64)
        facts["install_prerequisites"]["embedded_python"] = False
        self.assertEqual(installer.dependency_plan(facts)["status"], "unsupported")

    def test_unknown_hosts_never_install(self):
        variants = [("python_version", "3.13.2"), ("machine", "ARM64"), ("cuda_build", "12.8"), ("platform", "darwin"), ("cuda_available", False)]
        for key, value in variants:
            facts = report()
            facts[key] = value
            plan = installer.dependency_plan(facts)
            self.assertEqual(plan["status"], "unsupported", key)
            self.assertFalse(plan["steps"])
            with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run") as run:
                self.assertEqual(installer.execute_plan(plan)["exit_code"], 1)
                run.assert_not_called()
        for sm in (70, 86, 120):
            self.assertEqual(installer.dependency_plan(report(sm))["status"], "unsupported")

    def test_torch_version_never_changed(self):
        facts = report()
        facts["packages"]["torch"]["version"] = "2.9.0+cu130"
        self.assertEqual(installer.dependency_plan(facts)["status"], "unsupported")

    def test_existing_wrong_or_broken_triton_not_replaced(self):
        for key, value in [("version", "3.8.0.post29"), ("distribution", "triton"), ("status", "incompatible"), ("shadow_conflict", True)]:
            facts = report(installed=True)
            facts["packages"]["triton"][key] = value
            self.assertFalse(installer.dependency_plan(facts)["steps"])
            self.assertEqual(installer.dependency_plan(facts)["status"], "unsupported")

    def test_missing_installed_toolchain_and_override(self):
        for key, value in [("c_compiler", False), ("cuda_compiler", False), ("compiler_override", "custom.exe")]:
            facts = report(installed=True)
            facts["install_prerequisites"][key] = value
            self.assertEqual(installer.dependency_plan(facts)["status"], "unsupported")

    def test_exact_interpreter_no_dependencies_and_verification(self):
        plan = installer.dependency_plan(report())
        ready = installer.dependency_plan(report(installed=True))
        with patch.object(installer, "dependency_plan", side_effect=[plan, ready]), patch.object(installer.subprocess, "run") as run:
            result = installer.execute_plan(plan)
        self.assertEqual(result["status"], "prepared")
        self.assertFalse(result["read_only"])
        self.assertTrue(result["restart_required"])
        command = run.call_args_list[0].args[0]
        self.assertEqual(command, [sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", "--index-url", "https://pypi.org/simple", "triton-windows==3.6.0.post26"])
        self.assertNotIn("--upgrade", command)

    def test_failing_pip_is_failure(self):
        plan = installer.dependency_plan(report())
        with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["pip"])):
            self.assertEqual(installer.execute_plan(plan)["exit_code"], 1)

    def test_changed_plan_not_executed(self):
        plan = installer.dependency_plan(report())
        forged = copy.deepcopy(plan)
        forged["steps"][0]["requirement"] = "torch==2.11.0"
        with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run") as run:
            self.assertEqual(installer.execute_plan(forged)["status"], "plan_changed")
            run.assert_not_called()

    def test_linux_uses_exact_torch_metadata_only(self):
        facts = report(platform="linux")
        facts["torch_requires"] = ['triton==3.6.0; platform_system == "Linux" and platform_machine == "x86_64"']
        plan = installer.dependency_plan(facts)
        self.assertEqual(plan["steps"][0]["requirement"], "triton==3.6.0")
        for requirements in [[], ["triton>=3.6"], ['triton==3.6.0; python_version < "3.10"']]:
            facts["torch_requires"] = requirements
            self.assertEqual(installer.dependency_plan(facts)["status"], "unsupported")
        facts["torch_requires"] = ["triton==3.6.0"]
        facts["device"]["sm"] = 75
        self.assertEqual(installer.dependency_plan(facts)["status"], "unsupported")

    def test_import_does_not_execute_installer_or_network(self):
        fresh_spec = importlib.util.spec_from_file_location("moss_installer_import_probe", spec.origin)
        fresh = importlib.util.module_from_spec(fresh_spec)
        with patch.object(subprocess, "run") as run, patch.object(installer.urllib.request, "urlopen") as network:
            fresh_spec.loader.exec_module(fresh)
            run.assert_not_called()
            network.assert_not_called()

    def test_bad_header_download_is_rejected_before_writes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "python312._pth").touch()
            with patch.object(installer.sys, "platform", "win32"), patch.object(installer.sys, "version_info", (3, 12, 10)), patch.object(installer.sys, "base_prefix", temporary), patch.object(installer.urllib.request, "urlopen") as network:
                network.return_value.__enter__.return_value.read.return_value = b"untrusted"
                with self.assertRaisesRegex(RuntimeError, "hash"):
                    installer._install_headers()
                self.assertFalse((root / "include").exists())


    def test_header_archive_scoped_and_no_overwrite(self):
        import hashlib
        import io
        import tempfile
        import zipfile
        for unsafe in (False, True):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("include/Python.h", "new-header")
                archive.writestr("libs/python312.lib", "import-library")
                if unsafe:
                    archive.writestr("include/../../escape", "bad")
            data = buffer.getvalue()
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "python312._pth").touch()
                (root / "include").mkdir()
                (root / "include/Python.h").write_text("keep-existing")
                with patch.object(installer.sys, "platform", "win32"), patch.object(installer.sys, "version_info", (3, 12, 10)), patch.object(installer.sys, "base_prefix", temporary), patch.object(installer, "HEADERS_SHA256", hashlib.sha256(data).hexdigest()), patch.object(installer.urllib.request, "urlopen") as network:
                    network.return_value.__enter__.return_value.read.return_value = data
                    if unsafe:
                        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                            installer._install_headers()
                        self.assertFalse((root / "libs").exists())
                    else:
                        installer._install_headers()
                        self.assertEqual((root / "libs/python312.lib").read_text(), "import-library")
                    self.assertEqual((root / "include/Python.h").read_text(), "keep-existing")

    def test_native_import_failure_is_not_ready(self):
        plan = installer.dependency_plan(report(installed=True))
        with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run", side_effect=subprocess.CalledProcessError(1, [sys.executable, "-c"])):
            result = installer.execute_plan(plan)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["exit_code"], 1)


    def test_sage_only_for_qualified_sm89_and_not_required_for_auto(self):
        facts = report()
        facts["sage_installation"] = {"status": "missing"}
        plan = installer.dependency_plan(facts)
        self.assertEqual(plan["optional_steps"][0]["kind"], "optional_sage")
        self.assertEqual(plan["optional_steps"][0]["bytes"], 16656067)
        self.assertEqual(plan["optional_steps"][0]["sha256"], installer.SAGE_SHA256)
        self.assertFalse(plan["optional_dependencies"]["sageattention"]["required_for_auto"])
        for sm, platform in [(75, "win32"), (86, "win32"), (120, "win32"), (89, "linux")]:
            facts = report(sm=sm, platform=platform)
            facts["sage_installation"] = {"status": "missing"}
            if platform == "linux":
                facts["torch_requires"] = ["triton==3.6.0"]
            plan = installer.dependency_plan(facts)
            self.assertFalse(any(step["kind"] == "optional_sage" for step in plan["optional_steps"]))
            self.assertEqual(plan["optional_dependencies"]["sageattention"]["status"], "skipped")

    def test_sage_optional_failure_preserves_auto_success(self):
        facts = report(installed=True)
        facts["sage_installation"] = {"status": "missing"}
        plan = installer.dependency_plan(facts)
        with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run"), patch.object(installer, "_prepare_sage", side_effect=RuntimeError("hash mismatch")):
            result = installer.execute_plan(plan)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["optional_dependencies"]["sageattention"]["status"], "failed")
        self.assertIn("optional_sage_preparation_failed", [w["code"] for w in result["warnings"]])

    def test_sage_success_and_idempotent_reuse(self):
        facts = report(installed=True)
        for state in ("missing", "installed"):
            facts["sage_installation"] = {"status": state}
            plan = installer.dependency_plan(facts)
            with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run"), patch.object(installer, "_prepare_sage") as prepare, patch.object(installer, "_verify_sage") as verify:
                result = installer.execute_plan(plan)
            self.assertEqual(prepare.call_count, 1 if state == "missing" else 0)
            verify.assert_called_once()
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["optional_dependencies"]["sageattention"]["status"], "available")

    def test_sage_native_import_failure_not_claimed_available(self):
        facts = report(installed=True)
        facts["sage_installation"] = {"status": "installed"}
        plan = installer.dependency_plan(facts)
        with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run"), patch.object(installer, "_verify_sage", side_effect=subprocess.CalledProcessError(1, ["python"])):
            result = installer.execute_plan(plan)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["optional_dependencies"]["sageattention"]["status"], "failed")

    def test_sage_conflict_never_downloads_or_overwrites(self):
        plan = installer.dependency_plan(report(installed=True))
        with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run"), patch.object(installer, "_prepare_sage") as prepare, patch.object(installer, "_verify_sage") as verify:
            self.assertEqual(installer.execute_plan(plan)["exit_code"], 0)
            prepare.assert_not_called()
            verify.assert_not_called()

    def test_sage_bad_download_rejected_before_helper(self):
        with patch.object(installer.urllib.request, "urlopen") as network, patch.object(installer.subprocess, "run") as run:
            network.return_value.__enter__.return_value.read.return_value = b"bad-download"
            with self.assertRaisesRegex(RuntimeError, "size/hash"):
                installer._prepare_sage()
            run.assert_not_called()

    def test_sage_provenance_inventory_and_missing_member(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sageattn220"
            with patch.object(installer, "SAGE_ROOT", root):
                self.assertEqual(installer._sage_state()["status"], "missing")
                root.mkdir()
                self.assertEqual(installer._sage_state()["status"], "conflict")
                expected = {"url": installer.SAGE_URL, "sha256": installer.SAGE_SHA256, "wheel_bytes": installer.SAGE_BYTES, "torch": "2.10.0+cu130", "scope": "Optional SM89 attention; no global dependency changes"}
                (root / "SAGE_WHEEL_PROVENANCE.json").write_text(json.dumps(expected))
                for name in installer.SAGE_MEMBERS:
                    file = root / name
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.touch()
                self.assertEqual(installer._sage_state()["status"], "installed")
                (root / installer.SAGE_MEMBERS[-1]).unlink()
                self.assertEqual(installer._sage_state()["status"], "conflict")

    def test_sage_step_injection_is_rejected(self):
        facts = report(installed=True)
        facts["sage_installation"] = {"status": "missing"}
        plan = installer.dependency_plan(facts)
        forged = copy.deepcopy(plan)
        forged["optional_steps"][0]["url"] = "https://untrusted.example/wheel"
        with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run") as run:
            self.assertEqual(installer.execute_plan(forged)["status"], "plan_changed")
            run.assert_not_called()


    def test_sage_download_reuses_helper_and_same_python(self):
        import hashlib
        from types import SimpleNamespace
        from unittest.mock import Mock
        data = b"verified-wheel-fixture"
        helper = SimpleNamespace(URL=installer.SAGE_URL, SHA256=hashlib.sha256(data).hexdigest(), BYTES=len(data), prepare=Mock())
        seen = []
        def run(command, **kwargs):
            self.assertEqual(command[0], sys.executable)
            helper_path = Path(command[1])
            self.assertEqual(helper_path.name, "install_optional_sage.py")
            self.assertEqual(helper_path.parent.name, "tools")
            self.assertEqual(command[2], "--wheel")
            wheel = Path(command[3])
            self.assertEqual(wheel.read_bytes(), data)
            self.assertTrue(kwargs["check"])
            seen.append(wheel)
        with patch.object(installer, "SAGE_BYTES", len(data)), patch.object(installer, "SAGE_SHA256", helper.SHA256), patch.object(installer.urllib.request, "urlopen") as network, patch.object(installer.importlib.util, "spec_from_file_location") as loader, patch.object(installer.importlib.util, "module_from_spec", return_value=helper), patch.object(installer.subprocess, "run", side_effect=run), patch.object(installer, "_sage_state", return_value={"status": "installed"}):
            network.return_value.__enter__.return_value.read.return_value = data
            installer._prepare_sage()
            loader.return_value.loader.exec_module.assert_called_once_with(helper)
            helper.prepare.assert_called_once()
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0].exists())


    def test_turing_optional_failure_keeps_auto_ready(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        plan = installer.dependency_plan(report(sm=75, installed=True))
        plan["optional_dependencies"]["turing_flash"] = {"status": "installed"}
        helper = SimpleNamespace(verify_import=Mock(side_effect=RuntimeError("native ABI failure")))
        with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run"), patch.object(installer, "_turing_helper", return_value=helper):
            result = installer.execute_plan(plan)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["optional_dependencies"]["turing_flash"]["status"], "failed")
        self.assertIn("optional_turing_preparation_failed", [w["code"] for w in result["warnings"]])

    def test_turing_optional_prepare_success_is_separate_from_sage(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        plan = installer.dependency_plan(report(sm=75, installed=True))
        plan["optional_dependencies"]["turing_flash"] = {"status": "missing"}
        plan["optional_steps"] = [{"id": "turing_flash", "kind": "optional_turing"}]
        helper = SimpleNamespace(prepare=Mock(return_value=True), verify_import=Mock())
        with patch.object(installer, "dependency_plan", return_value=plan), patch.object(installer.subprocess, "run"), patch.object(installer, "_turing_helper", return_value=helper), patch.object(installer, "_prepare_sage") as sage:
            result = installer.execute_plan(plan)
        helper.prepare.assert_called_once()
        helper.verify_import.assert_called_once()
        sage.assert_not_called()
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["read_only"])
        self.assertTrue(result["restart_required"])
        self.assertEqual(result["optional_dependencies"]["turing_flash"]["status"], "available")
        self.assertFalse(result["optional_steps"])


if __name__ == "__main__":
    unittest.main()
