"""Synthetic optional archive fixtures; no real binary/network/GPU operations."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location("optional_turing_test", Path(__file__).resolve().parents[1] / "runtime/optional_turing.py")
turing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(turing)


def fixture():
    source = "a" * 40
    binary = b"fixture-not-a-native-binary"
    manifest = {"binary": {"filename": "flash_attn_turing.cp312-win_amd64.pyd", "sha256": hashlib.sha256(binary).hexdigest()}, "source_head": source, "source_url": turing.SOURCE_REPOSITORY, "cutlass_head": "b" * 40, "source_license": "Fixture: original upstream license unconfirmed; no blanket clearance"}
    files = {"MANIFEST.json": (json.dumps(manifest).encode(), "build_record"), "flash_attn_turing.cp312-win_amd64.pyd": (binary, "binary"), "flash_attention_interface.py": (b"def flash_attn_func(): pass\n", "interface"), "source/kernel.cu": (b"fixture source", "source"), "licenses/CUTLASS-LICENSE.txt": (b"fixture actual component license bytes", "license"), "NOTICE.txt": (b"fixture original notices", "notice")}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, (data, _) in files.items():
            archive.writestr(name, data)
    data = buffer.getvalue()
    record = {"status": "public", "abi": copy.deepcopy(turing.ABI), "distribution_authorization": "user_authorized_preserve_recorded_license_status", "source": {"repository": turing.SOURCE_REPOSITORY, "commit": source, "cutlass_commit": "b" * 40}, "license_status": "Fixture upstream license unconfirmed", "artifact": {"url": "https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/download/unit-fixture/turing.zip", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, "files": [{"path": name, "size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest(), "role": role} for name, (value, role) in files.items()]}
    return record, data


def report(sm=75):
    return {"platform": "win32", "machine": "AMD64", "python_version": "3.12.13", "cuda_build": "13.0", "cuda_available": True, "device": {"sm": sm}, "packages": {"torch": {"status": "available", "version": "2.10.0+cu130"}}}


class TuringTests(unittest.TestCase):
    def test_pending_artifact_never_plans_download(self):
        with patch.object(turing, "publication", return_value=None), patch.object(turing.urllib.request, "urlopen") as network:
            state, steps = turing.plan(report())
            self.assertEqual(state["status"], "publish_pending")
            self.assertFalse(steps)
            network.assert_not_called()

    def test_sm89_sm86_unknown_hosts_skip(self):
        for facts in [report(89), report(86), {**report(), "python_version": "3.13.1"}, {**report(), "machine": "ARM64"}, {**report(), "platform": "linux"}]:
            with patch.object(turing, "publication") as publication:
                self.assertEqual(turing.plan(facts)[0]["status"], "skipped")
                publication.assert_not_called()

    def test_fixed_manifest_requires_source_license_notice_build_inventory(self):
        record, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "optional_binary_manifest.json"
            with patch.object(turing, "MANIFEST", manifest):
                manifest.write_text(json.dumps({"schema_version": 1, "packages": {"turing_flash": record}}))
                self.assertEqual(turing.publication(), record)
                for role in ("source", "license", "notice", "build_record"):
                    missing = copy.deepcopy(record)
                    missing["files"] = [f for f in missing["files"] if f["role"] != role]
                    manifest.write_text(json.dumps({"schema_version": 1, "packages": {"turing_flash": missing}}))
                    with self.assertRaises(ValueError):
                        turing.publication()

    def test_unbounded_or_other_owner_urls_rejected(self):
        for url in ["file:///tmp/bundle.zip", "https://evil.example/bundle.zip", "https://huggingface.co/Andylinux/repo/resolve/main/bundle.zip", "https://github.com/other/project/releases/download/v1/bundle.zip"]:
            with self.assertRaises(ValueError):
                turing._url(url)

    def test_safe_atomic_install_and_idempotence(self):
        record, data = fixture()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "vendor/turing_bundle"
            with patch.object(turing, "TARGET", target), patch.object(turing, "publication", return_value=record), patch.object(turing, "validate_host"), patch.object(turing.urllib.request, "urlopen", return_value=io.BytesIO(data)) as network:
                self.assertTrue(turing.prepare())
                self.assertTrue((target / "dlls").is_dir())
                self.assertEqual(turing.inventory()["status"], "installed")
                self.assertFalse(turing.prepare())
                self.assertEqual(network.call_count, 1)
                (target / "NOTICE.txt").write_text("changed")
                self.assertEqual(turing.inventory()["status"], "conflict")
                with self.assertRaises(ValueError):
                    turing.prepare()
                self.assertEqual((target / "NOTICE.txt").read_text(), "changed")

    def test_bad_hash_or_traversal_never_creates_target(self):
        record, data = fixture()
        for unsafe in (False, True):
            changed = copy.deepcopy(record)
            if unsafe:
                buffer = io.BytesIO(data)
                with zipfile.ZipFile(buffer, "a") as archive:
                    archive.writestr("../escape", b"bad")
                payload = buffer.getvalue()
                changed["artifact"]["size_bytes"] = len(payload)
                changed["artifact"]["sha256"] = hashlib.sha256(payload).hexdigest()
            else:
                payload = b"bad zip bytes"
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "turing_bundle"
                with patch.object(turing, "TARGET", target), patch.object(turing, "publication", return_value=changed), patch.object(turing, "validate_host"), patch.object(turing.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                    with self.assertRaises((ValueError, RuntimeError)):
                        turing.prepare()
                    self.assertFalse(target.exists())

    def test_host_validation_is_exact_sm75(self):
        cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda index: (8, 9))
        torch = SimpleNamespace(version=SimpleNamespace(cuda="13.0"), cuda=cuda)
        with patch.object(turing.sys, "platform", "win32"), patch.object(turing.sys, "version_info", (3, 12)), patch.object(turing.platform, "machine", return_value="AMD64"), patch.object(turing.metadata, "version", return_value="2.10.0+cu130"), patch.object(turing.importlib, "import_module", return_value=torch):
            with self.assertRaisesRegex(RuntimeError, "SM75"):
                turing.validate_host()
            cuda.get_device_capability = lambda index: (7, 5)
            turing.validate_host()

    def test_native_import_is_same_interpreter_and_no_kernel(self):
        with patch.object(turing, "validate_host"), patch.object(turing.subprocess, "run") as run:
            turing.verify_import()
        command = run.call_args.args[0]
        self.assertEqual(command[:3], [turing.sys.executable, "-B", "-c"])
        self.assertIn("import flash_attn_turing", command[3])
        self.assertIn("callable(interface.flash_attn_func)", command[3])
        self.assertNotIn("interface.flash_attn_func(", command[3])


if __name__ == "__main__":
    unittest.main()
