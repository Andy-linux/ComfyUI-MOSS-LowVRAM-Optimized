import hashlib, importlib.util, io, tempfile, unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

P = Path(__file__).resolve().parents[1] / "tools/download_models.py"
spec = importlib.util.spec_from_file_location("model_download", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DownloadTests(unittest.TestCase):
    def row(self, data):
        return {
            "repo_id": "owner/repo",
            "revision": "a" * 40,
            "filename": "weights.safetensors",
            "destination": "diffusion_models/test.safetensors",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def test_verified_atomic_publish_and_reuse(self):
        data = b"bounded fixture"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row = self.row(data)
            with patch.object(m, "urlopen", return_value=io.BytesIO(data)):
                self.assertEqual(m.install(row, root)["status"], "DOWNLOADED_VERIFIED")
            with patch.object(
                m, "urlopen", side_effect=AssertionError("No redownload")
            ):
                self.assertEqual(m.install(row, root)["status"], "VERIFIED_EXISTING")
            self.assertFalse(list(root.rglob("*.partial")))

    def test_hash_failure_never_publishes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            row = self.row(b"good")
            with patch.object(m, "urlopen", return_value=io.BytesIO(b"evil")):
                with self.assertRaises(ValueError):
                    m.install(row, root)
            self.assertFalse((root / row["destination"]).exists())
            self.assertFalse(list(root.rglob("*.partial")))

    def test_verify_only_no_network_or_directory_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "absent"
            with patch.object(m, "urlopen", side_effect=AssertionError("No network")):
                self.assertEqual(
                    m.install(self.row(b"x"), root, verify_only=True)["status"],
                    "MISSING_OR_MISMATCH",
                )
            self.assertFalse(root.exists())

    def test_missing_metadata_and_traversal_rejected(self):
        for key, value in [
            ("revision", "main"),
            ("sha256", None),
            ("destination", "../escape"),
        ]:
            row = self.row(b"x")
            row[key] = value
            with self.assertRaises(ValueError):
                m.checked_file(row, Path("/tmp/models"))

    def test_windows_native_rendering_accepts_posix_manifest(self):
        # Exercise Windows separators without depending on this test host OS.
        class WindowsPath(PureWindowsPath):
            def resolve(self):
                return self

        root = WindowsPath("C:/ComfyUI/models")
        with patch.object(m, "Path", WindowsPath):
            target, url = m.checked_file(self.row(b"x"), root)
        self.assertEqual(
            target, WindowsPath("C:/ComfyUI/models/diffusion_models/test.safetensors")
        )
        self.assertIn("\\", str(target))  # Native rendering must not invalidate input.
        self.assertTrue(url.startswith("https://huggingface.co/owner/repo/resolve/"))

    def test_raw_windows_paths_and_aliases_are_rejected(self):
        for destination in (
            "diffusion_models\\test.safetensors",
            "C:/escape",
            "C:escape",
            "//server/share/file",
            "/absolute",
            "a/../escape",
            "a/./file",
            "a//file",
            "",
        ):
            row = self.row(b"x")
            row["destination"] = destination
            with self.subTest(destination=destination):
                with self.assertRaises(ValueError):
                    m.checked_file(row, Path("/tmp/models"))


if __name__ == "__main__":
    unittest.main()
