"""Tiny pinned HF fixtures; no model/GPU download and no ComfyUI imports."""
import asyncio
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import catalog
from runtime.model_preparation import ModelPreparation, PreparationConflict, download_models
from runtime.setup_service import SetupService, _request_body, hardware_options


def environment(sm=89, missing=False):
    return {"python_version": "3.12.9", "platform": "win32", "machine": "AMD64",
            "cuda_available": True, "cuda_build": "13.0", "device": {"sm": sm},
            "packages": {name: {"status": "missing" if missing else "available", "version": version,
                                 "distribution": "triton-windows" if name == "triton" else name}
                         for name, version in {"torch": "2.10.0+cu130", "triton": "3.2.0.post21" if sm == 75 else "3.6.0.post26",
                                               "comfy_kitchen": "0.2.37", "transformers": "5.20.1", "diffusers": "0.41.0", "safetensors": "0.8.0", "accelerate": "1.15.0", "einops": "0.8.2"}.items()},
            "install_prerequisites": {"python_headers": True, "python_import_library": True,
                                       "c_compiler": True, "cuda_compiler": True},
            "torch_flash_compiled": True}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "models"
        self.selection = {"te_storage": "Original float", "te_compute": "FP16"}
        defaults = catalog.checkpoint_defaults(self.selection)
        self.payloads = {}
        def row(destination):
            data = ("fixture:" + destination).encode()
            self.payloads[destination] = data
            return {"repo_id": "owner/repo", "revision": "a" * 40,
                    "filename": destination, "destination": destination,
                    "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        te_names = ["text_encoder/config.json"] + [f"text_encoder/model-0000{i}-of-00004.safetensors" for i in range(1, 5)] + ["text_encoder/model.safetensors.index.json", "tokenizer/chat_template.jinja", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
        self.manifest = {"assets": [
            {"id": "sft_w5", "status": "public", "files": [row("diffusion_models/" + defaults["dit_checkpoint"])]},
            {"id": "te_original", "status": "public", "files": [row("text_encoders/MOSS-Image-Base/" + name) for name in te_names]},
            {"id": "vae", "status": "public", "files": [row("vae/" + defaults["vae_checkpoint"])]},
        ]}
        self.preparations = []
        self.prep = self.make()

    def tearDown(self):
        for prep in self.preparations:
            prep.close()
        self.temp.cleanup()

    def make(self, resolver=None, registered_roots=None):
        prep = ModelPreparation(self.manifest, self.root, resolver,
                                Path(self.temp.name) / "verified.json", max_jobs=3, registered_roots=registered_roots)
        self.preparations.append(prep)
        return prep

    def fetch(self, request, timeout=None):
        destination = request.full_url.split("/resolve/" + "a" * 40 + "/", 1)[1]
        return io.BytesIO(self.payloads[destination])

    def wait(self, job):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = self.prep.get(job["job_id"])
            if result["state"] not in ("queued", "running"):
                return result
            time.sleep(0.005)
        self.fail("fixture worker timed out")

    def test_existing_renamed_same_hash_reused_and_inference_uses_actual_file(self):
        extra = Path(self.temp.name) / "extra"
        location = extra / "nested" / "my-renamed-weights.safetensors"
        location.parent.mkdir(parents=True)
        row = self.manifest["assets"][0]["files"][0]
        location.write_bytes(self.payloads[row["destination"]])
        self.prep = self.make(registered_roots={"diffusion_models": [extra]})
        with patch.object(download_models, "verify", wraps=download_models.verify) as verify:
            plan = self.prep.plan(self.selection)
            self.assertEqual(plan[0]["status"], "candidate")
            verify.assert_not_called()  # Inspection is metadata only.
        with self.assertRaisesRegex(ValueError, "Prepare"):
            self.prep.runtime_checkpoints(self.selection)
        with patch.object(download_models, "urlopen", side_effect=self.fetch) as fetch:
            job = self.wait(self.prep.start(self.selection))
        self.assertEqual(job["state"], "ready", job.get("error"))
        self.assertFalse((self.root / row["destination"]).exists())
        self.assertFalse(any(row["destination"] in call.args[0].full_url for call in fetch.call_args_list))
        self.assertEqual(self.prep.plan(self.selection)[0]["status"], "ready")
        self.assertEqual(self.prep.runtime_checkpoints(self.selection)["dit_checkpoint"], str(location.absolute()))
        location.write_bytes(b"x" * len(self.payloads[row["destination"]]))
        self.assertNotEqual(self.prep.plan(self.selection)[0]["status"], "ready")

    def test_wrong_same_size_candidate_is_preserved_and_next_match_reused(self):
        extra = Path(self.temp.name) / "extra"
        extra.mkdir()
        row = self.manifest["assets"][0]["files"][0]
        wrong = extra / Path(row["destination"]).name
        right = extra / "renamed-right.safetensors"
        payload = self.payloads[row["destination"]]
        wrong.write_bytes(b"x" * len(payload))
        right.write_bytes(payload)
        self.prep = self.make(registered_roots={"diffusion_models": [extra]})
        with patch.object(download_models, "urlopen", side_effect=self.fetch):
            job = self.wait(self.prep.start(self.selection))
        self.assertEqual(job["state"], "ready", job.get("error"))
        self.assertEqual(wrong.read_bytes(), b"x" * len(payload))
        self.assertFalse((extra / ".moss_lowvram_rejected").exists())
        self.assertTrue(any("SHA256" in rejected["reason"] for rejected in self.prep.plan(self.selection)[0]["candidate_mismatches"]))
        self.assertEqual(self.prep.runtime_checkpoints(self.selection)["dit_checkpoint"], str(right.absolute()))

    def test_wrong_candidate_only_downloads_canonical_without_repairing_candidate(self):
        extra = Path(self.temp.name) / "extra"
        extra.mkdir()
        row = self.manifest["assets"][0]["files"][0]
        wrong = extra / "other-variant.safetensors"
        data = b"x" * row["size_bytes"]
        wrong.write_bytes(data)
        self.prep = self.make(registered_roots={"diffusion_models": [extra]})
        with patch.object(download_models, "urlopen", side_effect=self.fetch):
            job = self.wait(self.prep.start(self.selection))
        self.assertEqual(job["state"], "ready", job.get("error"))
        self.assertEqual(wrong.read_bytes(), data)
        self.assertEqual((self.root / row["destination"]).read_bytes(), self.payloads[row["destination"]])

    def test_original_te_reuses_one_complete_bundle_and_rejects_mixed_directories(self):
        from runtime.model_preparation import bundle_relative
        extra = Path(self.temp.name) / "extra-te"
        wrong, right = extra / "a-wrong", extra / "z-right"
        rows = self.manifest["assets"][1]["files"]
        for row in rows:
            for bundle in (wrong, right):
                path = bundle / bundle_relative(row)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.payloads[row["destination"]])
        config = wrong / "text_encoder/config.json"
        config.write_bytes(b"x" * config.stat().st_size)
        self.prep = self.make(registered_roots={"text_encoders": [extra]})
        with patch.object(download_models, "urlopen", side_effect=self.fetch) as fetch:
            job = self.wait(self.prep.start(self.selection))
        self.assertEqual(job["state"], "ready", job.get("error"))
        actual = self.prep.runtime_checkpoints(self.selection)
        self.assertEqual(actual["te_checkpoint"], str((right / "text_encoder").absolute()))
        self.assertFalse(any("text_encoders/" in call.args[0].full_url for call in fetch.call_args_list))
        self.assertEqual(config.read_bytes(), b"x" * config.stat().st_size)
        # Invalidate a companion: a partial collection may not become a ready TE.
        (right / "tokenizer/tokenizer.json").unlink()
        self.assertFalse(all(item["ready"] for item in self.prep.plan(self.selection) if item["asset_id"] == "te_original"))
        with self.assertRaisesRegex(ValueError, "Original TE"):
            self.prep.runtime_checkpoints(self.selection)

    def test_sumomo_upstream_name_without_v2_is_verified_and_reused(self):
        defaults = catalog.checkpoint_defaults({})
        destination = "text_encoders/" + defaults["te_checkpoint"]
        data = b"tiny Sumomo v2 fixture"
        self.payloads[destination] = data
        row = {"repo_id": "owner/repo", "revision": "a" * 40, "filename": destination,
               "destination": destination, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.manifest["assets"][1] = {"id": "te_int8", "status": "public", "files": [row]}
        self.selection = {}
        extra = Path(self.temp.name) / "extra-te"
        path = extra / "nested/moss-image-qwen35-int8-convrot.safetensors"
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        self.prep = self.make(registered_roots={"text_encoders": [extra]})
        with patch.object(download_models, "urlopen", side_effect=self.fetch):
            job = self.wait(self.prep.start(self.selection))
        self.assertEqual(job["state"], "ready", job.get("error"))
        self.assertFalse((self.root / destination).exists())
        self.assertEqual(self.prep.runtime_checkpoints({})["te_checkpoint"], str(path.absolute()))

    def test_original_te_all_nine_files_and_dedup(self):
        self.manifest["assets"][1]["files"].append(self.manifest["assets"][1]["files"][0])
        prep = self.make()
        plan = prep.inspect(self.selection)
        self.assertEqual(len(plan["files"]), 11)
        self.assertEqual(sum(f["asset_id"] == "te_original" for f in plan["files"]), 9)
        self.assertFalse(plan["ready"])
        self.assertEqual(plan["total_bytes"], sum(map(len, self.payloads.values())))
        self.assertEqual(plan["missing_bytes"], plan["total_bytes"])
        self.assertFalse(self.root.exists())

    def test_download_full_sha_then_ready_and_persistent_cache_reuse(self):
        phases = []
        original_install = download_models.install
        def install(row, root, progress=None, **kwargs):
            def wrapped(event):
                phases.append(event["phase"])
                if event["phase"] == "download":
                    self.assertFalse((root / row["destination"]).exists())
                progress(event)
            return original_install(row, root, progress=wrapped, **kwargs)
        with patch.object(download_models, "urlopen", side_effect=self.fetch), patch.object(download_models, "install", side_effect=install):
            result = self.wait(self.prep.start(self.selection))
        self.assertEqual(result["state"], "ready", result)
        self.assertEqual(result["bytes_done"], result["bytes_total"])
        self.assertTrue({"download", "verification"}.issubset(phases))
        fresh = self.make()
        with patch.object(download_models, "digest", side_effect=AssertionError("cached files must not rehash")), patch.object(download_models, "urlopen", side_effect=AssertionError("no network")):
            self.assertTrue(fresh.inspect(self.selection)["ready"])
            self.prep = fresh
            self.assertEqual(self.wait(fresh.start(self.selection))["state"], "ready")

    def test_stat_only_existing_file_is_unverified_then_full_verify(self):
        for destination, payload in self.payloads.items():
            path = self.root / destination
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        with patch.object(download_models, "digest", side_effect=AssertionError("inspect cannot hash")):
            plan = self.prep.inspect(self.selection)
        self.assertTrue(all(f["status"] == "unverified" for f in plan["files"]))
        with patch.object(download_models, "urlopen", side_effect=AssertionError("no network")):
            result = self.wait(self.prep.start(self.selection))
        self.assertEqual(result["state"], "ready")

    def test_changed_file_invalidates_cache_and_hash_mismatch_never_overwrites(self):
        with patch.object(download_models, "urlopen", side_effect=self.fetch):
            self.assertEqual(self.wait(self.prep.start(self.selection))["state"], "ready")
        destination, original = next(iter(self.payloads.items()))
        target = self.root / destination
        bad = b"!" * len(original)
        target.write_bytes(bad)
        self.assertFalse(self.prep.inspect(self.selection)["ready"])
        with patch.object(download_models, "urlopen", side_effect=AssertionError("no network")):
            failed = self.wait(self.prep.start(self.selection))
        self.assertEqual(failed["state"], "failed")
        self.assertIn("SHA256", failed["error"])
        self.assertEqual(target.read_bytes(), bad)
        target.write_bytes(b"wrong size")
        self.assertEqual(self.prep.inspect(self.selection)["files"][0]["status"], "mismatch")
        self.assertEqual(self.wait(self.prep.start(self.selection))["state"], "failed")
        self.assertEqual(target.read_bytes(), b"wrong size")

    def test_registered_extra_roots_are_reused_without_copy_or_private_paths(self):
        extra = Path(self.temp.name) / "private-extra"
        for destination, payload in self.payloads.items():
            path = extra / destination
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        self.prep = self.make(lambda category, name: extra / category / name)
        with patch.object(download_models, "urlopen", side_effect=AssertionError("no network")):
            result = self.wait(self.prep.start(self.selection))
        self.assertEqual(result["state"], "ready")
        self.assertFalse(self.root.exists())
        self.assertNotIn(str(extra), json.dumps(self.prep.inspect(self.selection)))
        self.assertNotIn(str(extra), json.dumps(result))

    def test_job_failure_retry_and_dedup_conflict(self):
        entered, release = Event(), Event()
        def blocked(request, timeout=None):
            entered.set()
            release.wait(3)
            raise OSError("fixture network failure")
        with patch.object(download_models, "urlopen", side_effect=blocked):
            first = self.prep.start(self.selection)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.prep.start(self.selection)["job_id"], first["job_id"])
            with self.assertRaises(PreparationConflict):
                self.prep.start({**self.selection, "dit_checkpoint": "custom.safetensors"})
            release.set()
            failed = self.wait(first)
        self.assertEqual(failed["state"], "failed")
        with patch.object(download_models, "urlopen", side_effect=self.fetch):
            retry = self.prep.start(self.selection)
            self.assertNotEqual(first["job_id"], retry["job_id"])
            self.assertEqual(self.wait(retry)["state"], "ready")

    def test_bad_download_hash_never_ready_or_published(self):
        with patch.object(download_models, "urlopen", return_value=io.BytesIO(b"bad")):
            result = self.wait(self.prep.start(self.selection))
        self.assertEqual(result["state"], "failed")
        self.assertFalse(self.prep.inspect(self.selection)["ready"])
        self.assertFalse(list(self.root.rglob("*.partial")))
        self.assertFalse((self.root / next(iter(self.payloads))).exists())

    def test_service_contract_custom_metadata_and_server_rejects_urls_paths(self):
        service = SetupService(self.prep, lambda: environment())
        report = service.inspect(self.selection)
        self.assertTrue({"environment", "evaluation", "assets", "options", "defaults", "models", "dependency_plan"}.issubset(report))
        self.assertEqual(len(report["models"]), 2)  # DiT + VAE, not shards/tokenizer.
        self.assertEqual(report["models"][0]["name"], catalog.checkpoint_defaults(self.selection)["dit_checkpoint"])
        custom = service.inspect({**self.selection, "dit_checkpoint": "my-own.safetensors"})
        item = custom["assets"]["files"][0]
        self.assertEqual(item["status"], "custom")
        self.assertIsNone(item["url"])
        self.assertIsNone(item["sha256"])
        self.assertEqual(custom["models"][0]["name"], "my-own.safetensors")
        for body in ({"selection": {}, "url": "https://example.org"},
                     {"selection": {"manifest": "evil.json"}},
                     {"selection": {"dit_checkpoint": "../evil"}},
                     {"selection": {"dit_checkpoint": "C:/evil"}},
                     {"selection": {"dit_checkpoint": "https://evil/a"}}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                _request_body(body)

    def test_missing_dependencies_do_not_gray_hardware_compatible_choices(self):
        options = hardware_options({}, environment(missing=True))
        self.assertTrue(next(o for o in options["precision"] if o["value"] == "bf16")["enabled"])
        sm75 = hardware_options({"precision": "bf16"}, environment(75, missing=True))
        self.assertFalse(next(o for o in sm75["precision"] if o["value"] == "bf16")["enabled"])
        self.assertTrue(next(o for o in sm75["precision"] if o["value"] == "fp16_fixed")["enabled"])
        self.assertTrue(next(o for o in sm75["vae_precision"] if o["value"] == "fp32")["enabled"])

    def test_explicit_reviewed_repair_preserves_wrong_file_and_downloads_fixed_bytes(self):
        destination, payload = next(iter(self.payloads.items()))
        path = self.root / destination
        path.parent.mkdir(parents=True)
        wrong = b"!" * len(payload)
        path.write_bytes(wrong)
        # Same-size corruption is discovered only by the explicit hash worker.
        self.assertEqual(self.prep.inspect(self.selection)["files"][0]["status"], "unverified")
        self.assertEqual(self.wait(self.prep.start(self.selection))["state"], "failed")
        item = self.prep.inspect(self.selection)["files"][0]
        self.assertEqual(item["status"], "mismatch")
        self.assertEqual(item["repair_action"], "quarantine_then_download")
        self.assertIn("保留全部字节", item["message"])
        # A retry without the reviewed relative repair list still cannot move it.
        self.assertEqual(self.wait(self.prep.start(self.selection))["state"], "failed")
        self.assertEqual(path.read_bytes(), wrong)
        with self.assertRaises(ValueError):
            self.prep.start(self.selection, repair_files=["../anything"])
        with patch.object(download_models, "urlopen", side_effect=self.fetch):
            result = self.wait(self.prep.start(self.selection, repair_files=[destination]))
        self.assertEqual(result["state"], "ready", result)
        self.assertEqual(path.read_bytes(), payload)
        preserved = result["files"][0]["quarantined_name"]
        self.assertFalse(Path(preserved).is_absolute())
        self.assertEqual((self.root / preserved).read_bytes(), wrong)

    def test_directory_and_symlink_never_quarantined(self):
        destination = next(iter(self.payloads))
        path = self.root / destination
        path.mkdir(parents=True)
        self.assertEqual(self.prep.inspect(self.selection)["files"][0]["status"], "error")
        with self.assertRaises(ValueError):
            self.prep.start(self.selection, repair_files=[destination])
        self.assertTrue(path.is_dir())
        path.rmdir()
        other = Path(self.temp.name) / "other"
        other.write_bytes(b"not a model")
        try:
            path.symlink_to(other)
        except OSError as error:
            if getattr(error, "winerror", None) != 1314:
                raise
            # Windows portable installs need no Developer Mode/admin privilege.
            # Exercise the same refusal branch with a mocked link observation;
            # Linux and privileged Windows still exercise a real filesystem link.
            original_is_symlink = Path.is_symlink
            def observed_link(candidate):
                return candidate == path or original_is_symlink(candidate)
            with patch.object(Path, "is_symlink", observed_link):
                self.assertEqual(self.prep.inspect(self.selection)["files"][0]["status"], "error")
                with self.assertRaises(ValueError):
                    self.prep.start(self.selection, repair_files=[destination])
            self.assertFalse(path.exists())
        else:
            self.assertEqual(self.prep.inspect(self.selection)["files"][0]["status"], "error")
            with self.assertRaises(ValueError):
                self.prep.start(self.selection, repair_files=[destination])
            self.assertTrue(path.is_symlink())
        self.assertEqual(other.read_bytes(), b"not a model")

    def test_repair_rechecks_stat_before_atomic_move(self):
        destination, payload = next(iter(self.payloads.items()))
        path = self.root / destination
        path.parent.mkdir(parents=True)
        path.write_bytes(b"!" * len(payload))
        row = self.manifest["assets"][0]["files"][0]
        self.prep.cache.remember(path, row["sha256"], valid=False)
        original_verify = download_models.verify
        def changed(target, expected, progress=None):
            result = original_verify(target, expected, progress)
            target.write_bytes(b"changed during verification")
            return result
        with patch.object(download_models, "verify", side_effect=changed):
            result = self.wait(self.prep.start(self.selection, repair_files=[destination]))
        self.assertEqual(result["state"], "failed")
        self.assertIn("发生变化", result["error"])
        self.assertEqual(path.read_bytes(), b"changed during verification")
        self.assertFalse((path.parent / ".moss_lowvram_rejected").exists())

    def test_route_registration_dispatch_and_http_errors_without_comfy_imports(self):
        import runtime.setup_service as routes_module
        handlers = {}
        class Routes:
            def post(self, path):
                return lambda handler: handlers.setdefault(("POST", path), handler)
            def get(self, path):
                return lambda handler: handlers.setdefault(("GET", path), handler)
        class Request:
            def __init__(self, body=None, job_id=None):
                self.body = body
                self.match_info = {"job_id": job_id}
            async def json(self):
                return self.body
        def response(data, status=200):
            return {"data": data, "status": status}
        async def inline_thread(function, *args):
            return function(*args)
        service = SetupService(self.prep, lambda: environment())
        with (patch.dict(sys.modules, {
                "server": SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(routes=Routes()))),
                "aiohttp": SimpleNamespace(web=SimpleNamespace(json_response=response))}),
              patch.object(routes_module, "_registered", False),
              patch.object(routes_module.asyncio, "to_thread", side_effect=inline_thread)):
            routes_module.register_setup_routes(service)
            routes_module.register_setup_routes(service)
            self.assertEqual(len(handlers), 3)
            inspect = handlers["POST", "/moss_lowvram/setup/inspect"]
            result = asyncio.run(asyncio.wait_for(inspect(Request({"selection": self.selection, "verify": False})), timeout=2))
            self.assertEqual(result["status"], 200)
            self.assertFalse(result["data"]["assets"]["ready"])
            download = handlers["POST", "/moss_lowvram/setup/download"]
            result = asyncio.run(download(Request({"selection": self.selection, "url": "https://evil"})))
            self.assertEqual(result["status"], 400)
            with patch.object(service, "download", side_effect=PreparationConflict("已有任务")):
                result = asyncio.run(asyncio.wait_for(download(Request({"selection": self.selection})), timeout=2))
                self.assertEqual(result["status"], 409)
            job = handlers["GET", "/moss_lowvram/setup/jobs/{job_id}"]
            self.assertEqual(asyncio.run(job(Request(job_id="unknown")))["status"], 404)

    def test_default_service_resolver_preserves_lexical_symlink_identity(self):
        import runtime.setup_service as service_module
        registered = Path(self.temp.name) / "registered"
        name = catalog.checkpoint_defaults({})["dit_checkpoint"]
        lexical_link = registered / "diffusion_models" / name
        target = registered / "other-root" / "ordinary.safetensors"
        original_is_file, original_is_symlink, original_resolve = Path.is_file, Path.is_symlink, Path.resolve
        def observed_file(path):
            return path in (lexical_link, target) or original_is_file(path)
        def observed_link(path):
            return path == lexical_link or original_is_symlink(path)
        def observed_resolve(path, *args, **kwargs):
            return target if path == lexical_link else original_resolve(path, *args, **kwargs)
        fake_folders = SimpleNamespace(models_dir=str(self.root), get_filename_list=lambda category: [],
                                       get_folder_paths=lambda category: [str(registered / category), str(registered / "other-root")])
        with (patch.dict(sys.modules, {"folder_paths": fake_folders}),
              patch.object(service_module, "_service", None),
              patch.object(Path, "is_file", observed_file),
              patch.object(Path, "is_symlink", observed_link),
              patch.object(Path, "resolve", observed_resolve),
              patch("runtime.assets._registered", side_effect=AssertionError("ordinary resolver must not hide symlink identity"))):
            service = service_module.default_service()
            self.preparations.append(service.preparation)
            found = service.preparation.resolver("diffusion_models", name)
            self.assertEqual(found, lexical_link)
            self.assertEqual(service.preparation._find("diffusion_models/" + name), lexical_link)
            item = service.preparation.plan({})[0]
            self.assertEqual(item["status"], "error")
            self.assertIsNone(item["repair_action"])
            self.assertIn("符号链接", item["message"])

    def test_unpublished_sm75_flash_disabled_but_installable_sage_enabled(self):
        sm75 = hardware_options({}, environment(75))
        flash = next(o for o in sm75["attention"] if o["value"] == "Flash FP16/BF16")
        self.assertFalse(flash["enabled"])
        self.assertIn("turing_flash_missing", flash["error_codes"])
        self.assertTrue(next(o for o in sm75["attention"] if o["value"] == "auto")["enabled"])
        sm89 = hardware_options({}, environment(89))
        sage = next(o for o in sm89["attention"] if o["value"] == "Sage QK8 PV16")
        self.assertTrue(sage["enabled"])

    def test_publish_pending_has_no_invented_download_link(self):
        self.manifest["assets"][0].update(status="local_only/publish_pending", files=[],
                                           local_filename=catalog.checkpoint_defaults(self.selection)["dit_checkpoint"])
        self.prep = self.make()
        item = self.prep.inspect(self.selection)["files"][0]
        self.assertEqual(item["status"], "publish_pending")
        self.assertIsNone(item["url"])
        result = self.wait(self.prep.start(self.selection))
        self.assertEqual(result["state"], "failed")
        self.assertIn("尚未公开发布", result["error"])


if __name__ == "__main__":
    unittest.main()
