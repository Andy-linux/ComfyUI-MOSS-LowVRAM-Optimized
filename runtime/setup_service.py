"""ComfyUI setup inspection and explicit background model preparation routes.

Imports have no ComfyUI, network, installation, or download side effects.
"""
import asyncio
import json
from pathlib import Path

from . import catalog
from .local_models import inspect_models
from .environment import evaluate_selection, inspect_environment
from .install_dependencies import dependency_plan
from .dependency_preparation import DependencyPreparation
from .compatibility import architecture_policy
from .model_preparation import ModelPreparation, PreparationConflict, relative_name

_HARD_CODES = {"bf16_architecture", "te_a8_architecture", "fp8_architecture",
               "sm75_triton", "payload_stream_architecture", "te_partial_architecture",
               "flash_architecture", "attention_architecture", "turing_flash_abi", "turing_flash_missing"}
_SELECTION_KEYS = set(catalog.DEFAULTS) | {"dit_checkpoint", "te_checkpoint", "vae_checkpoint"}


def validate_selection(selection):
    if not isinstance(selection, dict):
        raise ValueError("selection 必须是模型配置对象。")
    unknown = set(selection) - _SELECTION_KEYS
    if unknown:
        raise ValueError("不接受 URL、路径或自定义清单等未知配置字段：" + ", ".join(sorted(unknown)))
    for field in ("dit_checkpoint", "te_checkpoint", "vae_checkpoint"):
        if field in selection and selection[field] != "auto":
            relative_name(selection[field])
    catalog.resolve_configuration(selection)
    return dict(selection)


def _probe_selection(selection, field, value):
    probe = {**selection, field: value}
    # Storage/family switches need a valid dependent operation for their probe.
    options = catalog.configuration_options(probe)
    for dependent in ("dit_storage", "dit_compute", "te_compute"):
        current = probe.get(dependent, catalog.DEFAULTS[dependent])
        choices = options[dependent]
        if not any(o["value"] == current and o["enabled"] for o in choices):
            probe[dependent] = next(o["value"] for o in choices if o["enabled"])
            options = catalog.configuration_options(probe)
    # Hardware probes describe a hypothetical model selection, not the current
    # checkpoint filename; keep dependent catalog filenames consistent too.
    probe.update(catalog.checkpoint_defaults(probe))
    return catalog.environment_selection(probe)


def hardware_options(selection, environment):
    options = catalog.configuration_options(selection)
    # Each choice asks canonical policy about its own hardware implications.
    # Neutralize unrelated hardware selections to avoid disabling every choice
    # because another current field is incompatible. Dependency errors never gray
    # these choices: Manager can install dependencies separately.
    related = {
        "model_family": ("candidate",), "dit_storage": ("candidate",),
        "dit_compute": ("candidate", "fp8_activation_mode"),
        "profile": ("profile", "candidate", "resident_blocks"),
        "precision": ("precision",), "te_storage": ("te_mode", "te_compute_dtype"),
        "te_compute": ("te_mode", "te_compute_dtype"), "attention": ("attention",),
        "vae_precision": ("vae_precision",), "vae_asset": (),
    }
    neutral = {"candidate": "q5_ridge_gptq", "profile": "6g", "precision": "fp16_fixed",
               "te_mode": "w8a16", "te_compute_dtype": "fp16", "attention": "auto",
               "vae_precision": "fp32", "te_memory": "auto", "resident_blocks": -1}
    neutral_errors = {e["code"] for e in evaluate_selection(environment, neutral)["errors"]}
    for field, choices in options.items():
        for option in choices:
            if not option["enabled"]:
                option["reason"] = "与所选模型系列、存储格式或计算模式不兼容。"
                continue
            try:
                internal = _probe_selection(selection, field, option["value"])
            except (ValueError, StopIteration):
                continue
            probe = dict(neutral)
            probe.update({k: internal[k] for k in related.get(field, ())})
            hard = [e for e in evaluate_selection(environment, probe)["errors"]
                    if e["code"] in _HARD_CODES and e["code"] not in neutral_errors]
            if hard:
                option.update(enabled=False, reason="当前硬件不兼容：" + "; ".join(e["message"] for e in hard),
                              error_codes=[e["code"] for e in hard])
    return options


class SetupService:
    def __init__(self, preparation, environment_provider=inspect_environment, dependency_directory=None):
        self.preparation = preparation
        self.environment_provider = environment_provider
        self.dependencies = DependencyPreparation(dependency_directory or preparation.root / ".moss_lowvram" / "setup")

    def inspect(self, selection, verify=False):
        selection = validate_selection(selection)
        if type(verify) is not bool:
            raise ValueError("verify 必须为布尔值。")
        environment = self.environment_provider()
        internal = catalog.environment_selection(selection)
        files = self.preparation.plan(selection)
        if verify:
            # Explicit full verification is run by the route's worker thread.
            # Inspection never downloads missing files.
            for item in files:
                row, path = item["_row"], item["_path"]
                if row and path.is_file() and item["status"] == "unverified":
                    before = self.preparation.cache.key(path, row["sha256"])
                    from .model_preparation import download_models
                    if download_models.verify(path, row):
                        self.preparation.cache.remember(path, row["sha256"], before)
                        item.update(status="ready", ready=True, verified=True, message="完整 SHA256 已校验，可直接使用。")
                    else:
                        self.preparation.cache.remember(path, row["sha256"], before, valid=False)
                        item.update(status="mismatch", ready=False,
                                    repair_action="quarantine_then_download" if item["url"] else None,
                                    message="SHA256 不符；确认准备后将旧文件移入同目录 .moss_lowvram_rejected 保留全部字节，再下载固定清单版本。")
        models = []
        # A directory TE is a nine-file asset, never a pretend single model URL.
        for item in files:
            if item["asset_id"] == "te_original":
                continue
            directory, name = item["destination"].split("/", 1)
            models.append({"name": name, "url": item["url"], "directory": directory,
                           "hash": item["sha256"], "hash_type": "sha256" if item["sha256"] else None,
                           "size_bytes": item["size_bytes"], "status": item["status"]})
        plan = dependency_plan(environment)
        evaluation = evaluate_selection(environment, internal)
        dependency_state = self.dependencies.state()
        ready = (not evaluation["errors"] and plan["status"] == "ready"
                 and not dependency_state["restart_required"] and dependency_state["state"] != "running")
        return {"environment": environment, "runtime_ready": ready,
                "architecture_policy": architecture_policy(environment),
                "dependency_preparation": dependency_state,
                "evaluation": evaluation,
                "assets": self.preparation.summary(files),
                "options": hardware_options(selection, environment),
                "defaults": catalog.checkpoint_defaults(selection), "models": models,
                "local_models": [{k: v for k, v in row.items() if k != "_path"} for row in inspect_models(selection)],
                "dependency_plan": plan}

    def download(self, selection, repair_files=None, selected_files=None):
        return self.preparation.start(validate_selection(selection), repair_files=repair_files,
                                      selected_files=selected_files)

    def job(self, job_id):
        return self.preparation.get(job_id)


_service = None
_registered = False


def default_service():
    global _service
    if _service is None:
        import folder_paths

        root = Path(folder_paths.models_dir)
        manifest = json.loads((Path(__file__).resolve().parents[1] / "model_manifest.json").read_text(encoding="utf-8"))
        # Resolve every manifest file, including Original TE tokenizer companions,
        # through registered roots. No private local override file is returned.
        def resolver(category, name):
            for directory in folder_paths.get_folder_paths(category):
                candidate = Path(directory) / name
                if candidate.is_file():
                    # Preserve the lexical file identity: resolving here would
                    # hide a symlink and let repair move its target instead.
                    return candidate
            return None
        def candidates(category):
            # ComfyUI owns the cached registered file inventory, including
            # extra_model_paths.yaml and nested subdirectories. No weights read.
            for name in folder_paths.get_filename_list(category):
                for directory in folder_paths.get_folder_paths(category):
                    yield Path(directory) / name
            if category == "text_encoders":
                for directory in folder_paths.get_folder_paths(category):
                    root = Path(directory)
                    if root.is_dir():
                        yield from root.rglob("model.safetensors.index.json")
        roots = {category: folder_paths.get_folder_paths(category)
                 for category in ("diffusion_models", "text_encoders", "vae")}
        _service = SetupService(ModelPreparation(manifest, root, resolver,
                                                 root / ".moss_lowvram" / "verified_models.json",
                                                 registered_roots=roots, candidate_provider=candidates),
                                dependency_directory=Path(folder_paths.get_user_directory()) / "moss_lowvram" / "setup")
    return _service


def _request_body(body, *, inspect=False):
    allowed = {"selection", "verify"} if inspect else {"selection", "repair_files", "selected_files"}
    if not isinstance(body, dict) or set(body) - allowed or "selection" not in body:
        raise ValueError("仅接受 selection 配置；不接受 URL、自定义清单或下载路径。")
    return validate_selection(body["selection"])


def register_setup_routes(service=None):
    """Called once by plugin registration; no download starts on registration."""
    global _registered
    if _registered:
        return
    from aiohttp import web
    from server import PromptServer

    get_service = (lambda: service) if service is not None else default_service

    @PromptServer.instance.routes.post("/moss_lowvram/setup/inspect")
    async def setup_inspect(request):
        try:
            body = await request.json()
            selection = _request_body(body, inspect=True)
            result = await asyncio.to_thread(get_service().inspect, selection, body.get("verify", False))
            return web.json_response(result)
        except (ValueError, OSError) as exc:
            return web.json_response({"error": "配置检查失败：" + str(exc)}, status=400)

    @PromptServer.instance.routes.post("/moss_lowvram/setup/dependencies")
    async def setup_dependencies(request):
        body = await request.json()
        if body != {}:
            return web.json_response({"error": "运行环境由当前 ComfyUI 自动识别，无需传入命令或路径。"}, status=400)
        running, queued = PromptServer.instance.prompt_queue.get_current_queue()
        if running or queued:
            return web.json_response({"error": "请先等待生成队列完成，再准备运行环境。"}, status=409)
        return web.json_response(await asyncio.to_thread(get_service().dependencies.start))

    @PromptServer.instance.routes.get("/moss_lowvram/setup/dependencies")
    async def setup_dependencies_state(request):
        return web.json_response(get_service().dependencies.state())

    @PromptServer.instance.routes.post("/moss_lowvram/setup/download")
    async def setup_download(request):
        try:
            body = await request.json()
            selection = _request_body(body)
            return web.json_response(await asyncio.to_thread(get_service().download, selection, body.get("repair_files", []), body.get("selected_files")))
        except PreparationConflict as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except (ValueError, OSError) as exc:
            return web.json_response({"error": "无法准备模型：" + str(exc)}, status=400)

    @PromptServer.instance.routes.get("/moss_lowvram/setup/jobs/{job_id}")
    async def setup_job(request):
        try:
            return web.json_response(get_service().job(request.match_info["job_id"]))
        except KeyError:
            return web.json_response({"error": "准备任务不存在或已过期。"}, status=404)

    _registered = True
