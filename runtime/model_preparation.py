"""Explicit model preparation from the packaged immutable manifest only.

Inspection is stat/cache-only. A preparation worker verifies full SHA256 before
marking ready; it never replaces an existing mismatching file.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from threading import RLock
import uuid

from . import catalog
try:
    from ..tools import download_models
except ImportError:  # Standalone CPU tests and tools import runtime at top level.
    from tools import download_models


class PreparationConflict(RuntimeError):
    """Another selection currently owns the single preparation worker."""


def relative_name(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or "\x00" in value or value.startswith("/")
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("模型名称必须是已注册模型目录内的相对名称，不能提供路径或 URL。")
    return value


def bundle_relative(row):
    parts = PurePosixPath(row.get("filename", row["destination"])).parts
    for index, part in enumerate(parts):
        if part in ("text_encoder", "tokenizer"):
            return PurePosixPath(*parts[index:])
    raise ValueError("Original TE manifest requires text_encoder/tokenizer bundle paths")


def public_file(item):
    return {key: value for key, value in item.items() if not key.startswith("_")}


class VerificationCache:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.lock = RLock()
        self.records = {}
        if self.path and self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.records = data
            except (OSError, ValueError):
                pass

    @staticmethod
    def key(path, sha):
        stat = path.stat()
        return json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                           stat.st_ctime_ns, sha.lower()], separators=(",", ":"))

    def contains(self, path, sha):
        try:
            key = self.key(path, sha)
        except OSError:
            return False
        with self.lock:
            return self.records.get(key) is True

    def mismatch(self, path, sha):
        try:
            key = self.key(path, sha)
        except OSError:
            return False
        with self.lock:
            return self.records.get(key) is False

    def remember(self, path, sha, expected_key=None, valid=True):
        key = self.key(path, sha)
        if expected_key is not None and key != expected_key:
            raise ValueError("校验时模型文件发生变化，请停止修改该文件后重试。")
        with self.lock:
            self.records[key] = valid
            self.records = dict(list(self.records.items())[-2048:])
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    temporary.write_text(json.dumps(self.records), encoding="utf-8")
                    temporary.replace(self.path)
                finally:
                    temporary.unlink(missing_ok=True)


class ModelPreparation:
    """Inject root/manifest/registered-name resolver for small deterministic tests.

    resolver(category, relative_name) may return an existing registered file or
    directory. It never receives URL/path input from a caller.
    """
    def __init__(self, manifest, root, resolver=None, cache_path=None, max_jobs=32, registered_roots=None, candidate_provider=None):
        self.manifest = deepcopy(manifest)
        self.assets = {a["id"]: a for a in self.manifest["assets"]}
        self.root = Path(root).resolve()
        self.candidate_provider = candidate_provider
        self.resolver = resolver or (lambda category, name: None)
        self.registered_roots = {category: [Path(p).resolve() for p in paths]
                                 for category, paths in (registered_roots or {}).items()}
        self.cache = VerificationCache(cache_path)
        self.lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moss-model-prepare")
        self.jobs = OrderedDict()
        self.active = None
        self.max_jobs = max(2, max_jobs)

    def _find(self, destination):
        category, name = destination.split("/", 1)
        found = self.resolver(category, name)
        if found is not None and Path(found).is_file():
            return Path(found).absolute()
        return self.root.joinpath(*PurePosixPath(destination).parts)

    def _candidates(self, row, asset_id, rejections=None):
        """Stat-only inventory. Full content reads occur exclusively in workers."""
        category, name = row["destination"].split("/", 1)
        aliases = {Path(name).name, Path(row.get("filename", name)).name}
        if asset_id == "te_int8":
            aliases.add("moss-image-qwen35-int8-convrot.safetensors")
        if self.candidate_provider:
            inventory = self.candidate_provider(category)
        else:
            inventory = (path for root in self.registered_roots.get(category, []) if root.is_dir()
                         for path in root.rglob("*") if ".moss_lowvram_rejected" not in path.parts)
        found = []
        for candidate in inventory:
            path = Path(candidate).absolute()
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                if path.stat().st_size != row.get("size_bytes"):
                    if rejections is not None and path.name in aliases:
                        rejections.append({"name": self._display_location(path, category), "reason": "同名或已知别名文件大小与所选固定版本不同，保留原文件，不视作此模型。"})
                    continue
                if asset_id == "te_original":
                    # Original TE must remain a coherent HF bundle; only its
                    # text_encoder/tokenizer relative layout can be considered.
                    suffix = bundle_relative(row)
                    if tuple(path.parts[-len(suffix.parts):]) != suffix.parts:
                        continue
                if not self.cache.mismatch(path, row["sha256"]):
                    found.append(path)
                elif rejections is not None:
                    rejections.append({"name": self._display_location(path, category), "reason": "完整 SHA256 与所选固定版本不符；保留原文件，不复用此候选。"})
            except OSError:
                continue
        return sorted(set(found), key=lambda path: (path.name not in aliases, str(path)))

    def _display_location(self, path, category):
        for root in [self.root / category] + self.registered_roots.get(category, []):
            try:
                return category + "/" + path.relative_to(root.absolute()).as_posix()
            except ValueError:
                continue
        return category + "/" + path.name

    def runtime_checkpoints(self, selection):
        from .local_models import runtime_checkpoints
        return runtime_checkpoints(selection)

    def checkpoint_defaults(self, selection):
        # Production agrees with catalog; injected manifests remain self-contained
        # rather than silently comparing fixture filenames to production files.
        defaults = catalog.checkpoint_defaults(selection)
        for asset_id, field in zip(catalog.asset_ids_for_selection(selection),
                                   ("dit_checkpoint", "te_checkpoint", "vae_checkpoint")):
            asset = self.assets.get(asset_id, {})
            rows = asset.get("files") or []
            if asset_id == "te_original":
                index = next((r for r in rows if r["destination"].endswith("/model.safetensors.index.json")), None)
                if index:
                    defaults[field] = str(PurePosixPath(index["destination"].split("/", 1)[1]).parent)
            elif rows:
                defaults[field] = rows[0]["destination"].split("/", 1)[1]
            elif asset.get("local_filename"):
                defaults[field] = asset["local_filename"]
        return defaults

    def plan(self, selection):
        ids = catalog.asset_ids_for_selection(selection)
        defaults = self.checkpoint_defaults(selection)
        from .local_models import inspect_models
        local = {row["field"]: row for row in inspect_models(selection)}
        files, seen = [], set()
        for asset_id, field in zip(ids, ("dit_checkpoint", "te_checkpoint", "vae_checkpoint")):
            asset = self.assets.get(asset_id)
            if asset is None:
                raise ValueError("所选模型未列入固定发布清单：" + asset_id)
            chosen = selection.get(field, defaults[field])
            chosen = defaults[field] if chosen == "auto" else relative_name(chosen)
            category = {"dit_checkpoint": "diffusion_models", "te_checkpoint": "text_encoders", "vae_checkpoint": "vae"}[field]
            present = local[field]
            if present["ready"]:
                files.append({"asset_id": asset_id, "name": present["filename"],
                              "destination": category + "/" + chosen, "url": None,
                              "size_bytes": present["size_bytes"], "sha256": None,
                              "status": "local", "ready": True, "verified": False,
                              "message": present["message"], "_path": present["_path"], "_row": None})
                continue
            # A different explicitly selected filename has no invented provenance.
            if chosen != defaults[field]:
                path = self._find(category + "/" + chosen)
                exists = path.is_file() or path.is_dir()
                files.append({"asset_id": asset_id, "name": chosen,
                              "destination": category + "/" + chosen, "url": None,
                              "size_bytes": None, "sha256": None, "status": "custom",
                              "ready": exists, "verified": False,
                              "message": "已选择自定义模型；无固定下载/校验记录，请自行确认文件。" if exists else "自定义模型未找到；请在已注册模型目录中放置所选文件。",
                              "_path": path, "_row": None})
                continue
            rows = asset.get("files") or []
            if not rows:
                name = asset.get("local_filename", chosen)
                rows = [{"destination": category + "/" + name,
                         "size_bytes": asset.get("local_artifact_bytes"),
                         "sha256": asset.get("local_artifact_sha256")}]
            bundle, bundles = None, []
            if asset_id == "te_original":
                index = next((r for r in rows if r["destination"].endswith("model.safetensors.index.json")), None)
                if index:
                    indexes = [self._find(index["destination"])] + self._candidates(index, asset_id)
                    for index_path in indexes:
                        root = index_path.parent.parent
                        if all((root / bundle_relative(r)).is_file() and not (root / bundle_relative(r)).is_symlink()
                               and (root / bundle_relative(r)).stat().st_size == r["size_bytes"] for r in rows):
                            if root not in bundles:
                                bundles.append(root)
                    bundles.sort(key=lambda root: not all(self.cache.contains(root / bundle_relative(r), r["sha256"]) for r in rows))
                    bundle = bundles[0] if bundles else None
            for source in rows:
                row = dict(source)
                destination = relative_name(row["destination"])
                if destination in seen:
                    continue
                seen.add(destination)
                downloadable = asset.get("status") == "public" and "revision" in row
                path = self._find(destination)
                if downloadable and not path.is_symlink():
                    _, url = download_models.checked_file(row, self.root)
                else:
                    url = None
                size, sha = row.get("size_bytes"), row.get("sha256")
                status, message = "missing", "尚未下载；点击准备模型后开始下载并校验。"
                if path.is_symlink() or (path.exists() and not path.is_file()):
                    status, message = "error", "同名位置是目录或符号链接，不能隔离或覆盖；请改选普通模型文件。"
                elif path.exists():
                    if not path.is_file() or not size or path.stat().st_size != size:
                        status, message = "mismatch", "同名文件大小不符；不会覆盖，请先移动或更名该文件。"
                    elif sha and self.cache.mismatch(path, sha):
                        status, message = "mismatch", "完整 SHA256 不符；确认准备后保留旧文件并下载固定版本。"
                    elif sha and self.cache.contains(path, sha):
                        status, message = "ready", "完整 SHA256 已校验，可直接使用。"
                    else:
                        status, message = "unverified", "文件已存在，等待后台完整 SHA256 校验。"
                elif not url:
                    status, message = "publish_pending", "该模型尚未公开发布，缺少固定下载地址；请提供对应本地模型。"
                canonical_path = path
                rejections = []
                candidates = self._candidates(row, asset_id, rejections) if sha else []
                if asset_id == "te_original":
                    candidates = [bundle / bundle_relative(row)] if bundle is not None else []
                candidates = [candidate for candidate in candidates if candidate != canonical_path]
                verified = next((candidate for candidate in candidates if self.cache.contains(candidate, sha)), None)
                if verified is not None:
                    path = verified
                    status, message = "ready", "已复用注册模型目录中的同 SHA256 文件，无需重复下载。"
                elif candidates and status not in ("ready", "error", "unverified"):
                    status, message = "candidate", "发现同类别、同大小的已有文件；准备时完整 SHA256 校验。匹配则复用，不匹配则保留原文件并检查其它候选或下载固定版本。"
                repair = status == "mismatch" and bool(url)
                if repair:
                    message = "文件大小或 SHA256 不符。确认准备后，将旧文件原子移入同目录 .moss_lowvram_rejected 保留全部字节，再下载固定清单版本；不覆盖或删除旧文件。"
                files.append({"asset_id": asset_id, "name": destination.split("/", 1)[1],
                              "destination": destination, "url": url, "size_bytes": size,
                              "sha256": sha, "status": status, "ready": status == "ready",
                              "verified": status == "ready", "message": message,
                              "repair_action": "quarantine_then_download" if repair else None,
                              "candidate_count": len(candidates), "candidate_mismatches": rejections,
                              "resolved_name": self._display_location(path, category) if status == "ready" else None,
                              "_canonical_path": canonical_path, "_candidates": candidates, "_bundles": bundles,
                              "_path": path, "_row": row})
        return files

    @staticmethod
    def summary(files):
        return {"files": [public_file(f) for f in files],
                "total_bytes": sum(f["size_bytes"] or 0 for f in files),
                "missing_bytes": sum((f["size_bytes"] or 0) for f in files if f["status"] in ("missing", "publish_pending")),
                "ready": bool(files) and all(f["ready"] for f in files)}

    def inspect(self, selection):
        return self.summary(self.plan(selection))

    def start(self, selection, repair_files=None, selected_files=None):
        files = self.plan(selection)
        available = {f["destination"] for f in files if f.get("url") and not f["ready"]}
        if selected_files is not None and (not isinstance(selected_files, list) or any(not isinstance(name, str) for name in selected_files)):
            raise ValueError("selected_files 必须是准备窗口中勾选的文件名列表。")
        selected = available if selected_files is None else set(selected_files)
        if not selected or selected - available:
            raise ValueError("请勾选要下载的缺失文件；已有本地模型不会被覆盖或强制校验。")
        files = [f for f in files if f["destination"] in selected]
        repair_files = [] if repair_files is None else repair_files
        if not isinstance(repair_files, list) or any(not isinstance(name, str) for name in repair_files):
            raise ValueError("repair_files 必须是已检查计划内的相对文件名列表。")
        approved = set(repair_files)
        repairable = {f["destination"] for f in files if f.get("repair_action") == "quarantine_then_download"}
        if approved - repairable:
            raise ValueError("隔离请求与当前固定模型计划不符，请刷新并重新检查后确认。")
        # Include selected relative filenames and immutable hashes, never just IDs.
        key = hashlib.sha256(json.dumps([(f["asset_id"], f["destination"], f["sha256"]) for f in files], sort_keys=True).encode()).hexdigest()
        with self.lock:
            if self.active:
                active = self.jobs[self.active]
                if active["state"] in ("queued", "running"):
                    if active["_key"] == key:
                        return self.get(self.active)
                    raise PreparationConflict("另一组模型正在准备中，请等待当前任务完成后再试。")
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {"job_id": job_id, "state": "queued",
                                 "files": [public_file(f) for f in files],
                                 "bytes_done": 0, "bytes_total": sum(f["size_bytes"] or 0 for f in files),
                                 "error": None, "_key": key}
            self.active = job_id
            while len(self.jobs) > self.max_jobs:
                self.jobs.popitem(last=False)
            self.executor.submit(self._run, job_id, files, approved)
            return self.get(job_id)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("准备任务不存在或已过期。")
            return deepcopy(public_file(self.jobs[job_id]))

    def _quarantine(self, item, progress=None):
        path, row = item["_path"], item["_row"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("目录或符号链接不能隔离或覆盖。")
        category = item["destination"].split("/", 1)[0]
        roots = [self.root / category] + self.registered_roots.get(category, [])
        if not any(path.resolve().is_relative_to(root.resolve()) for root in roots):
            raise ValueError("错误文件不在已注册模型目录内，不能隔离。")
        before = path.stat()
        fingerprint = lambda stat: (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev)
        # Confirm the mismatch anew; a now-correct replacement is reused instead.
        before_key = self.cache.key(path, row["sha256"])
        correct = before.st_size == row["size_bytes"] and download_models.verify(path, row, progress)
        if path.is_symlink() or fingerprint(path.stat()) != fingerprint(before):
            raise ValueError("隔离前文件发生变化，已停止；请刷新并重新检查。")
        if correct:
            self.cache.remember(path, row["sha256"], before_key)
            return None
        folder = path.parent / ".moss_lowvram_rejected"
        if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
            raise ValueError("保留旧文件的目录不可用，已停止。")
        folder.mkdir(exist_ok=True)
        target = folder / (path.name + "." + uuid.uuid4().hex + ".rejected")
        if target.exists() or path.is_symlink() or fingerprint(path.stat()) != fingerprint(before):
            raise ValueError("隔离前文件或目标发生变化，已停止。")
        os.rename(path, target)
        return (PurePosixPath(item["destination"]).parent / ".moss_lowvram_rejected" / target.name).as_posix()

    def _run(self, job_id, files, approved):
        completed = 0
        try:
            with self.lock:
                self.jobs[job_id]["state"] = "running"
            original = [item for item in files if item["asset_id"] == "te_original"]
            if original and any(item.get("_candidates") for item in original):
                matched_bundle = None
                for bundle in original[0].get("_bundles", []):
                    matched = True
                    for item in original:
                        row = item["_row"]
                        candidate = bundle / bundle_relative(row)
                        if self.cache.mismatch(candidate, row["sha256"]):
                            matched = False
                            break
                        if not self.cache.contains(candidate, row["sha256"]):
                            before = self.cache.key(candidate, row["sha256"])
                            position = files.index(item)
                            def bundle_progress(event):
                                with self.lock:
                                    self.jobs[job_id]["files"][position].update(
                                        phase="verification", status="verifying", bytes_done=event["bytes_done"],
                                        bytes_total=event["bytes_total"], message="正在完整校验同一 Original TE 候选 bundle…")
                            valid = download_models.verify(candidate, row, bundle_progress)
                            self.cache.remember(candidate, row["sha256"], before, valid=valid)
                            if not valid:
                                matched = False
                                break
                    if matched:
                        matched_bundle = bundle
                        break
                for item in original:
                    if matched_bundle is not None:
                        item["_path"] = matched_bundle / bundle_relative(item["_row"])
                        item["status"] = "ready"
                    else:
                        item["_path"] = item["_canonical_path"]
                        item["_candidates"] = []
                        item["status"] = "unverified" if item["_path"].is_file() else "missing"
            for index, item in enumerate(files):
                row, path = item["_row"], item["_path"]
                if row is None:
                    raise ValueError("自定义模型没有固定 SHA256，不能自动下载或声明校验通过；请使用对应公开文件或自行准备。")
                if item["status"] == "error":
                    raise ValueError(item["message"])
                def progress(event):
                    with self.lock:
                        job = self.jobs[job_id]
                        job["files"][index].update(phase=event["phase"], status="verifying" if event["phase"] == "verification" else "downloading",
                                                     bytes_done=event["bytes_done"], bytes_total=event["bytes_total"],
                                                     message="正在完整校验 SHA256…" if event["phase"] == "verification" else "正在下载模型…")
                        job["bytes_done"] = min(job["bytes_total"], completed + event["bytes_done"])
                for candidate in item.get("_candidates", []) if item["status"] == "candidate" else []:
                    if self.cache.contains(candidate, row["sha256"]):
                        path = candidate
                        break
                    before = self.cache.key(candidate, row["sha256"])
                    matched = download_models.verify(candidate, row, progress)
                    self.cache.remember(candidate, row["sha256"], before, valid=matched)
                    if matched:
                        path = candidate
                        break
                item["_path"] = path
                # Candidate files are never quarantined or overwritten. If no
                # candidate matched, preserve canonical repair authorization.
                if item["status"] == "candidate" and path == item.get("_canonical_path") and path.exists():
                    if path.stat().st_size != row["size_bytes"] or self.cache.mismatch(path, row["sha256"]):
                        raise ValueError(item["name"] + "：已有候选均未匹配；请刷新计划后确认推荐目标的修复操作。")
                if item["status"] == "mismatch":
                    if item["destination"] not in approved:
                        raise ValueError(item["name"] + "：文件校验不符；请刷新计划，确认保留旧文件并下载固定版本后重试。")
                    quarantined = self._quarantine(item, progress)
                    if quarantined:
                        with self.lock:
                            self.jobs[job_id]["files"][index]["quarantined_name"] = quarantined
                if path.is_file():
                    if not row.get("sha256") or path.stat().st_size != row.get("size_bytes"):
                        raise ValueError(item["name"] + "：缺少固定校验信息或文件大小不符，不会覆盖。")
                    if not self.cache.contains(path, row["sha256"]):
                        before = self.cache.key(path, row["sha256"])
                        if not download_models.verify(path, row, progress):
                            self.cache.remember(path, row["sha256"], before, valid=False)
                            raise ValueError(item["name"] + "：SHA256 不符；请刷新计划，再确认保留旧文件并下载固定版本。")
                        self.cache.remember(path, row["sha256"], before)
                else:
                    if not item["url"]:
                        raise ValueError(item["message"])
                    installed = download_models.install(row, self.root, progress=progress)
                    path = Path(installed["path"])
                    item["_path"] = path
                    # install hashes the complete stream before atomic publication.
                    self.cache.remember(path, row["sha256"])
                completed += row["size_bytes"]
                with self.lock:
                    self.jobs[job_id]["files"][index].update(status="ready", ready=True, verified=True,
                                                            phase="complete", resolved_name=self._display_location(path, item["destination"].split("/", 1)[0]), message="完整 SHA256 已校验，可直接使用；已有匹配文件直接复用。")
                    self.jobs[job_id]["bytes_done"] = completed
            with self.lock:
                self.jobs[job_id]["state"] = "ready"
        except Exception as exc:
            # Do not return absolute local paths or private configuration contents.
            message = str(exc).replace(str(self.root), "模型目录")
            for item in files:
                message = message.replace(str(item["_path"]), item["name"])
            with self.lock:
                job = self.jobs[job_id]
                job["state"], job["error"] = "failed", "模型准备失败：" + message
                if files:
                    failed = next((f for f in job["files"] if not f.get("ready")), job["files"][-1])
                    failed.update(status="error", message=job["error"])

    def close(self):
        self.executor.shutdown(wait=True)
