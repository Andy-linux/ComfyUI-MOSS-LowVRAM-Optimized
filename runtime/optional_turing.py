"""Fixed-manifest optional SM75 bundle preparation, installation phase only.

Import and planning use stdlib and never import native extensions. Manifest
publication facts must be supplied from real artifacts; pending data never downloads.
Source/license provenance is preserved as recorded, not promoted to license clearance.
"""
import hashlib
import io
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "optional_binary_manifest.json"
TARGET = ROOT / "runtime/vendor/turing_bundle"
SOURCE_REPOSITORY = "https://github.com/ssiu/flash-attention-turing"
ABI = {"platform": "win32", "machine": "x86_64", "python": "3.12", "torch": "2.10.0+cu130", "cuda": "13.0", "sm": 75}
MAX_ARCHIVE = 512 * 1024 * 1024
MAX_UNPACKED = 1024 * 1024 * 1024


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _path(value):
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ValueError("Unsafe Turing member path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(p in (".", "..") or p.endswith((".", " ")) for p in path.parts):
        raise ValueError("Unsafe Turing member path")
    for part in path.parts:
        if re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part, re.I):
            raise ValueError("Reserved Windows Turing member path")
    return path


def _url(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Turing artifact requires a fixed public HTTPS URL")
    github = parsed.hostname == "github.com" and re.fullmatch(r"/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/download/[A-Za-z0-9_.-]+/[^/]+\.zip", parsed.path)
    hf = parsed.hostname == "huggingface.co" and re.fullmatch(r"/Andylinux/[A-Za-z0-9_.-]+/resolve/[a-f0-9]{40}/[^/]+\.zip", parsed.path)
    if not github and not hf:
        raise ValueError("Turing artifact URL is outside the fixed publisher/revision policy")


def publication():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unknown optional binary manifest schema")
    packages = data.get("packages", {})
    if not isinstance(packages, dict):
        raise ValueError("Optional binary packages must be an object")
    record = packages.get("turing_flash", {})
    if not isinstance(record, dict):
        raise ValueError("Turing record must be an object")
    if record.get("status") != "public":
        return None
    return validate_record(record)


def validate_record(record):
    if not isinstance(record, dict) or record.get("status") != "public":
        raise ValueError("Turing record must describe a real public artifact")
    if record.get("abi") != ABI or record.get("distribution_authorization") != "user_authorized_preserve_recorded_license_status":
        raise ValueError("Turing ABI/distribution record differs from fixed policy")
    source = record.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("Turing source record must be an object")
    if source.get("repository") != SOURCE_REPOSITORY or not re.fullmatch(r"[a-f0-9]{40}", source.get("commit", "")):
        raise ValueError("Turing source repository/commit missing")
    if not re.fullmatch(r"[a-f0-9]{40}", source.get("cutlass_commit", "")):
        raise ValueError("Turing CUTLASS commit missing")
    if not isinstance(record.get("license_status"), str) or not record["license_status"].strip():
        raise ValueError("Turing source/license status must be recorded without invented clearance")
    artifact = record.get("artifact", {})
    if not isinstance(artifact, dict):
        raise ValueError("Turing artifact record must be an object")
    _url(artifact.get("url", ""))
    if type(artifact.get("size_bytes")) is not int or not 0 < artifact["size_bytes"] <= MAX_ARCHIVE or not re.fullmatch(r"[a-f0-9]{64}", artifact.get("sha256", "")):
        raise ValueError("Turing artifact exact size/SHA256 missing")
    files = record.get("files", [])
    if not isinstance(files, list) or not files:
        raise ValueError("Turing per-file inventory missing")
    seen, roles, total = set(), set(), 0
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("Turing member must be an object")
        path = str(_path(row["path"]))
        if path.casefold() in seen or type(row.get("size_bytes")) is not int or row["size_bytes"] < 0 or not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", "")):
            raise ValueError("Turing member duplicate/size/SHA256 invalid")
        seen.add(path.casefold())
        roles.add(row.get("role"))
        total += row["size_bytes"]
    if total > MAX_UNPACKED or not {"binary", "interface", "source", "license", "notice", "build_record"} <= roles:
        raise ValueError("Turing source/license/notice/build inventory incomplete")
    if not {"manifest.json", "flash_attention_interface.py"} <= seen:
        raise ValueError("Turing runtime manifest/interface missing")
    binaries = [r for r in files if r["role"] == "binary"]
    if len(binaries) != 1 or not re.fullmatch(r"flash_attn_turing(?:\.[A-Za-z0-9_-]+)*\.pyd", binaries[0]["path"]):
        raise ValueError("Turing binary filename is not the expected import module")
    return record


def _runtime_manifest(raw, record):
    manifest = json.loads(raw)
    binary = next(row for row in record["files"] if row["role"] == "binary")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("binary"), dict):
        raise ValueError("Turing runtime manifest must describe its binary")
    if manifest.get("binary", {}).get("filename") != binary["path"] or manifest["binary"].get("sha256") != binary["sha256"] or manifest.get("source_head") != record["source"]["commit"]:
        raise ValueError("Turing runtime manifest disagrees with pinned source/binary")
    if manifest.get("source_url") != SOURCE_REPOSITORY or manifest.get("cutlass_head") != record["source"]["cutlass_commit"]:
        raise ValueError("Turing runtime source/CUTLASS records differ")
    if "source_license" not in manifest or not manifest["source_license"]:
        raise ValueError("Original Turing source_license record missing")


def _check_tree(target, record):
    expected = {row["path"]: row for row in record["files"]}
    if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
        raise ValueError("Turing target must not be a link")
    actual = set()
    for path in target.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Turing installed member must not be a link")
        if not path.is_file():
            continue
        relative = path.relative_to(target).as_posix()
        # CPython may create bytecode after isolated import; never import unknown
        # source files, but generated bytecode is not an archive payload member.
        if "__pycache__" in path.relative_to(target).parts and path.suffix == ".pyc":
            continue
        actual.add(relative)
    if actual != set(expected):
        raise ValueError("Existing Turing target has missing/unexpected files")
    for name, row in expected.items():
        file = target.joinpath(*_path(name).parts)
        if file.stat().st_size != row["size_bytes"] or _digest(file.read_bytes()) != row["sha256"]:
            raise ValueError("Installed Turing member size/hash mismatch: " + name)
    _runtime_manifest((target / "MANIFEST.json").read_bytes(), record)
    if not (target / "dlls").is_dir():
        raise ValueError("Turing DLL search directory missing")


def inventory():
    try:
        record = publication()
        if record is None:
            return {"status": "publish_pending", "reason": "用户已授权可选Turing分发；真实公开ZIP URL/大小/SHA及完整来源许可清单尚待填入，不下载占位数据。", "required_for_auto": False}
        if not TARGET.exists():
            return {"status": "missing", "required_for_auto": False}
        _check_tree(TARGET, record)
        return {"status": "installed", "verification": "pinned_file_hashes_and_recorded_source_license_only", "required_for_auto": False}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"status": "conflict", "reason": str(exc), "required_for_auto": False}


def plan(report):
    torch = report.get("packages", {}).get("torch", {})
    eligible = (report.get("platform") == "win32" and report.get("machine") in ("AMD64", "x86_64") and
                re.fullmatch(r"3\.12\.\d+", str(report.get("python_version", ""))) and
                torch.get("status") == "available" and torch.get("version") == "2.10.0+cu130" and
                report.get("cuda_build") == "13.0" and report.get("cuda_available") and
                (report.get("device") or {}).get("sm") == 75)
    if not eligible:
        return {"status": "skipped", "reason": "可选Turing仅覆盖Windows x64/Python3.12/Torch2.10+cu130/SM75；当前宿主跳过，auto不依赖此包。", "required_for_auto": False}, []
    state = dict(report["turing_installation"]) if "turing_installation" in report else inventory()
    steps = []
    if state["status"] == "missing":
        record = publication()
        if record is None:
            raise ValueError("Turing publication record disappeared")
        steps = [{"id": "turing_flash", "kind": "optional_turing", **record["artifact"], "message": "仅安装阶段准备固定SM75 Turing ZIP，逐文件验证来源/许可清单；失败保持auto，不替换Torch。"}]
    return state, steps


def validate_host():
    if sys.platform != "win32" or sys.version_info[:2] != (3, 12) or platform.machine() not in ("AMD64", "x86_64") or metadata.version("torch") != "2.10.0+cu130":
        raise RuntimeError("Turing preparation requires exact Windows x64/Python3.12/Torch2.10+cu130")
    torch = importlib.import_module("torch")
    if torch.version.cuda != "13.0" or not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) != (7, 5):
        raise RuntimeError("Turing preparation requires the selected CUDA13.0 SM75 host")


def validate_archive(data, record=None):
    """Pure artifact validation for root's packaging review, no host/GPU needed."""
    record = publication() if record is None else validate_record(record)
    if record is None:
        raise ValueError("Turing public artifact is not recorded yet")
    artifact = record["artifact"]
    if len(data) != artifact["size_bytes"] or _digest(data) != artifact["sha256"]:
        raise RuntimeError("Turing ZIP size/hash verification failed")
    expected = {row["path"]: row for row in record["files"]}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        seen, actual = set(), set()
        for member in archive.infolist():
            name = str(_path(member.filename.rstrip("/")))
            if name.casefold() in seen or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Turing ZIP duplicate/symlink member")
            seen.add(name.casefold())
            if member.is_dir():
                if not any(p.startswith(name + "/") for p in expected) and name != "dlls":
                    raise ValueError("Unexpected Turing ZIP directory")
                continue
            row = expected.get(name)
            if row is None or member.file_size != row["size_bytes"] or _digest(archive.read(member)) != row["sha256"]:
                raise ValueError("Turing ZIP member not in fixed inventory or hash differs")
            actual.add(name)
        if actual != set(expected):
            raise ValueError("Turing ZIP missing inventoried files")
        _runtime_manifest(archive.read("MANIFEST.json"), record)
    return True


def prepare():
    validate_host()
    record = publication()
    if record is None:
        raise RuntimeError("Turing public artifact is not recorded yet")
    if TARGET.exists():
        _check_tree(TARGET, record)
        return False
    artifact = record["artifact"]
    with urllib.request.urlopen(artifact["url"], timeout=60) as response:
        data = response.read(artifact["size_bytes"] + 1)
    validate_archive(data, record)
    expected = {row["path"]: row for row in record["files"]}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="turing-staging-", dir=TARGET.parent))
        try:
            for name in expected:
                path = staging.joinpath(*_path(name).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
            (staging / "dlls").mkdir(exist_ok=True)
            _check_tree(staging, record)
            if TARGET.exists():
                raise FileExistsError("Turing target appeared; refusing replacement")
            staging.rename(TARGET)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return True


def verify_import():
    validate_host()
    # No adapter construction/device workload. Keep DLL handle alive through the
    # import, then let the isolated installation-stage Python process terminate.
    script = "import os,sys; import torch; root=" + repr(str(TARGET)) + "; dll=os.add_dll_directory(os.path.join(root,'dlls')); sys.path.insert(0,root); import flash_attn_turing; import flash_attention_interface as interface; assert callable(interface.flash_attn_func)"
    subprocess.run([sys.executable, "-B", "-c", script], check=True, timeout=60)
