"""Build dependency plans and run them during Manager installation."""
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from packaging.requirements import Requirement
from packaging.markers import default_environment
from .compatibility import windows_triton_requirement, satisfies, architecture_policy

SOURCE = "https://github.com/triton-lang/triton-windows/blob/readme/README.md"
HEADERS_URL = "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip"
HEADERS_SHA256 = "145cde69c753dd5a78f4525ef1f5cbe0f2642678fdefd9e79e2bbea1a91d80ea"
HEADERS_313_URL = "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.13.2_include_libs.zip"
HEADERS_313_SHA256 = "e3e8c3abb17ea8de1cd2cd264ad15aae3de5eb7dc61ab37cb1a3d07f9c86238d"
SAGE_URL = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl"
SAGE_SHA256 = "1635283f5c01ec3cda58a784d0d7eabbcaffaf9511d1b263db4750e1ed7958bb"
SAGE_BYTES = 16656067
SAGE_ROOT = Path(__file__).resolve().parent / "vendor/sageattn220"
SAGE_MEMBERS = ("sageattention/core.py", "sageattention/_fused.pyd", "sageattention/_qattn_sm80.pyd", "sageattention/_qattn_sm89.pyd", "sageattention/_qattn_sm90.pyd")
MANAGER_ACTION = "请在配置节点点击“准备运行环境”；完成后按引导重启并刷新页面。"
HOST_ACTION = "请先通过 ComfyUI 安装与维护流程准备上述 Python、Torch 和 CUDA 环境，再重新检查：https://www.comfy.org/download"


def _environment():
    importlib.invalidate_caches()
    path = Path(__file__).with_name("environment.py")
    spec = importlib.util.spec_from_file_location("moss_install_inventory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inspect_environment(check_extensions=False)


def _free_threaded(report):
    abi = report.get("python_abi") or {}
    markers = " ".join(str(abi.get(key) or "") for key in ("abiflags", "soabi", "cache_tag"))
    version = str(report.get("python_version", ""))
    return (abi.get("free_threaded") is True or abi.get("py_gil_disabled") in (1, "1")
            or abi.get("gil_enabled") is False or "t" in str(abi.get("abiflags", ""))
            or bool(re.search(r"(?:cp|cpython-?)3\d{2}t(?:[-_]|$)", markers))
            or bool(re.search(r"3\.\d+t|3\.\d+\.\d+t", version)))


def _prerequisites(report):
    """Read the current interpreter's development files and compiler inventory."""
    if "install_prerequisites" in report:
        return report["install_prerequisites"]
    base = Path(sys.base_prefix)
    include = Path(sysconfig.get_path("include") or base / "include")
    version = re.match(r"(\d+)\.(\d+)", str(report.get("python_version", "")))
    suffix = "".join(version.groups()) if version else str(sys.version_info.major) + str(sys.version_info.minor)
    suffix += "t" if _free_threaded(report) else ""
    library_name, pth_name = "python" + suffix + ".lib", "python" + suffix + "._pth"
    triton = report.get("packages", {}).get("triton", {})
    origin = triton.get("origin")
    root = Path(origin).parent if origin else None
    windows = report.get("platform") == "win32"
    return {
        "python_headers": (include / "Python.h").is_file(),
        "python_import_library": (base / "libs" / library_name).is_file() if windows else True,
        "python_library_name": library_name, "python_pth_name": pth_name,
        "header_root": str(base),
        "embedded_python": (base / pth_name).is_file(),
        "c_compiler": bool(shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")) if not windows else bool(root and (root / "runtime/tcc/tcc.exe").is_file()),
        "cuda_compiler": bool(root and (root / "backends/nvidia/bin" / ("ptxas.exe" if windows else "ptxas")).is_file()),
        "compiler_override": os.environ.get("CC"),
    }


def _linux_requirement(report):
    requirements = report.get("torch_requires")
    if requirements is None:
        requirements = metadata.requires("torch") or []
    environment = {**default_environment(), "platform_system": "Linux", "sys_platform": "linux",
                   "platform_machine": report.get("machine", "x86_64"), "extra": ""}
    for text in requirements:
        item = Requirement(text)
        if item.name == "triton" and (item.marker is None or item.marker.evaluate(environment)):
            return "triton" + str(item.specifier)
    return None


def _sage_state():
    """Check optional Sage files and the recorded wheel identity."""
    if not SAGE_ROOT.exists():
        return {"status": "missing"}
    expected = {"url": SAGE_URL, "sha256": SAGE_SHA256,
                "wheel_bytes": SAGE_BYTES, "torch": "2.10.0+cu130",
                "scope": "Optional SM89 attention; no global dependency changes"}
    try:
        marker = json.loads((SAGE_ROOT / "SAGE_WHEEL_PROVENANCE.json").read_text(encoding="utf-8"))
        if marker == expected and all((SAGE_ROOT / name).is_file() for name in SAGE_MEMBERS):
            return {"status": "installed", "verification": "presence_and_provenance_only"}
    except (OSError, ValueError):
        pass
    return {"status": "conflict", "reason": "Sage 组件目录不完整，请通过宿主维护流程修复；期间使用 auto。"}


def _sage_plan(report):
    torch = report.get("packages", {}).get("torch", {})
    eligible = (report.get("platform") == "win32" and
                report.get("machine") in ("AMD64", "x86_64") and
                re.fullmatch(r"3\.12\.\d+", str(report.get("python_version", ""))) and
                torch.get("status") == "available" and torch.get("version") == "2.10.0+cu130" and
                report.get("cuda_build") == "13.0" and report.get("cuda_available") and
                (report.get("device") or {}).get("sm") == 89)
    if not eligible:
        return {"status": "skipped", "reason": "可选 Sage 准备范围：Windows x64 / Python 3.12 / Torch 2.10+cu130 / SM89；当前请使用 auto。"}, []
    state = dict(report["sage_installation"]) if "sage_installation" in report else _sage_state()
    state["required_for_auto"] = False
    steps = []
    if state["status"] == "missing":
        steps.append({"id": "sageattention", "kind": "optional_sage", "url": SAGE_URL,
                      "sha256": SAGE_SHA256, "bytes": SAGE_BYTES,
                      "message": "安装阶段将固定 Sage wheel 准备到插件隔离目录；失败时使用 auto。"})
    return state, steps


def _turing_helper():
    path = Path(__file__).with_name("optional_turing.py")
    spec = importlib.util.spec_from_file_location("moss_optional_turing_install", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependency_plan(report=None):
    """Build an installation plan for the detected Python, Torch and CUDA host."""
    report = _environment() if report is None else report
    prerequisites = _prerequisites(report)
    plan = {"schema_version": 1, "status": "ready", "read_only": True,
            "execution_phase": "explicit_dependency_preparation",
            "scope": "automatic_dependency_preparation", "runtime_readiness": "see_environment_evaluation",
            "steps": [], "optional_steps": [], "optional_dependencies": {}, "errors": [],
            "warnings": [], "actions": [MANAGER_ACTION], "source": SOURCE,
            "qualification": "安装阶段依赖准备",
            "prerequisites": prerequisites, "architecture_policy": architecture_policy(report)}
    def error(code, message):
        plan["errors"].append({"code": code, "message": message})
    packages = report.get("packages", {})
    torch = packages.get("torch", {})
    triton = packages.get("triton", {})
    sm = (report.get("device") or {}).get("sm")
    platform = report.get("platform")
    if not report.get("cuda_available") or not sm:
        error("cuda_required", "未检测到可用 NVIDIA CUDA GPU；请先通过宿主图形维护流程修复驱动/环境。")
    if torch.get("status") != "available":
        error("torch_unavailable", "当前 Torch 不可用，请先修复 ComfyUI 运行环境。")
    requirement = None
    if platform == "win32":
        version = str(report.get("python_version", ""))
        py312 = bool(re.fullmatch(r"3\.12\.\d+", version))
        py313 = bool(re.fullmatch(r"3\.13\.\d+", version))
        implementation = str(report.get("python_implementation") or ("cpython" if py312 else "")).lower()
        if _free_threaded(report):
            error("python_free_threaded", "free-threaded Python 不支持此原生扩展，请使用普通带 GIL 的 CPython。")
        if report.get("machine") not in ("AMD64", "x86_64") or implementation != "cpython" or not (py312 or py313):
            error("python_unsupported", "需要 Windows x64 的普通 CPython 3.12 或 3.13；请使用官方 ComfyUI Portable 或 Desktop。")
        else:
            try:
                requirement = windows_triton_requirement(report)
            except ValueError as exc:
                error("host_unqualified", str(exc))
        if sm in (75, 120):
            plan["warnings"].append({"code": "gpu_unvalidated", "message": f"SM{sm} 依赖可自动准备；完整推理验收范围见环境支持说明。"})
        if prerequisites.get("compiler_override"):
            error("compiler_override", "检测到 CC 编译器覆盖，请先在宿主环境中确认该设置。")
        if not prerequisites.get("python_headers") or not prerequisites.get("python_import_library"):
            if prerequisites.get("embedded_python") and (py312 or py313):
                headers_url, headers_sha = (HEADERS_313_URL, HEADERS_313_SHA256) if py313 else (HEADERS_URL, HEADERS_SHA256)
                plan["steps"].append({"id": "python_headers", "kind": "python_headers",
                                      "message": f"为便携 Python {'3.13' if py313 else '3.12'} 准备 include/libs 开发文件。",
                                      "url": headers_url, "sha256": headers_sha})
            else:
                library = prerequisites.get("python_library_name", "python313.lib" if py313 else "python312.lib")
                missing = []
                if not prerequisites.get("python_headers"):
                    missing.append("include/Python.h")
                if not prerequisites.get("python_import_library"):
                    missing.append("libs/" + library)
                detail = "；当前解释器不是受支持的便携 Python，请通过该 Python 的安装器补齐开发文件"
                error("python_development_files", "缺少 Python " + version + " 开发文件：" + "、".join(missing) + detail + "。")
    elif platform == "linux":
        requirement = _linux_requirement(report)
        if not requirement:
            error("torch_triton_pin_unknown", "请先准备具有明确 Triton 版本要求的 Linux Torch 环境。")
        if not re.fullmatch(r"3\.(?:1[2-9]|[2-9]\d)\.\d+", str(report.get("python_version", ""))):
            error("python_unqualified", "需要 Python >=3.12 的稳定版本。")
        if sm == 75 and requirement and not satisfies("3.2.0", requirement):
            error("sm75_triton", "当前 Torch 要求的 Triton 已移除 SM75 支持，请准备兼容宿主。")
        if not prerequisites.get("python_headers") or not prerequisites.get("c_compiler"):
            error("linux_build_prerequisites", "Linux 缺少 Python 开发头文件或 C 编译器；请由宿主图形软件管理器/管理员准备。")
        plan["warnings"].append({"code": "linux_unqualified", "message": "Linux 为实验性准备路径，依赖版本随当前 Torch 元数据确定。"})
    else:
        error("platform_unsupported", "自动准备仅支持 NVIDIA Windows x64 / Linux 环境。")
    if triton.get("shadow_conflict"):
        error("triton_shadow", "当前 Triton 存在重复安装，请通过 ComfyUI 环境维护流程处理后重启。")
    if requirement:
        distribution = Requirement(requirement).name
        installed = triton.get("status") == "available" and triton.get("distribution") == distribution
        compatible = installed and satisfies(triton.get("version"), requirement)
        if not compatible:
            previous = triton.get("version")
            plan["steps"].append({"id": "triton", "kind": "pip", "requirement": requirement,
                                  "message": f"为当前显卡和 Torch 准备 {requirement}" + (f"；当前为 {previous}，完成后需重启。" if previous else "。")})
        elif not prerequisites.get("cuda_compiler") or (platform == "win32" and not prerequisites.get("c_compiler")):
            plan["steps"].append({"id": "triton", "kind": "pip", "requirement": requirement,
                                  "reinstall": True, "message": "补齐当前 Triton wheel 的编译器文件，完成后重启。"})
    required = Path(__file__).resolve().parents[1] / "requirements.txt"
    for text in required.read_text(encoding="utf-8").splitlines():
        text = text.strip()
        if not text or text.startswith("#"):
            continue
        item = Requirement(text)
        try:
            current = metadata.version(item.name)
        except metadata.PackageNotFoundError:
            current = None
        if current is None or not satisfies(current, text):
            plan["steps"].append({"id": "package_" + item.name, "kind": "package", "requirement": text,
                                  "message": "补齐节点依赖 " + text})
    sage, optional_steps = _sage_plan(report)
    plan["optional_dependencies"]["sageattention"] = sage
    if sage["status"] == "conflict":
        plan["warnings"].append({"code": "optional_sage_conflict", "message": sage["reason"]})
    turing, turing_steps = _turing_helper().plan(report)
    plan["optional_dependencies"]["turing_flash"] = turing
    optional_steps += turing_steps
    if turing["status"] in ("conflict", "publish_pending"):
        plan["warnings"].append({"code": "optional_turing_" + turing["status"], "message": turing["reason"]})
    if plan["errors"]:
        plan["status"] = "unsupported"
        plan["steps"] = []
        plan["actions"].append(HOST_ACTION)
    else:
        plan["optional_steps"] = optional_steps
        if plan["steps"]:
            plan["status"] = "action_required"
    return plan


def _install_headers(step):
    base = Path(sys.base_prefix).resolve()
    version = sys.version_info[:2]
    archives = {
        (3, 12): (HEADERS_URL, HEADERS_SHA256, "python312._pth", "libs/python312.lib"),
        (3, 13): (HEADERS_313_URL, HEADERS_313_SHA256, "python313._pth", "libs/python313.lib"),
    }
    if sys.platform != "win32" or version not in archives:
        raise RuntimeError("Python header preparation requires portable CPython 3.12/3.13")
    url, checksum, marker, import_library = archives[version]
    if not (base / marker).is_file() or (step["url"], step["sha256"]) != (url, checksum):
        raise RuntimeError("Python header preparation does not match this interpreter")
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000 or hashlib.sha256(data).hexdigest() != checksum:
        raise RuntimeError("Python header archive size/hash verification failed")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if sum(m.file_size for m in members) > 10_000_000:
            raise RuntimeError("Python header archive exceeds extraction limit")
        for member in members:
            path = PurePosixPath(member.filename)
            if (not path.parts or path.parts[0] not in ("include", "libs") or
                    path.is_absolute() or ".." in path.parts or "\\" in member.filename or
                    ":" in member.filename or ((member.external_attr >> 16) & 0o170000) == 0o120000):
                raise RuntimeError("Unsafe Python header archive member")
            target = base.joinpath(*path.parts)
            if base not in target.resolve().parents:
                raise RuntimeError("Python header target escapes interpreter")
        if not {"include/Python.h", import_library}.issubset(archive.namelist()):
            raise RuntimeError("Python header archive missing required files")
        for member in members:
            target = base.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(archive.read(member))


def _prepare_sage():
    """Prepare the fixed Sage wheel through its isolated installer helper."""
    with urllib.request.urlopen(SAGE_URL, timeout=60) as response:
        data = response.read(SAGE_BYTES + 1)
    if len(data) != SAGE_BYTES or hashlib.sha256(data).hexdigest() != SAGE_SHA256:
        raise RuntimeError("Optional Sage wheel size/hash verification failed")
    helper = Path(__file__).resolve().parents[1] / "tools/install_optional_sage.py"
    # Reuse the existing helper's safety/identity validation before execution.
    spec = importlib.util.spec_from_file_location("moss_optional_sage_install", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (module.URL, module.SHA256, module.BYTES) != (SAGE_URL, SAGE_SHA256, SAGE_BYTES):
        raise RuntimeError("Optional Sage helper identity differs from fixed policy")
    with tempfile.TemporaryDirectory(prefix="moss-sage-install-") as directory:
        wheel = Path(directory) / "sageattention-qualified.whl"
        wheel.write_bytes(data)
        module.prepare(wheel, helper.parents[1])
        subprocess.run([sys.executable, str(helper), "--wheel", str(wheel)], check=True, timeout=180)
    if _sage_state()["status"] != "installed":
        raise RuntimeError("Optional Sage installation did not produce the verified target")


def _verify_sage():
    script = "import sys; sys.path.insert(0, " + repr(str(SAGE_ROOT)) + "); import torch; import sageattention; import sageattention._fused; import sageattention._qattn_sm89"
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


def _install_requirement(step):
    requirement = step["requirement"]
    item = Requirement(requirement)
    command = [sys.executable, "-s", "-m", "pip", "install", "--disable-pip-version-check"]
    if step["kind"] == "pip":
        if item.name not in ("triton", "triton-windows") or item.url:
            raise RuntimeError("Invalid Triton requirement")
        command += ["--no-deps", "--only-binary=:all:"]
    if step.get("reinstall"):
        command.append("--force-reinstall")
    # These are the user's existing host versions, not compatibility requirements.
    protected = ("torch", "torchvision", "torchaudio", "comfy-kitchen", "comfy-aimdo",
                 "comfyui-frontend-package", "comfyui-manager")
    versions = []
    for name in protected:
        try:
            versions.append(name + "==" + metadata.version(name))
        except metadata.PackageNotFoundError:
            pass
    with tempfile.TemporaryDirectory(prefix="moss-host-") as directory:
        constraints = Path(directory) / "preserve-host.txt"
        constraints.write_text("\n".join(versions), encoding="utf-8")
        command += ["--constraint", str(constraints), "--index-url", "https://pypi.org/simple", requirement]
        subprocess.run(command, check=True, timeout=600)


def execute_plan(plan):
    """Prepare the detected interpreter without replacing its Torch installation."""
    fresh = dependency_plan()
    if fresh["status"] == "unsupported":
        return dict(fresh, read_only=False, exit_code=1, restart_required=False)
    if plan.get("steps") != fresh["steps"] or plan.get("optional_steps", []) != fresh["optional_steps"]:
        return dict(fresh, read_only=False, status="plan_changed", exit_code=1, restart_required=False)
    changed = False
    try:
        for step in fresh["steps"]:
            if step["kind"] == "python_headers":
                _install_headers(step)
            elif step["kind"] in ("pip", "package"):
                _install_requirement(step)
            else:
                raise RuntimeError("Unknown install step")
            changed = True
        verified = dependency_plan()
        if verified["status"] != "ready":
            return dict(verified, read_only=False, exit_code=1, restart_required=changed)
        # A wheel can have correct metadata but fail to load its native DLL.
        # Check in an isolated install-stage process, without executing GPU work.
        subprocess.run([sys.executable, "-c", "import triton; import triton.language; from triton._C import libtriton"], check=True, timeout=60)
        optional = verified["optional_dependencies"]["sageattention"]
        if optional["status"] in ("missing", "installed"):
            try:
                if optional["status"] == "missing":
                    _prepare_sage()
                    changed = True
                _verify_sage()
                optional.update(status="available", verification="isolated_native_import_only; no GPU workload")
                verified["optional_steps"] = [step for step in verified["optional_steps"] if step["id"] != "sageattention"]
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile, ImportError) as exc:
                optional.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                verified["warnings"].append({"code": "optional_sage_preparation_failed", "message": "Sage 准备失败，请先使用 auto；可在 Manager 维护阶段重试：" + str(exc)})
        turing = verified["optional_dependencies"]["turing_flash"]
        if turing["status"] in ("missing", "installed"):
            try:
                helper = _turing_helper()
                if turing["status"] == "missing":
                    changed = helper.prepare() or changed
                helper.verify_import()
                turing.update(status="available", verification="pinned_inventory_and_isolated_native_import_only; no GPU workload")
                verified["optional_steps"] = [step for step in verified["optional_steps"] if step["id"] != "turing_flash"]
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile, ImportError) as exc:
                turing.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                verified["warnings"].append({"code": "optional_turing_preparation_failed", "message": "Turing 准备失败，请先使用 auto：" + str(exc)})
        return dict(verified, read_only=False, status="prepared" if changed else "ready", exit_code=0, restart_required=changed)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        fresh["errors"].append({"code": "preparation_failed", "message": f"依赖准备失败：{type(exc).__name__}: {exc}"})
        return dict(fresh, read_only=False, status="failed", exit_code=1, restart_required=changed)
