"""Environment inventory and configuration compatibility checks."""
import importlib
import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"torch": ("torch",), "triton": ("triton", "triton-windows"),
            "comfy_kitchen": ("comfy-kitchen",), "transformers": ("transformers",),
            "safetensors": ("safetensors",), "accelerate": ("accelerate",),
            "diffusers": ("diffusers",),
            "einops": ("einops",), "xformers": ("xformers",),
            "sageattention": ("sageattention",), "flash_attn": ("flash-attn",)}
MINIMUMS = {"diffusers": ((0, 41, 0), (1, 0, 0), "diffusers>=0.41.0,<1.0"),
            "transformers": ((5, 20, 1), (6, 0, 0), "transformers>=5.20.1,<6"),
            "safetensors": ((0, 8, 0), None, "safetensors>=0.8.0"),
            "einops": ((0, 8, 2), None, "einops>=0.8.2"),
            "comfy_kitchen": ((0, 2, 37), None, "comfy-kitchen>=0.2.37")}


def _version_tuple(value):
    return tuple(int(n) for n in re.findall(r"\d+", str(value).split("+")[0])[:3])


def _release_version(value):
    # Conservative stdlib parsing: unknown/dev/rc formats remain unvalidated,
    # rather than silently satisfying a stable release lower bound.
    text = str(value or "").split("+")[0]
    if not re.fullmatch(r"\d+(?:\.\d+){1,2}(?:\.post\d+)?", text):
        return None
    parts = tuple(int(n) for n in text.split(".post")[0].split("."))
    return parts + (0,) * (3 - len(parts))


def _package(name, distributions):
    result = {"status": "missing", "version": None, "distribution": None, "origin": None}
    try:
        spec = importlib.util.find_spec(name)
        if spec is not None:
            result.update(status="unvalidated", origin=spec.origin)
    except (ImportError, ValueError, AttributeError) as exc:
        result["discovery_error"] = str(exc)
    for dist in distributions:
        try:
            result.update(version=metadata.version(dist), distribution=dist)
            if result["status"] != "missing":
                result["status"] = "available"
            break
        except metadata.PackageNotFoundError:
            pass
    candidates = set()
    for entry in sys.path:
        base = Path(entry or os.getcwd())
        for candidate in (base / name, base / (name + ".py")):
            if candidate.exists():
                candidates.add(str(candidate.resolve()))
    result["import_candidates"] = sorted(candidates)
    result["shadow_conflict"] = len(candidates) > 1
    return result


def _turing_inventory():
    spec = importlib.util.spec_from_file_location("moss_turing_inventory", Path(__file__).with_name("optional_turing.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inventory()


def _optional_dependencies(packages):
    result = {}
    for name, relative in (("xformers", "xformers034cu128/xformers"),
                           ("sageattention", "sageattn220/sageattention"),
                           ("turing_flash", "turing_bundle")):
        path = ROOT / "runtime/vendor" / relative
        private = (path / "flash_attn_3/_C.pyd").is_file() if name == "xformers" else path.is_dir()
        public = packages.get(name, {})
        loaded = sys.modules.get(name)
        loaded_origin = getattr(loaded, "__file__", None)
        conflict = False
        turing = None
        if name == "turing_flash":
            conflict = False
            turing = _turing_inventory()
            private = turing["status"] == "installed"
        result[name] = {"status": "unvalidated" if private else public.get("status", "missing"),
                        "selected_source": "private" if private else "installed_package" if public.get("status") != "missing" and name in packages else None,
                        "private_present": private, "private_path": str(path),
                        "public_version": public.get("version"), "public_origin": public.get("origin"),
                        "loaded_origin": loaded_origin, "loaded_private_conflict": conflict,
                        "private_public_shadow_risk": private and public.get("status", "missing") != "missing"}
        if conflict:
            result[name]["status"] = "incompatible"
        if name == "turing_flash":
            result[name]["installation"] = turing
            result[name]["status"] = "unvalidated" if private else "incompatible" if turing["status"] == "conflict" else "missing"
            result[name]["distribution_note"] = "SM75 可选 Flash 组件，安装时校验固定文件清单。"
    return result


def python_abi_info():
    """Describe the interpreter ABI without importing extension modules."""
    flags = getattr(sys, "abiflags", "")
    soabi = sysconfig.get_config_var("SOABI")
    cache_tag = getattr(sys.implementation, "cache_tag", None)
    build_flag = sysconfig.get_config_var("Py_GIL_DISABLED")
    tagged = bool(re.search(r"(?:cp|cpython-?)3\d{2}t(?:[-_]|$)", str(soabi or "") + " " + str(cache_tag or "")))
    free_threaded = build_flag in (1, "1") or "t" in flags or tagged
    gil_query = getattr(sys, "_is_gil_enabled", None)
    try:
        gil_enabled = bool(gil_query()) if callable(gil_query) else True if sys.implementation.name == "cpython" and not free_threaded else None
    except Exception:
        gil_enabled = None
    free_threaded = free_threaded or gil_enabled is False
    return {"abiflags": flags, "soabi": soabi, "cache_tag": cache_tag,
            "py_gil_disabled": build_flag in (1, "1"), "gil_enabled": gil_enabled,
            "free_threaded": bool(free_threaded)}


def inspect_environment(device_index=0, check_extensions=True):
    """Report packages, CUDA device and optional backend availability."""
    packages = {name: _package(name, distributions) for name, distributions in PACKAGES.items()}
    report = {"schema_version": 1, "read_only": True, "gpu_workload_run": False,
              "platform": sys.platform, "machine": platform.machine(),
              "python_version": platform.python_version(), "python_executable": sys.executable,
              "python_implementation": sys.implementation.name, "python_abi": python_abi_info(),
              "packages": packages, "device_index": device_index, "device": None,
              "cuda_available": False, "cuda_build": None, "driver_version": None,
              "allocator": None, "torch_flash_compiled": False,
              "extensions_checked": bool(check_extensions), "extensions": {},
              "optional_dependencies": _optional_dependencies(packages)}
    try:
        torch = importlib.import_module("torch")
        packages["torch"].update(status="available", version=torch.__version__)
        report["cuda_build"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            if type(device_index) is not int or not 0 <= device_index < torch.cuda.device_count():
                raise ValueError("Requested CUDA device_index is unavailable")
            props = torch.cuda.get_device_properties(device_index)
            major, minor = torch.cuda.get_device_capability(device_index)
            report["device"] = {"index": device_index, "name": props.name, "capability": [major, minor],
                                "sm": major * 10 + minor, "total_vram_bytes": props.total_memory}
            getter = getattr(torch.cuda.memory, "get_allocator_backend", None)
            report["allocator"] = getter() if getter else "unknown"
        probe = getattr(torch.backends.cuda, "is_flash_attention_available", None)
        report["torch_flash_compiled"] = bool(probe()) if probe else False
    except Exception as exc:
        report["torch_inspection_error"] = f"{type(exc).__name__}: {exc}"
        packages["torch"]["status"] = "missing" if isinstance(exc, ModuleNotFoundError) and exc.name == "torch" else "incompatible"
    if report["cuda_available"]:
        try:
            probe = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=5, check=True)
            versions = sorted(set(probe.stdout.strip().splitlines()))
            report["driver_version"] = versions[0] if len(versions) == 1 else versions
        except (OSError, subprocess.SubprocessError) as exc:
            report["driver_probe_note"] = type(exc).__name__
    if check_extensions:
        try:
            ck = importlib.import_module("comfy_kitchen.backends.cuda")
            symbols = {"cublas_gemm_int8": callable(getattr(getattr(ck, "_C", None), "cublas_gemm_int8", None)),
                       "quantize_int8_rowwise_convrot64": callable(getattr(ck, "quantize_int8_rowwise_convrot64", None))}
            status = "available" if all(symbols.values()) else "incompatible"
            report["extensions"]["kitchen_cuda"] = {"status": status, "symbols": symbols}
            packages["comfy_kitchen"]["status"] = status
        except Exception as exc:
            missing = packages["comfy_kitchen"]["status"] == "missing"
            status = "missing" if missing else "incompatible"
            report["extensions"]["kitchen_cuda"] = {"status": status, "error": f"{type(exc).__name__}: {exc}"}
            packages["comfy_kitchen"]["status"] = status
    else:
        report["extensions"]["kitchen_cuda"] = {"status": "unvalidated", "note": "extension imports skipped"}
    report["python_abi"] = python_abi_info()
    return report


def evaluate_selection(report, selection):
    """Evaluate runtime requirements and configuration compatibility."""
    errors, warnings, actions = [], [], []
    def error(code, message, action):
        errors.append({"code": code, "message": message})
        if action not in actions:
            actions.append(action)
    def warn(code, message):
        warnings.append({"code": code, "message": message})
    if _version_tuple(report.get("python_version", "")) < (3, 12):
        error("python_version", "需要 Python >=3.12。", "请通过 ComfyUI 宿主安装与维护流程准备 Python >=3.12 环境。")
    if (report.get("python_abi") or {}).get("free_threaded"):
        error("python_free_threaded", "当前为 free-threaded Python，原生扩展需要普通带 GIL 的 CPython。", "请使用普通 CPython 3.12，或 SM89 的普通 CPython 3.13 环境。")
    packages = report.get("packages", {})
    for name in ("torch", "triton", "comfy_kitchen", "transformers", "safetensors", "accelerate", "einops", "diffusers"):
        state = packages.get(name, {}).get("status", "missing")
        if state in ("missing", "incompatible"):
            error("dependency_" + name, f"{name}: {state}", f"在 Manager 完成 MOSS 节点包的依赖安装，应用更改并重启后重新检查（{name}）。")
        elif state == "unvalidated":
            warn("dependency_unvalidated_" + name, f"已找到 {name}，版本或扩展状态待确认。")
        if state not in ("missing", "incompatible") and name in MINIMUMS:
            lower, upper, requirement = MINIMUMS[name]
            version = _release_version(packages.get(name, {}).get("version"))
            if version is None:
                warn("dependency_version_unknown_" + name, f"无法确认 {name} 是否满足 {requirement}。")
            elif version < lower or (upper is not None and version >= upper):
                error("dependency_version_" + name, f"{name} {packages[name].get('version')} 不满足 {requirement}。",
                      f"在 Manager 完成节点包依赖安装并应用更改；所需版本：{requirement}。")
    if not report.get("cuda_available") or report.get("device") is None:
        error("cuda_required", "需要可用的 NVIDIA CUDA 设备。", "请使用 NVIDIA CUDA 版 ComfyUI 环境。")
    device = report.get("device") or {}
    sm = device.get("sm", 0)
    candidate = selection.get("candidate", "w8_int8c")
    profile = selection.get("profile", "6g")
    precision = selection.get("precision", "fp16_fixed")
    te_mode = selection.get("te_mode", "auto")
    te_dtype = selection.get("te_compute_dtype", "fp16")
    attention = selection.get("attention", "auto")
    fp8 = candidate in ("base_fp8", "sft_fp8")
    slots = selection.get("resident_blocks", -1)
    triton = _version_tuple(packages.get("triton", {}).get("version"))
    if sm == 75:
        actions.append("SM75 的 Triton 3.2 依赖由匹配宿主的 Manager 安装阶段准备，完成后应用更改。")
    elif sm == 89:
        actions.append("SM89 的 Triton 依赖由 Manager 安装阶段准备，完成后应用更改。")
    if sm == 75:
        if triton >= (3, 3):
            error("sm75_triton", "SM75 不支持 Triton >=3.3。", "请在匹配的 SM75 宿主中通过 Manager 准备 Triton 3.2，并应用更改。")
        warn("sm75_scope", "SM75（RTX 20 系）本版整流程待复测；建议 W5、TE W8A16 和 attention=auto。")
    elif sm == 120:
        warn("sm120_unvalidated", "RTX 50 / SM120 整流程尚待验证。")
    elif sm == 86:
        warn("sm86_future", "RTX 30 / SM86 适配尚未完成。")
    elif sm and sm != 89:
        warn("architecture_unvalidated", f"SM{sm} 整流程尚未验证。")
    if sm and sm < 80 and (precision == "bf16" or te_dtype == "bf16" or selection.get("vae_precision", "fp32") == "bf16"):
        error("bf16_architecture", "BF16 需要 SM80+。", "请选择 fp16_fixed、FP16 TE 和 FP32 VAE。")
    if sm and sm < 80 and te_mode in ("w4a8", "w8a8"):
        error("te_a8_architecture", "TE 的 W8A8 / W4A8 计算需要 SM80+。", "请将 TE 计算改为 W8A16，或使用 Original FP16 TE。")
    if sm and sm < 89 and fp8:
        error("fp8_architecture", "FP8 DiT 需要 SM89+。", "请改选 W5 或 W8 整数 DiT。")
    # Mirrors current payload-streaming guard, including automatic FP8 4g prefix.
    stream = slots >= 0 or (profile == "4g" and candidate != "q5_ridge_gptq")
    if stream and sm and sm != 89:
        error("payload_stream_architecture", "部分 DiT 权重流式搬运仅支持 SM89。", "请设置 resident_blocks=-1，并选择非流式配置；SM75 建议使用 W5。")
    if profile == "4g" and candidate in ("q6_direct", "w8_int8c", "base_w8_int8c") and sm and sm != 89:
        warn("low_budget_unvalidated", "W6/W8 的 4 GB 方案仅在 SM89 验证；其他架构请先使用 W5。")
    if selection.get("te_memory", "auto") == "partial" and sm and sm != 89:
        error("te_partial_architecture", "TE partial 驻留仅支持 SM89。", "请将 te_memory 改为 auto 或 ring。")
    deps = report.get("optional_dependencies", {})
    if attention == "flash":
        if sm == 75:
            d = deps.get("turing_flash", {})
            if not d.get("private_present"):
                error("turing_flash_missing", "SM75 Turing Flash 组件缺失或不完整。", "请先使用 attention=auto；可在 Manager 安装阶段准备对应 SM75 组件后应用更改。")
            elif report.get("platform") != "win32" or not str(report.get("python_version", "")).startswith("3.12.") or not Version("2.10") <= Version(packages.get("torch", {}).get("version", "0")) < Version("2.11"):
                error("turing_flash_abi", "Turing Flash 需要 Windows、Python>=3.12,<3.13、Torch>=2.10,<2.11 与 CUDA13。", "请使用 attention=auto，或先准备匹配宿主的 Turing Flash 组件。")
            else:
                warn("turing_optional", "Turing Flash 仅适用于匹配的 Windows、Python 和 Torch 组合。")
        elif sm < 80:
            error("flash_architecture", "公共 Flash 路径需要 SM80+。", "请选择该显卡的注意力后端。")
        else:
            d = deps.get("xformers", {})
            if d.get("loaded_private_conflict"):
                error("xformers_shadow", "当前 xformers 与所选扩展冲突。", "请先使用 attention=auto，并在宿主维护流程中处理重复扩展后重启。")
            elif not d.get("private_present") and not report.get("torch_flash_compiled"):
                error("flash_missing", "当前环境没有可用的 Flash 后端。", "请使用 attention=auto；需要 Flash 时，先通过 ComfyUI 宿主维护流程准备匹配后端。")
            else:
                warn("flash_build_scope", "Flash 路径受显卡与 CUDA 构建限制。")
    elif attention.startswith("sage_") or attention in ("cudnn", "ck_int8"):
        minimum = 75 if attention == "ck_int8" else 89 if attention.startswith("sage_fp8") else 80
        if sm < minimum:
            error("attention_architecture", f"{attention} 需要 SM{minimum}+，当前 SM{sm}。", "请选择该显卡支持的计算方式。")
        if attention.startswith("sage_"):
            d = deps.get("sageattention", {})
            if d.get("status", "missing") in ("missing", "incompatible"):
                error("sage_dependency", "SageAttention 缺失或与当前扩展冲突。", "请先使用 attention=auto；可在 Manager 安装阶段准备 SageAttention 后应用更改。")
            else:
                warn("sage_build_scope", "Sage 选项需要匹配显卡的可用 CUDA 扩展。")
    for name, pkg in packages.items():
        if pkg.get("shadow_conflict"):
            warn("package_shadow_" + name, f"发现 {name} 多个安装位置，请检查重复安装。")
    qualification = f"按当前 CUDA SM{sm}、依赖范围及所选算子检查"
    return {"status": "error" if errors else "warning" if warnings else "available",
            "errors": errors, "warnings": warnings, "actions": actions,
            "qualification": qualification, "gpu_workload_run": False}
