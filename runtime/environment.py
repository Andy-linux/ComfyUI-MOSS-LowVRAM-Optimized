"""Read-only environment inventory and pure selection policy.

Importing this module uses only stdlib. Inspection imports torch/optional kitchen
extensions but never executes a GPU kernel, changes allocation policy, installs
packages or downloads files. Discovery is not whole-workflow qualification.
"""
import importlib
import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"torch": ("torch",), "triton": ("triton", "triton-windows"),
            "comfy_kitchen": ("comfy-kitchen",), "transformers": ("transformers",),
            "safetensors": ("safetensors",), "accelerate": ("accelerate",),
            "diffusers": ("diffusers",),
            "einops": ("einops",), "xformers": ("xformers",),
            "sageattention": ("sageattention",), "flash_attn": ("flash-attn",)}
MINIMUMS = {"diffusers": ((0, 41, 0), (0, 42, 0), "diffusers>=0.41.0,<0.42"),
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
        conflict = bool(private and loaded_origin and path.resolve() not in Path(loaded_origin).resolve().parents)
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
            result[name]["distribution_note"] = "User-authorized optional SM75 artifact; preserve recorded source/license status. Public manifest and per-file integrity required; presence is not new GPU qualification."
    return result


def inspect_environment(device_index=0, check_extensions=True):
    """Return JSON-serializable facts; safely reports missing torch as JSON."""
    packages = {name: _package(name, distributions) for name, distributions in PACKAGES.items()}
    report = {"schema_version": 1, "read_only": True, "gpu_workload_run": False,
              "platform": sys.platform, "machine": platform.machine(),
              "python_version": platform.python_version(), "python_executable": sys.executable,
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
    return report


def evaluate_selection(report, selection):
    """Pure policy: never imports torch, changes settings or launches work.

    Selection uses internal RuntimeConfig strings, not UI labels. Warnings denote
    unvalidated scope; errors mean a known unavailable/incompatible selection.
    """
    errors, warnings, actions = [], [], []
    def error(code, message, action):
        errors.append({"code": code, "message": message})
        if action not in actions:
            actions.append(action)
    def warn(code, message):
        warnings.append({"code": code, "message": message})
    if _version_tuple(report.get("python_version", "")) < (3, 12):
        error("python_version", "This package requires Python >=3.12", "Use a compatible Python >=3.12 ComfyUI environment; no packages were automatically changed.")
    packages = report.get("packages", {})
    for name in ("torch", "triton", "comfy_kitchen", "transformers", "safetensors", "accelerate", "einops", "diffusers"):
        state = packages.get(name, {}).get("status", "missing")
        if state in ("missing", "incompatible"):
            error("dependency_" + name, f"{name}: {state}", f"Install/repair a compatible {name} in ComfyUI's Python; do not automatically replace torch.")
        elif state == "unvalidated":
            warn("dependency_unvalidated_" + name, f"{name} was located but its version or extension was not validated")
        if state not in ("missing", "incompatible") and name in MINIMUMS:
            lower, upper, requirement = MINIMUMS[name]
            version = _release_version(packages.get(name, {}).get("version"))
            if version is None:
                warn("dependency_version_unknown_" + name, f"{name} version cannot be verified against {requirement}")
            elif version < lower or (upper is not None and version >= upper):
                error("dependency_version_" + name, f"Installed {name} {packages[name].get('version')} does not satisfy {requirement}",
                      f"Use ComfyUI's Python -m pip install '{requirement}' after reviewing its dependencies; do not automatically replace torch/CUDA.")
    if not report.get("cuda_available") or report.get("device") is None:
        error("cuda_required", "This runtime requires an available NVIDIA CUDA device", "Use a supported NVIDIA CUDA environment; CPU/ROCm/MPS are not supported by this runtime.")
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
        actions.append("SM75 Triton source: https://github.com/triton-lang/triton-windows/blob/readme/README.md ; historical Windows build: triton-windows==3.2.0.post21. Use ComfyUI's Python -m pip only in the selected environment; newer torch pairing is not guaranteed, and torch/CUDA must not be automatically replaced.")
    elif sm == 89:
        actions.append("SM89 Triton installation: Windows https://github.com/triton-lang/triton-windows/blob/readme/README.md ; Linux https://github.com/triton-lang/triton . Select the version matching existing torch (the upstream Torch2.10 pairing is Triton3.6), rather than blindly upgrading torch/CUDA or installing the newest Triton.")
    if sm == 75:
        if triton >= (3, 3):
            error("sm75_triton", "Triton >=3.3 dropped SM75 support", "Use the separately validated SM75 Triton 3.2 environment; do not install the SM89 stack unchanged.")
        warn("sm75_scope", "Four historical R3 W5 auto UI receipts passed (4g/6g/unlimited and 4g 2MP) with real TE on Windows 2080 Ti 22 GiB, Python3.12, Torch2.10+cu130, Triton3.2.0.post21; this updated release is not yet retested on that card.")
    elif sm == 120:
        warn("sm120_unvalidated", "RTX50/SM120 is recognized but this plugin's full workflow support is pending")
    elif sm == 86:
        warn("sm86_future", "RTX30/SM86 support is planned for later; no workflow qualification is inferred")
    elif sm and sm != 89:
        warn("architecture_unvalidated", f"SM{sm} is outside current SM75/SM89 qualification scope")
    if sm and sm < 80 and (precision == "bf16" or te_dtype == "bf16" or selection.get("vae_precision", "fp32") == "bf16"):
        error("bf16_architecture", "Requested BF16 requires SM80+", "Choose fp16_fixed, FP16 TE and FP32 VAE on SM75.")
    if sm and sm < 80 and te_mode in ("w4a8", "w8a8"):
        error("te_a8_architecture", "Explicit activation-INT8 TE currently requires SM80+", "Choose legacy/explicit W8A16 or original FP16 TE.")
    if sm and sm < 89 and fp8:
        error("fp8_architecture", "Stored FP8 inference binding requires SM89+", "Choose a validated W5/W8 INT8 model for this architecture.")
    # Mirrors current payload-streaming guard, including automatic FP8 4g prefix.
    stream = slots >= 0 or (profile == "4g" and candidate != "q5_ridge_gptq")
    if stream and sm and sm != 89:
        error("payload_stream_architecture", "Partial original-payload streaming is currently SM89-only", "Use resident_blocks=-1 and a nonstreamed configuration; on SM75 prefer the established W5 path.")
    if profile == "4g" and candidate in ("q6_direct", "w8_int8c", "base_w8_int8c") and sm and sm != 89:
        warn("low_budget_unvalidated", "This W6/W8 4g capacity plan is only qualified on SM89; the same budget is not promised on this architecture")
    if selection.get("te_memory", "auto") == "partial" and sm and sm != 89:
        error("te_partial_architecture", "Partial TE residency is currently SM89-only", "Choose te_memory=auto or ring.")
    deps = report.get("optional_dependencies", {})
    if attention == "flash":
        if sm == 75:
            d = deps.get("turing_flash", {})
            if not d.get("private_present"):
                error("turing_flash_missing", "Optional SM75 Turing Flash bundle is missing or its fixed inventory is not ready", "Choose auto, or complete the Manager installation-stage preparation for the published exact SM75 artifact; inspect its recorded source/license status.")
            elif report.get("platform") != "win32" or not str(report.get("python_version", "")).startswith("3.12.") or packages.get("torch", {}).get("version") != "2.10.0+cu130":
                error("turing_flash_abi", "Local Turing adapter requires Windows CPython3.12 / Torch2.10.0+cu130", "Choose auto or use the exact locally qualified optional build.")
            else:
                warn("turing_optional", "Local Turing binary presence is not license clearance or new-release qualification")
        elif sm != 89:
            error("flash_architecture", "Current public Flash adapter is restricted to SM89", "Choose auto; additional architectures need explicit adapter qualification.")
        else:
            d = deps.get("xformers", {})
            if d.get("loaded_private_conflict"):
                error("xformers_shadow", "An incompatible xformers was already loaded before the selected private package", "Restart with one selected installation; do not swap imported binaries.")
            elif not d.get("private_present") and not report.get("torch_flash_compiled"):
                error("flash_missing", "No private Flash binary or compiled torch Flash backend is available", "Install a compatible Flash-enabled CUDA torch build or choose auto; no ordinary SDPA substitution is allowed.")
            else:
                warn("flash_build_scope", "Flash build presence does not prove support for the actual layout or image quality")
    elif attention.startswith("sage_") or attention in ("cudnn", "ck_int8"):
        if sm != 89:
            error("attention_architecture", "This explicit attention adapter is SM89-only", "Choose auto.")
        if attention.startswith("sage_"):
            d = deps.get("sageattention", {})
            if d.get("status", "missing") in ("missing", "incompatible"):
                error("sage_dependency", "Requested SageAttention is missing or conflicts with a loaded package", "Install a compatible SageAttention build in this Python environment or choose auto.")
            else:
                warn("sage_build_scope", "Sage metadata presence is not proof that the requested compiled function/layout works")
    for name, pkg in packages.items():
        if pkg.get("shadow_conflict"):
            warn("package_shadow_" + name, f"Multiple import candidates for {name}; check origin before installation")
    exact = (report.get("platform") == "win32" and sm == 89 and
             str(report.get("python_version", "")).startswith("3.12.") and
             packages.get("torch", {}).get("version") == "2.10.0+cu130" and
             packages.get("comfy_kitchen", {}).get("version") == "0.2.37")
    qualification = "SM89 Windows reference runtime matches; selected features retain their individual scope" if exact else "unvalidated platform/build"
    if not exact and sm == 89:
        warn("platform_build_unvalidated", "SM89 hardware alone does not qualify this OS/Python/torch/kitchen combination; Linux remains unvalidated")
    return {"status": "error" if errors else "warning" if warnings else "available",
            "errors": errors, "warnings": warnings, "actions": actions,
            "qualification": qualification, "gpu_workload_run": False}
