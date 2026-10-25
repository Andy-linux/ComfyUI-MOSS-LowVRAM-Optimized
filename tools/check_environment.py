"""Read-only portable dependency inventory. No pip, downloads or test workloads.

Run with the exact Python executable used to start ComfyUI. Optional extensions
are located and inspected through metadata, never imported by this checker.
"""
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "torch": "torch", "triton": "triton", "comfy_kitchen": "comfy-kitchen",
    "xformers": "xformers", "sageattention": "sageattention",
    "flash_attn": "flash-attn", "transformers": "transformers",
    "safetensors": "safetensors", "accelerate": "accelerate",
}


def inventory(name, distribution):
    result = {"import_name": name, "distribution": distribution}
    try:
        spec = importlib.util.find_spec(name)
        result.update(found=spec is not None, origin=None if spec is None else spec.origin)
    except (ImportError, ValueError, AttributeError) as exc:
        result.update(found=False, discovery_error=str(exc))
    try:
        result["version"] = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        result["version"] = None
    candidates = set()
    for entry in sys.path:
        base = Path(entry or os.getcwd())
        for candidate in (base / name, base / (name + ".py")):
            if candidate.exists():
                candidates.add(str(candidate.resolve()))
    result["sys_path_candidates"] = sorted(candidates)
    result["shadow_conflict"] = len(candidates) > 1
    return result


def inspect():
    packages = {name: inventory(name, dist) for name, dist in PACKAGES.items()}
    private = {
        "xformers": ROOT / "runtime/vendor/xformers034cu128/xformers",
        "sageattention": ROOT / "runtime/vendor/sageattn220/sageattention",
    }
    for name, path in private.items():
        packages[name]["private_path"] = str(path)
        packages[name]["private_present"] = path.is_dir()
        packages[name]["private_public_shadow_risk"] = path.is_dir() and packages[name]["found"]
    result = {"read_only": True, "python": sys.executable, "python_version": sys.version,
              "platform": platform.platform(), "packages": packages,
              "optional_extensions_imported": False, "gpu_workload_run": False}
    try:
        import torch
        result["torch_runtime"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                                   "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            result["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                               "capability": list(torch.cuda.get_device_capability(i)),
                               "total_vram_bytes": torch.cuda.get_device_properties(i).total_memory,
                               "attention_qualification_scope": "SM89 hardware scope; installed build still requires validation"
                               if torch.cuda.get_device_capability(i) == (8, 9) else "not qualified for explicit R4 attention modes"}
                              for i in range(torch.cuda.device_count())]
        probe = getattr(torch.backends.cuda, "is_flash_attention_available", None)
        result["torch_flash_compiled"] = bool(probe()) if probe else False
    except Exception as exc:
        result["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"
    result["notes"] = [
        "Metadata presence does not prove an extension imports or supports the actual tensor layout.",
        "Private dependency directories take priority in the runtime; restart if a different package was already imported.",
        "Public Flash uses torch FLASH_ATTENTION exclusively; no math/efficient/cuDNN fallback.",
        "Use auto when optional explicit attention dependencies are absent or unqualified.",
    ]
    return result


if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2, ensure_ascii=False, default=str))
