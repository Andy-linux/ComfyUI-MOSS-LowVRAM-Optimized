"""ComfyUI-Manager installation hook: report prerequisites, never install them."""
import importlib.util
from pathlib import Path
import sys


def main():
    path = Path(__file__).resolve().parent / "runtime/environment.py"
    spec = importlib.util.spec_from_file_location("moss_install_environment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.inspect_environment(check_extensions=False)
    result = module.evaluate_selection(report, {"candidate": "q5_ridge_gptq", "profile": "6g"})
    print("MOSS LowVRAM Optimized: source installation finished.")
    print("Environment readiness:", result["status"])
    for item in result["errors"] + result["warnings"]:
        print(" -", item["message"])
    for action in result["actions"]:
        print(" - Suggested action:", action)
    print("No Torch/CUDA/Triton or model files were installed by this hook.")
    print("After installing prerequisites, restart ComfyUI and use the node's environment check.")
    print("Read docs/ENVIRONMENT_SUPPORT.md; CLI check:", sys.executable, "tools/check_environment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
