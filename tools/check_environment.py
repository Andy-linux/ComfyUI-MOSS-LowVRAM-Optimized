"""Read-only environment/selection JSON. No installation or GPU computation.
Uses the shared API without importing plugin registration. Use ComfyUI Python.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--no-check-extensions", action="store_true")
    ap.add_argument("--selection", help="JSON object using internal RuntimeConfig keys")
    args = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("moss_environment", Path(__file__).resolve().parents[1] / "runtime/environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.inspect_environment(args.device_index, not args.no_check_extensions)
    result = {"environment": report}
    if args.selection:
        selection = json.loads(args.selection)
        if not isinstance(selection, dict):
            ap.error("--selection must be a JSON object")
        result["selection"] = selection
        result["evaluation"] = module.evaluate_selection(report, selection)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result.get("evaluation", {}).get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
