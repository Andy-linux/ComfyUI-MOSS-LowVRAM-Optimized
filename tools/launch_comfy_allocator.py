"""Choose allocator in disposable probe; launch Comfy once in a new process.

No inference failure triggers an automatic restart. No installation or downloads.
"""

import argparse, ast, json, os, subprocess, sys
from pathlib import Path
from allocator_probe import allocator_env


def run_probe(python, backend):
    command = [
        python,
        str(Path(__file__).with_name("allocator_probe.py")),
        "--backend",
        backend,
    ]
    proc = subprocess.run(
        command, env=allocator_env(os.environ, backend), capture_output=True, text=True
    )
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError(
            "Allocator probe did not return JSON: " + proc.stderr[-1500:]
        )
    if (result.get("status") == "PASS") != (proc.returncode == 0):
        raise RuntimeError("Probe exit/status mismatch")
    return result


def comfy_allocator_flags(main):
    """Inspect declared CLI options without importing ComfyUI or touching CUDA."""
    path = main.parent / "comfy/cli_args.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except (OSError, SyntaxError, UnicodeError) as error:
        return {
            "async": False,
            "disable": False,
            "source": str(path),
            "read_error": str(error),
        }
    flags = {
        arg.value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        for arg in call.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    return {
        "async": "--cuda-malloc" in flags,
        "disable": "--disable-cuda-malloc" in flags,
        "source": str(path),
    }


def select_backend(main, requested="auto"):
    mechanism = comfy_allocator_flags(main)
    fallback = None
    async_probe = None
    if requested == "native":
        chosen = "native"
        reason = "Explicit native recovery requested by user"
    elif not mechanism["async"]:
        chosen = "native"
        reason = (
            "ComfyUI --cuda-malloc mechanism unavailable; native environment fallback"
        )
        fallback = {"reason": reason, "mechanism": mechanism}
    else:
        async_probe = run_probe(sys.executable, "cudaMallocAsync")
        if async_probe["status"] == "PASS":
            chosen = "cudaMallocAsync"
            reason = "ComfyUI mechanism present and async probe passed"
        elif async_probe["status"] == "UNSUPPORTED":
            chosen = "native"
            reason = "Async backend unsupported by isolated probe"
            fallback = {"reason": reason, "async_probe": async_probe}
        else:
            return {
                "status": "FAILED",
                "async_probe": async_probe,
                "comfy_mechanism": mechanism,
                "reason": "Compatibility test failure requires investigation; no silent fallback",
            }
    selected_probe = (
        async_probe
        if chosen == "cudaMallocAsync"
        else run_probe(sys.executable, "native")
    )
    if selected_probe["status"] != "PASS":
        return {
            "status": "FAILED",
            "selected_backend": chosen,
            "probe": selected_probe,
            "async_probe": async_probe,
            "fallback": fallback,
            "comfy_mechanism": mechanism,
        }
    flags = (
        ["--cuda-malloc"]
        if chosen == "cudaMallocAsync"
        else ["--disable-cuda-malloc"]
        if mechanism["disable"]
        else []
    )
    return {
        "status": "READY",
        "requested_backend": requested,
        "selected_backend": chosen,
        "reason": reason,
        "probe": selected_probe,
        "fallback": fallback,
        "comfy_mechanism": mechanism,
        "comfy_allocator_flags": flags,
        "new_process": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-main", required=True, type=Path)
    ap.add_argument("--backend", choices=["auto", "native"], default="auto")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("comfy_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = a.comfy_args[1:] if a.comfy_args[:1] == ["--"] else a.comfy_args
    if any(x in ("--cuda-malloc", "--disable-cuda-malloc") for x in args):
        ap.error(
            "Allocator flags are selected by this launcher; remove them from passthrough args"
        )
    if not a.comfy_main.is_file():
        ap.error("Comfy main.py does not exist")
    receipt = select_backend(a.comfy_main.resolve(), a.backend)
    print(json.dumps(receipt), flush=True)
    if receipt["status"] != "READY":
        return 2
    if a.probe_only:
        return 0
    return subprocess.call(
        [
            sys.executable,
            "-u",
            str(a.comfy_main.resolve()),
            *receipt["comfy_allocator_flags"],
            *args,
        ],
        env=allocator_env(os.environ, receipt["selected_backend"]),
        cwd=str(a.comfy_main.resolve().parent),
    )


if __name__ == "__main__":
    sys.exit(main())
