"""Describe a running ComfyUI using its own Python environment."""
import json
from pathlib import Path
import sys
import psutil

process = psutil.Process(int(sys.argv[1]))
arguments = process.cmdline()
cwd = Path(process.cwd())
main = next(Path(argument) for argument in arguments[1:] if Path(argument).name == "main.py")
main = main if main.is_absolute() else cwd / main
root = Path(arguments[arguments.index("--base-directory") + 1]) if "--base-directory" in arguments else main.parent
root = root if root.is_absolute() else cwd / root
port = int(arguments[arguments.index("--port") + 1]) if "--port" in arguments else 8188
print(json.dumps({"python": sys.executable, "root": str(root.resolve()), "port": port}))
