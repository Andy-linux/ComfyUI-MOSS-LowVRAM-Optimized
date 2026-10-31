"""Describe a running ComfyUI using its own Python environment."""
import argparse
import json
from pathlib import Path
import sys
import psutil

process = psutil.Process(int(sys.argv[1]))
arguments = process.cmdline()
cwd = Path(process.cwd())
main = next(Path(argument) for argument in arguments[1:] if Path(argument).name == "main.py")
main = main if main.is_absolute() else cwd / main
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--base-directory", type=Path, default=main.parent)
parser.add_argument("--port", type=int, default=8188)
options, _ = parser.parse_known_args(arguments[1:])
root = options.base_directory if options.base_directory.is_absolute() else cwd / options.base_directory
port = options.port
print(json.dumps({"python": sys.executable, "root": str(root.resolve()), "port": port}))
