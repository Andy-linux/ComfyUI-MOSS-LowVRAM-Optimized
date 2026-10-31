"""Prepare the current ComfyUI interpreter from Manager or the setup dialog."""
import argparse
import importlib.util
import subprocess
import json
from pathlib import Path
import sys


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime.network import configure_proxy
    configure_proxy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    if args.log_file:
        class Tee:
            def __init__(self, stream, log): self.stream, self.log = stream, log
            def write(self, text): self.stream.write(text); self.log.write(text); self.log.flush()
            def flush(self): self.stream.flush(); self.log.flush()
        log = args.log_file.open("a", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log)
        sys.stderr = Tee(sys.stderr, log)
    if not args.dry_run and importlib.util.find_spec("pip") is None:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
    if not args.dry_run and importlib.util.find_spec("packaging") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "packaging>=20"], check=True)
    from runtime import install_dependencies as module
    plan = module.dependency_plan()
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 1 if plan["status"] == "unsupported" else 0
    print("MOSS：正在准备当前环境所需依赖及匹配的可选组件。模型请在节点准备窗口下载。")
    result = module.execute_plan(plan)
    if args.result_file:
        args.result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for action in result["actions"]:
        print(action)
    if result["exit_code"] == 0:
        print("依赖准备完成。请按 Manager 提示应用更改或重启，再打开 MOSS 配置与准备。")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
