"""ComfyUI-Manager install phase only: prepare narrowly supported prerequisites."""
import importlib.util
import json
from pathlib import Path
import sys


def main(argv=None):
    spec = importlib.util.spec_from_file_location("moss_install_dependencies", Path(__file__).resolve().parent / "runtime/install_dependencies.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.dependency_plan()
    if "--dry-run" in (sys.argv[1:] if argv is None else argv):
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 1 if plan["status"] == "unsupported" else 0
    print("MOSS：正在准备当前环境所需依赖及匹配的可选组件。模型请在节点准备窗口下载。")
    result = module.execute_plan(plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for action in result["actions"]:
        print(action)
    if result["exit_code"] == 0:
        print("依赖准备完成。请按 Manager 提示应用更改或重启，再打开 MOSS 配置与准备。")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
