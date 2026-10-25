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
    print("MOSS：Manager 安装阶段依赖准备。不会替换 Torch/CUDA/Python，不下载模型。仅精确匹配宿主会尝试固定SM89 Sage或SM75 Turing公开资产；未发布清单不下载，失败保持auto。")
    result = module.execute_plan(plan)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for action in result["actions"]:
        print(action)
    if result["exit_code"] == 0:
        print("依赖准备完成。请在 Manager 点击 Restart / 重启，再打开 MOSS 节点环境检查；依赖齐备不等于整工作流验证。")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
