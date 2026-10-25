# 通过 ComfyUI Manager 安装并准备依赖

导入工作流后，在 Manager 中安装缺失节点。Manager 先处理本插件的
`requirements.txt`，随后用 **当前 ComfyUI 的 Python** 执行 `install.py`。
Windows 上官方 Manager 4.2.2 会先将这些步骤排入下次启动的任务文件；
UI 队列显示成功可能仅表示已安排，尚未安装。请点击 Restart / 重启 ComfyUI，
等待启动依赖处理完成，再打开 MOSS 节点环境检查，
按引导准备模型。本安装钩子不下载模型。精确匹配的SM89宿主还会尝试准备固定公开Sage wheel到插件隔离目录，失败仅提示warning；SM75仅在真实公开清单完整后准备单独的Turing包；不下载占位清单或额外SM89 Flash包。

如果节点已存在但依赖不完整，先重启以完成已排队的步骤，保留安装日志中的
具体错误码。然后查看当前 Manager 实际提供的修复/重装入口。4.2.2 包内旧界面
名称为 **Try fix / Reinstall**；默认新界面由独立 ComfyUI frontend 提供，
不能仅凭 Manager 版本承诺存在名为“Fix Dependencies”的按钮。4.2.2 的 fix
操作要求 high+ 权限，默认 normal 策略可能拒绝；请使用官方宿主维护流程，
不要为此降低安全级别。无需在节点面板输入终端命令。

## 统一节点内准备模型

默认打开 [00_moss_lowvram.json](../example_workflows/00_moss_lowvram.json)，在
**MOSS · 配置与准备** 节点选择系列、DiT存储/计算、TE格式/计算与VAE，点击
**Prepare / check selected configuration**。弹窗列出所选资产的大小、来源、
目标位置及已有文件状态；确认后只下载当前所选资产。Original TE按完整资产
准备分片、index/config与tokenizer。切换配置后刷新并重新确认。常规流程无需CLI。

当前00默认SFT W5 + Sumomo INT8 TE v2 / W8A16 +原MOSS VAE；00是唯一用户模板。
旧01–07移至tests/historical_workflows，仅保留为历史测试fixture，不进入Registry包。Base/SFT row共两份FP8权重，对同一文件切换FP8或16-bit计算；
旧独立weight-only文件仅legacy local，不重复作为第三份公开下载。新FP8公开下载
须等实际上传完成且固定revision/hash进入清单。

## 已完成和待完成的安装验证

Windows独立基线（Python3.12.13、Torch2.10+cu130、Manager4.2.2，初始缺少
Triton和accelerate）已直接运行requirements处理与install.py，退出码均为0。
此前delta04的Windows137项unit通过、0skip，GUI相关32项通过；冻结Turing helper另有
Windows8项focused及真实17成员ZIP纯校验通过，没有把focused当作重新运行137项或SM75 GPU验证。新00候选服务GUI19检查与准备作业全部ready，复用三份预存模型，
没有模型网络下载；默认SFT W5 / TE W8A16的1024²、16步真实prompt成功输出PNG。
这与干净Python的依赖准备分别记录，也不等于空基线从Manager安装缺失插件。
真正Manager发现→安装→Restart→导入00的闭环仍待完成，不把本地预先放置候选算作新装。
详见 [统一00验证及边界](GUIDED_SETUP_VALIDATION.md)。

## 自动准备的边界

| 当前宿主 | 安装行为 | 验证范围 |
| --- | --- | --- |
| Windows x64、Python 3.12、Torch 2.10.0+cu130、SM89 | 缺少 Triton 时安装 `triton-windows==3.6.0.post26` | 使用既定配对；依赖准备成功仍需节点检查和实际工作流验证 |
| 同上、SM75 | 缺少 Triton 时安装 `triton-windows==3.2.0.post21` | 历史实测的非上游标准 Torch/Triton 配对；本次发布尚未重新验证该卡整流程 |
| Linux NVIDIA、Python >=3.12 | 只接受现有 Torch 包元数据中可判定适用的精确 `triton==版本` 依赖 | 不宣称 Linux 整工作流已实测；缺少开发头文件/C 编译器时停止 |
| 未知版本、其他 Windows Python/Torch/CUDA/SM、不可用 CUDA、已安装冲突 Triton | 返回明确“不支持自动准备”的诊断，不执行安装 | 通过 [官方 ComfyUI Desktop](https://www.comfy.org/download) 图形安装/维护流程准备宿主，或由管理员维护 |

安装使用同一解释器、固定版本、仅 wheel、`--no-deps`，不会升级/卸载/替换
Torch、CUDA、Python，不修改全局环境变量。已匹配的 Triton 不重复安装。
安装阶段会检查 Python 开发文件、Triton wheel 的 TinyCC/ptxas 文件，最后
用独立 Python 进程导入 Triton 原生库；不运行 GPU kernel。原生库导入失败
（例如 Visual C++ 运行库缺失）会如实使安装失败，请使用宿主图形维护流程修复。

## SM89可选Sage的安装阶段准备

在上述Windows x64 / Python3.12 / Torch2.10.0+cu130 / CUDA build13.0 / SM89
组合中，Manager安装钩子会尝试准备固定公开SageAttention wheel。
它不是默认auto的必要依赖，`auto`仍使用PyTorch SDPA。此步骤不通过运行期路由执行，
不使用全局pip，不更换Torch，不增加模型下载。

- 官方固定资产：[SageAttention v2.2.0-windows.post6 wheel](https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl)。
- 精确大小：16,656,067 bytes。
- SHA256：`1635283f5c01ec3cda58a784d0d7eabbcaffaf9511d1b263db4750e1ed7958bb`。
- 目标：本插件的 `runtime/vendor/sageattn220/`，不是Python全局site-packages。

下载校验后复用 `tools/install_optional_sage.py` 的身份、ABI、路径检查与staging/rename安装。
已有同来源provenance且必需文件完整时直接复用；未知/不完整目录不覆盖。安装后仅在独立进程
验证Sage及SM89原生模块可导入，不运行GPU kernel。下载/校验/原生导入失败会记录
`optional_sage_preparation_failed`，不伪称Sage已就绪，也不让默认auto因此失败。
失败时保持auto；后续只在Manager安装/维护阶段重试。其他宿主/架构跳过并说明原因。
安装成功是依赖准备结果，不代替完整采样或画质/速度验证。

## SM75可选Turing包：固定公开清单驱动

用户已明确授权分发自编Turing bundle；这不改变上游源码的真实许可状态。
安装器使用 `optional_binary_manifest.json` 独立记录来源commit、原始许可状态、
实际组件LICENSE/NOTICE、构建记录与每个ZIP成员的精确bytes/SHA256。
真实本地ZIP为43,707,834 bytes，SHA256与完整17成员表已写入独立清单；
独立Release已公开、非draft；root匿名HTTP200完整下载的大小/SHA与远端digest全部匹配，
清单已启用public。这个公开身份/完整性结果不声称完成新SM75 GPU工作流验证。

只有Windows x64 / Python3.12 / Torch2.10.0+cu130 / CUDA13.0 / SM75匹配才计划安装；
SM89不会安装或导入Turing包。ZIP根直接对应 `runtime/vendor/turing_bundle/`。
整体ZIP和每个成员在写入前验证，拒绝穿越、符号链接、大小写冲突与未知成员；
保留原source/license/NOTICE/build内容，staging后发布目录，已有未知目录不覆盖。
本机probe仅导入扩展及接口，不构造adapter、不执行GPU kernel。失败单独记为
`optional_turing_preparation_failed` 并保持auto；不会改Torch/Python/CUDA。

运行时环境检查要求固定清单中的文件完整匹配，再把bundle视为已安装；仅有同名目录
不能让SM75 Flash显示可用。`installed`为文件/来源检查，`available`仅为安装阶段
原生导入通过，均不代表本次SM75采样/画质/速度验证。实际adapter仍限定FP16、
全dense单段、THD H24/D128，不承诺SM89或BF16。公开资产与许可事实完善后再更新此状态。

## 便携 Python 3.12 开发文件

官方便携 CPython 的 `python312._pth` 存在，且缺少 `include/Python.h` 或
`libs/python312.lib` 时，安装阶段会补充上游 Triton Windows README 指向的
[Python 3.12.7 include/libs 包](https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/python_3.12.7_include_libs.zip)。
上游说明开发文件的 Python 次版本必须一致、补丁版本可不同。

固定 SHA256：`145cde69c753dd5a78f4525ef1f5cbe0f2642678fdefd9e79e2bbea1a91d80ea`。
此摘要由维护者于 2026-10-25 对官方发布资产直接下载核验；该旧资产 API 未提供摘要。
安装器校验大小/摘要后，只允许当前解释器目录下的 `include/` 和 `libs/` 路径，
拒绝目录穿越/符号链接，不覆盖已有文件，不修改 `Lib/`。
非便携 Python 不会自动补充这些文件。

## Manager 4.2.2 生命周期与发现限制

已检查官方 PyPI `comfyui_manager-4.2.2-py3-none-any.whl`：
`glob/manager_core.py` 的 `execute_install_script` 先处理 requirements、再安排
`[sys.executable, "install.py"]`；`try_install_script` 在 Windows 默认保存到
`user/__manager/startup-scripts/install-scripts.txt`。下一次启动，Manager
prestartup 顺序执行这些脚本，之后 ComfyUI 才导入自定义节点。安装失败会输出
`management script failed`；Manager UI 先前显示的排队成功不能代替安装日志和
重启后环境检查。开启可选统一依赖解析时，依赖处理路径有所不同，需另行验证。

默认新 Manager 使用 `/v2/manager/queue/task`，不提供旧界面独有的
`/v2/customnode/install/git_url`。未上 Registry、但已经收录官方
custom-node-list 的 GitHub 包可以按 `unknown` 版本解析；一个仅公开 GitHub、
尚未进入任何 Manager 清单/Registry 的新包，不能保证由默认新界面发现并安装。
需要先完成官方收录/注册，才可宣称“导入工作流 → 安装缺失节点”的用户闭环。

## 上游依据

- [Triton Windows README](https://github.com/triton-lang/triton-windows/blob/readme/README.md)：Torch 2.10 对应 Triton 3.6；Triton 3.3 起移除 Turing；post13 起打包 TinyCC，post11 起打包最小 CUDA 工具。
- [3.6.0.post26 PyPI 文件](https://pypi.org/project/triton-windows/3.6.0.post26/)：存在 CPython 3.12 Windows x64 wheel，SHA256 `189d8c57911aa9d2ff983a715e5c967b325f576307db60924cab22b501a36515`。
- [3.2.0.post21 PyPI 文件](https://pypi.org/project/triton-windows/3.2.0.post21/)：存在 CPython 3.12 Windows x64 wheel，SHA256 `f0294d4f85e98e735942d4db0771c4f05d7b13a481c99b520f1d9701f3e83966`。

## 后端计划接口（维护者）

`runtime.install_dependencies.dependency_plan(report=None)` 是只读接口；省略
report 时读取当前环境，传入 report 时使用环境检查结构，可附加
`install_prerequisites`、`torch_requires` 与 `sage_installation` 供测试。返回 JSON 结构：

- `schema_version: 1`、`status: ready | action_required | unsupported`。
- `read_only: true`、`execution_phase: manager_install_only`。
- `scope: automatic_dependency_preparation`；`runtime_readiness: see_environment_evaluation`。
  `unsupported` 仅表示不支持自动准备，不能直接用作运行时 blocked：已安装且可用
  的自定义编译器/其他版本仍应由独立 environment evaluation 判断。
- `steps` 为必要依赖安装步骤（`python_headers`、`pip`）；`optional_steps`为固定Sage准备，
  `optional_dependencies.sageattention`及`turing_flash`分别报告missing/installed/available/failed/conflict/skipped（Turing还可为publish_pending）。
  installed仅表示provenance与文件存在，available表示安装阶段独立原生导入检查通过。
  `errors`/`warnings` 为 `{code, message}` 对象；`actions` 为用户指导字符串。
- `qualification` 明确依赖准备不等于工作流验证；`prerequisites` 为本机检查事实。

`execute_plan(plan)` 仅供 `install.py`，重新读取宿主、拒绝过期/篡改步骤，返回
`exit_code`、`restart_required` 和结构化结果。不得通过 HTTP 路由或推理调用。
安装钩子可用 `--dry-run` 输出计划而不执行，供维护/自动化检查。
