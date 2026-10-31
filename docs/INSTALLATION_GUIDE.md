# 安装与首次生成

## 普通用户：从同一个工作流开始

先启动 ComfyUI，将 [00_moss_lowvram.json](../example_workflows/00_moss_lowvram.json) 拖入窗口。
缺失节点侧栏中的 5 个节点都属于 **MOSS LowVRAM Optimized**，点击安装即可，不需要安装五个不同插件。
在 Manager 完成安装后点击 **应用更改 / 重启**。

**后端重启不等于网页扩展已经加载。** 重启完成后，按键盘 F5 / Ctrl+R 刷新网页。新的准备向导会自动打开；所有默认模型和采样选项都可以保持不动。

向导先检查运行环境。需要组件时，点击 **准备运行环境**，等界面显示完成，再点击 **重启并继续**。
程序会保留当前工作流并在重连后刷新。浏览器若询问是否离开，选择“离开”，随后会恢复工作流。这个操作使用正在运行 ComfyUI 的解释器，
不会把依赖装到另一套系统 Python，也不会更换当前 Torch。

模型与环境独立准备。已有文件时，在窗口的 DiT、文本编码器、VAE 下拉框选择它们，或在配置节点选择。
标准目录和额外目录中的文件直接读取，不需要经过下载助手。
缺文件时展开 **可选下载助手**，只勾选需要的文件，再点击 **下载勾选项目**；默认不勾选任何文件。
不必下载整套组合，也不必先安装好推理环境才能单独下载权重。

完整生成需要对应的 DiT、文本编码器和 VAE；单独使用某个执行阶段只加载它自己的文件。
运行环境与所需文件就绪后关闭窗口，点击 ComfyUI 顶部的 **运行**。
默认生成 1024×1024 图像、SFT16步，首次执行含内核编译。

## 市场不可用时：完整包安装器

从 [GitHub Releases](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/latest) 下载完整 ZIP 并解压。
保持要使用的 ComfyUI 打开，双击 **Install-MOSS.cmd**。安装器从正在运行的 ComfyUI 识别实际 Python、
工作目录以及 Desktop 的 base-directory；不会直接调用 PATH 里的 `python` 或 `pip`。
有多个实例时，按端口与目录选择对应实例。安装器会合并 Windows venv 的重复子进程；
一个旁边实例的探测失败只写入日志，其他有效实例仍可选择。

确认安装位置后，安装器复制节点包并准备依赖。完成后按提示关闭并重新打开 ComfyUI、刷新网页，
拖入包里的 `example_workflows/00_moss_lowvram.json`，选择已有模型，或按需使用可选下载助手。
没有启动 ComfyUI 时，可以通过文件夹窗口指定 Portable、Desktop 实例或源码目录。
外置 venv／Conda 布局可进一步选择该环境的 `python.exe`。
完整安装日志保存在 `%TEMP%\MOSS-install-日期时间.log`，失败窗口直接显示位置。

这条路线不依赖 Registry，不需要先安装 Manager，不会额外安装另一种 Manager。
依赖和模型分别按需联网下载。Python 依赖遵循当前 pip 配置，模型使用界面显示的来源。
显式代理环境变量优先；Windows Internet Settings 中的 HTTP/HTTPS 代理会自动用于本次准备进程。

## 官方宿主与 Python 的关系

Desktop、Portable 与源码部署都运行同一套节点，但 Python 的位置并不相同。
克隆 ComfyUI 仓库不会附带一套完整 Python；Portable 的嵌入式 Python 也默认没有开发头文件。
节点准备检查实际解释器，不把“源码部署”或“已装 Python”当成开发文件齐全的证据。

Desktop 自带并启用官方 Manager，无需重复安装。Portable / 手动部署的官方 Manager 是独立依赖，
需要按 [官方安装说明](https://docs.comfy.org/manager/install)启用。
已有可用 Manager 时直接使用，不能为了本插件再装第二份。

Portable 需要启用官方 Manager 时，在 `ComfyUI_windows_portable` 文件夹打开终端执行：

```powershell
.\python_embeded\python.exe -m pip install -r ComfyUI\manager_requirements.txt
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --enable-manager
```

源码部署由你用于启动 ComfyUI 的 Python 执行 `python -m pip install -r manager_requirements.txt`，
然后用同一个解释器运行 `python main.py --enable-manager`。不熟悉环境激活时使用上面的完整包安装器。

## 开发者的命令行入口

在实际 ComfyUI Python 环境中安装节点 `requirements.txt`，运行本包 `install.py`，重启 ComfyUI。
已经配置 comfy-cli 和 Manager 的开发者也可使用：

```sh
comfy --workspace "C:\ComfyUI" node install https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized
```

`C:\ComfyUI` 是自己的 ComfyUI 目录；这不是让普通用户寻找系统 Python 的替代流程。

## 依赖准备与推理范围

Windows 普通 CPython>=3.12,<3.15 的运行组件根据实际显卡和 Torch 小版本准备，不使用 Torch 补丁版本相等作为门槛。
SM89、SM120 的 Torch 2.10/2.11 对应 Triton 3.6，2.12/2.13 对应 3.7，2.14 对应 3.8，2.15 对应 3.9；
完整映射与来源见 `runtime/compatibility.py`。每个系列使用下限/上限范围，保留已安装且兼容的 patch/post 版本。
缺失或跨系列不匹配时，向导列出要准备的组件，用户确认后安装，重启生效。

SM75 固定在仍支持 Turing 的 Triton 3.2 系列，使用 standalone JIT 与 eager PyTorch，不使用本包的 Inductor/torch.compile 路线。
不因此锁死整个 ComfyUI 或要求 Torch 等于 2.10。这个组合不保证其他节点的 torch.compile 可以使用；
SM75 也不支持本包的 BF16、FP8 和 A8 TE。默认使用 W5、TE W8A16、auto SDPA 与 FP32 VAE。
SM120 默认使用 ring 编码和 SDPA；可显式选择 Sage、公共 Flash 或 CK INT8。
SM89 专用的驻留预算表继续按自身硬件条件工作。

Python 开发文件针对当前解释器 ABI 补齐。3.12/3.13 使用上游开发文件归档；3.14 使用 Python Software Foundation 的官方 NuGet 包，只提取 include/libs。
完整 CPython／venv 环境直接复用已有开发文件。缺少 pip 时通过当前解释器的 ensurepip 补齐。
本包的依赖安装和运行版本判断使用兼容范围；按需安装时不生成版本等值约束。
Linux 的 Triton 按 Torch 小版本选择发布范围；Windows 使用对应的 triton-windows 范围。
Sage 优先复用已有全局安装，缺失时按 CUDA12/13 准备稳定 ABI wheel 到插件目录。

默认注意力使用 PyTorch SDPA，不依赖 Sage 或第三方 Flash；可选二进制继续遵守其 ABI 条件。
依赖准备与实卡验证是两回事，已完成的实际执行范围见[环境支持](ENVIRONMENT_SUPPORT.md)。

## Manager 提示冲突或没有可用更新

先看提示正文。`安全验证待定 / 无法验证兼容性` 表示 Registry 审核状态。运行依赖的检查结果在准备窗口单独显示。
本包旧版本收到 `policy-v0.7: reflag-pending-review`，后续版本可能仍等待扫描。
维护人已提交[官方复核请求](https://github.com/Comfy-Org/registry-backend/issues/311)。
此时直接使用本指南的完整包安装器更新，不要降级 ComfyUI、重复安装 Manager 或降低其安全等级。
插件不会伪造 Registry 的 Active 状态；运行环境是否就绪由准备向导独立报告。

## 模型目录与选择

DiT 放在 `models/diffusion_models`，文本编码器放在 `models/text_encoders`，VAE 放在 `models/vae`。
遵循 ComfyUI 的注册目录检索，支持子目录和 `extra_model_paths.yaml`。添加文件后在向导点击“重新检查”，
从对应下拉框选择；自定义文件名不会因切换其它参数而被静默覆盖。

不使用 Prepare 也能运行已有模型；固定下载来源、完整 SHA256 和断点续传只属于可选下载助手。
用户自备权重不因文件名、文件大小或下载记录不符而被替换。
加载器在实际执行时检查文件格式，Original TE 必须提供完整分片、配置与 tokenizer。
文件清单和各自下载项目见[模型说明](../MODEL_FILES.md)。

## 显存分配器启动选项

ComfyUI原生cudaMallocAsync优先。需要独立启动探测与回退时，使用启动器：

```sh
python tools/launch_comfy_allocator.py --comfy-main /path/to/ComfyUI/main.py -- --listen 127.0.0.1
```

请使用启动ComfyUI的同一Python，从插件目录执行。启动器在独立进程检查后端，支持时使用ComfyUI的
`--cuda-malloc`；宿主明确不支持时在新进程选择native与现有内存管理路径。
显式恢复可加 `--backend native`，仅检查可加 `--probe-only`。启动器统一设置两种allocator环境变量，
避免互相覆盖；运行中的推理失败交由用户处理。

## 常见问题

- **安装后没有 Prepare 按钮：** 按 F5 / Ctrl+R 加载新安装的网页扩展。
- **节点已安装、环境仍缺组件：** 准备向导内点击“准备运行环境”，完成后“重启并继续”；不要重复下载模型或安装另一份 Manager。
- **Manager显示CPU/CUDA冲突：** ComfyUI前端1.55.17存在详情兼容性误报。CUDA宿主可继续正常安装确认，以Prepare检查的实际CUDA/SM为准；CPU-only不可用。
- **已有模型再次出现在计划中：** 等待后台校验。匹配文件会复用，名称相同但字节不同的文件会提示冲突。
- **升级后重复节点：** 检查是否同时启用了旧包和新包，保留一份安装并备份私人模型映射。

GPU和平台要求见[环境支持](ENVIRONMENT_SUPPORT.md)。
