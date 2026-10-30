# 安装与首次生成

## 普通用户：从同一个工作流开始

先启动 ComfyUI，将 [00_moss_lowvram.json](../example_workflows/00_moss_lowvram.json) 拖入窗口。
缺失节点侧栏中的 5 个节点都属于 **MOSS LowVRAM Optimized**，点击安装即可，不需要安装五个不同插件。
在 Manager 完成安装后点击 **应用更改 / 重启**。

**后端重启不等于网页扩展已经加载。** 重启完成后，点击工作流顶部的“刷新并继续准备”链接，
或按 F5 / Ctrl+R。新的准备向导会自动打开；所有默认模型和采样选项都可以保持不动。

向导先检查运行环境。需要组件时，点击 **准备运行环境**，等界面显示完成，再点击 **重启并继续**。
程序会保留当前工作流并在重连后刷新。浏览器若询问是否离开，选择“离开”，随后会恢复工作流。这个操作使用正在运行 ComfyUI 的解释器，
不会把依赖装到另一套系统 Python，也不会更换当前 Torch。

环境就绪后，核对模型大小和来源并点击 **确认下载 / 校验**。只有环境就绪才开放这一步。
看到 **准备完成**后关闭窗口，点击 ComfyUI 顶部的 **运行**。
默认生成 1024×1024 图像、SFT 16 步，第一次还包括内核编译。

## 市场不可用时：完整包安装器

从 [GitHub Releases](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/latest) 下载完整 ZIP 并解压。
保持要使用的 ComfyUI 打开，双击 **Install-MOSS.cmd**。安装器从正在运行的 ComfyUI 识别实际 Python、
工作目录以及 Desktop 的 base-directory；不会直接调用 PATH 里的 `python` 或 `pip`。
有多个实例时，在窗口选择对应实例。

确认安装位置后，安装器复制节点包并准备依赖。完成后按提示关闭并重新打开 ComfyUI、刷新网页，
拖入包里的 `example_workflows/00_moss_lowvram.json`，按准备向导下载模型。
没有启动 ComfyUI 时，可以通过文件夹选择窗口指定 Portable 根目录或带 `.venv` / `venv` 的源码目录。

这条路线不依赖 Registry，不需要先安装 Manager，不会额外安装另一种 Manager。
它不是完全离线安装包：Python 依赖来自 PyPI，模型来自工作流显示的固定来源。

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

Windows 的普通 CPython 3.12/3.13、SM80+ GPU 按上游 Torch/Triton 小版本关系准备：
Torch 2.10/2.11 → Triton 3.6，Torch 2.12/2.13 → Triton 3.7，Torch 2.14 → Triton 3.8。
固定 wheel 来自 PyPI；不把 2.14.0 和 2.14.1 划成两种 GPU 支持范围。
Python 3.12/3.13 便携版缺少开发文件时，从上游固定校验和的压缩包准备 `include` / `libs`。
无须另装系统 Python、Visual Studio 或完整 CUDA Toolkit 来使用默认 Triton JIT 路径。

SM75 保留历史 Python3.12 / Torch2.10 / Triton3.2 组合，不能安装已移除该架构支持的新版 Triton。
Linux 从当前 Torch 包元数据读取它要求的 Triton 版本；需要已有 Python 开发文件和 C 编译器。
依赖准备规则不等于所有显卡都已实测，具体见[环境支持](ENVIRONMENT_SUPPORT.md)。

默认注意力使用 PyTorch SDPA，不依赖 Sage 或第三方 Flash。
可选加速组件保持其原有 ABI 限制，不为装一个可选组件而降级官方宿主。
来源与许可见[组件说明](../THIRD_PARTY_NOTICES.md)。

## 模型准备

默认是 SFT W5、Sumomo INT8 TE v2 / W8A16、auto 注意力和原 MOSS FP32 VAE。
模型写入 ComfyUI 标准目录，可复用 `extra_model_paths.yaml` 已注册目录中的匹配文件。
大文件分段下载并保留断点，网络中断后再次确认准备会从已完成字节继续。准备过程校验完整 SHA256；文件冲突先显示处理方式，不静默覆盖。
Original TE 下载完整分片、index/config 和 tokenizer。切换配置后，准备向导显示对应资源。

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

- **安装后没有 Prepare 按钮：** 点击工作流顶部“刷新并继续准备”，或按 F5 / Ctrl+R 加载新安装的网页扩展。
- **节点已安装、环境仍缺组件：** 准备向导内点击“准备运行环境”，完成后“重启并继续”；不要重复下载模型或安装另一份 Manager。
- **Manager显示CPU/CUDA冲突：** ComfyUI前端1.55.17存在详情兼容性误报。CUDA宿主可继续正常安装确认，以Prepare检查的实际CUDA/SM为准；CPU-only不可用。
- **已有模型再次出现在计划中：** 等待后台校验。匹配文件会复用，名称相同但字节不同的文件会提示冲突。
- **升级后重复节点：** 检查是否同时启用了旧包和新包，保留一份安装并备份私人模型映射。

GPU和平台要求见[环境支持](ENVIRONMENT_SUPPORT.md)。
