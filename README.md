# MOSS LowVRAM Optimized

使用 **KohakuMoss** 推理组件的 ComfyUI 节点包，提供 MOSS Base/SFT 多精度推理、显存预算管理和模型资源准备。

## 安装与首次运行

**不需要安装系统 Python，也不需要理解量化参数。先打开你的 ComfyUI，保留工作流默认配置即可。**

### 拖入工作流

下载 [00_moss_lowvram.json](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/latest/download/00_moss_lowvram.json)，拖进 ComfyUI。
缺失节点提示里的 **5 个节点属于同一个 MOSS LowVRAM Optimized 包**。点击 **查看详情 → 安装 → 应用更改 / 重启**。

重启完成后，按键盘 **F5 / Ctrl+R 刷新网页**。准备向导会自动打开。
如果显示 **准备运行环境**，点击它，再点击 **重启并继续**；插件会为这个 ComfyUI 的 Python 补齐 Triton 和必要开发文件，不更换 Torch。
环境就绪后点击 **确认下载 / 校验**。看到 **准备完成**，关闭窗口并点击顶部 **运行**。

首次生成包含内核编译。以后再次打开同一个工作流，无需重复安装或下载。

### 没有 Manager，或插件市场连接失败

下载 [完整安装包](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/latest)，解压后双击 **Install-MOSS.cmd**。
保持 ComfyUI 打开，安装器会识别它实际使用的 Python 和节点目录；有多个实例时会让你选择。
安装完成后关闭并重新启动 ComfyUI、刷新网页，再拖入包内 `example_workflows/00_moss_lowvram.json`，继续上面的准备向导。

这个入口不依赖插件市场，不安装第二个 Manager；仍需要网络下载 Python 依赖和模型。
Windows Desktop、Windows Portable 和源码部署使用同一套节点和准备向导。
Desktop 自带 Manager；Portable / 源码部署不保证已启用 Manager，不能靠一个 JSON 安装 Manager。

[详细安装说明](docs/INSTALLATION_GUIDE.md)包含官方 Manager 启用方式、命令行部署和平台边界。

## 配置选项

| 配置 | 选项 |
| --- | --- |
| 模型系列 | Base / SFT |
| DiT权重 | SFT W5/W6、Base/SFT INT8、Base/SFT Kohaku FP8 row |
| FP8 row计算 | FP8激活 / 16-bit激活；共享同一份模型文件 |
| 文本编码器 | INT8 W8A16、INT8 W8A8、W4 W4A16/W4A8、Original BF16/FP16 |
| 注意力 | auto（PyTorch SDPA）、符合宿主条件的Sage / Flash |
| VAE | 默认原MOSS VAE，FP32 |
| 显存 | 按配置选择预算profile |

默认组合为 **SFT W5 + Sumomo INT8 TE v2 / W8A16 + auto + FP32 VAE**，1024×1024、16步。
Base建议50步。切换系列保留提示词、seed与采样参数；需要时点击 **Apply recommended sampling parameters** 应用建议值。

## 模型与目录

模型使用ComfyUI的 `models/diffusion_models/`、`models/text_encoders/`、`models/vae/`，并识别已注册的 `extra_model_paths.yaml`。
Prepare会校验并复用匹配文件，只下载当前所选组合缺少的资源。Original TE按完整目录准备分片、配置和tokenizer。

Base/SFT各一份FP8 row权重，可选择FP8或16-bit计算。模型清单、来源、目录及离线导出见[MODEL_FILES.md](MODEL_FILES.md)。

## 兼容要求

Windows 安装器按 **当前 Torch 的小版本**匹配 Triton，不再把 RTX40 固定绑定到 Torch 2.10，
也不把 Torch 的补丁升级当成不兼容。支持普通 CPython 3.12/3.13，便携解释器缺少的 `include/libs` 自动准备。
当前官方 Portable 的 Python 3.13 / Torch 2.14.x + CUDA 13 使用 Triton 3.8。
依赖能够安装与特定显卡完整推理已验收是不同状态，实际验收范围见[环境支持](docs/ENVIRONMENT_SUPPORT.md)。

显式A8 TE要求SM80+，FP8计算要求SM89+；完整范围见[环境支持](docs/ENVIRONMENT_SUPPORT.md)。

ComfyUI前端1.55.17的Manager详情可能在CUDA宿主上误报CPU/CUDA冲突；已有CUDA宿主可继续正常安装确认，
以配置节点Prepare的实际 `torch.cuda` / SM检查为准。

升级时保留一份插件安装，避免旧 `moss-optimized`、`ComfyUI-KohakuMOSS` 与新包重复启用。

## 分享与许可

分享00 JSON或带工作流metadata的输出PNG，接收者导入后通过Manager安装节点，再准备模型。

正式包名为 `kohaku-moss-lowvram-optimized`。代码遵循[KohakuMoss License1.0](LICENSE)，保留Kohaku品牌与组件归属。
权重及依赖采用各自许可，见[NOTICE](NOTICE)、[组件说明](THIRD_PARTY_NOTICES.md)和[来源清单](LICENSE_PROVENANCE.json)。
