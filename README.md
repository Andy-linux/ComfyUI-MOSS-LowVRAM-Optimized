# MOSS LowVRAM Optimized

使用 **KohakuMoss** 推理组件的 ComfyUI 节点包，提供 MOSS Base/SFT 多精度推理、显存预算管理和模型资源准备。

## 安装与首次运行

**不需要安装系统 Python，也不需要理解量化参数。先打开你的 ComfyUI，保留工作流默认配置即可。**

### 拖入工作流

下载 [00_moss_lowvram.json](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/latest/download/00_moss_lowvram.json)，拖进 ComfyUI。
缺失节点提示里的 **5 个节点属于同一个 MOSS LowVRAM Optimized 包**。点击 **查看详情 → 安装 → 应用更改 / 重启**。

重启完成后，按键盘 **F5 / Ctrl+R 刷新网页**。准备向导会自动打开。
如果显示 **准备运行环境**，点击它，再点击 **重启并继续**；插件会为这个 ComfyUI 的 Python 补齐 Triton 和必要开发文件，不更换 Torch。
已有模型时，在向导或配置节点选择本地文件即可，**不要求下载记录或推荐文件的哈希**。
没有所需模型时展开 **可选下载助手**，只勾选需要的文件再下载；例如仅下载 VAE，不会附带下载 DiT 和文本编码器。
运行环境与模型分别准备。对应输入齐全后关闭窗口并点击顶部 **运行**。

首次生成包含内核编译。以后再次打开同一个工作流，无需重复安装或下载。

### 没有 Manager，或插件市场连接失败

下载 [完整安装包](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/latest)，解压后双击 **Install-MOSS.cmd**。
保持 ComfyUI 打开，安装器会识别它实际使用的 Python 和节点目录；有多个实例时按端口和目录选择。
Windows 虚拟环境的重复子进程会自动合并，旁边实例缺少依赖不会中断目标实例的识别。
源码／Conda 的外置环境也可在文件选择窗口指定。安装日志保存在 `%TEMP%\MOSS-install-日期时间.log`。
安装完成后关闭并重新启动 ComfyUI、刷新网页，再拖入包内 `example_workflows/00_moss_lowvram.json`，继续上面的准备向导。

这个入口不依赖插件市场，不安装第二个 Manager；缺少的 Python 依赖需要网络，模型不随插件安装。
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
本地模型由 ComfyUI 的目录检索接口查找，支持子文件夹、手动选择的名称和已有额外目录。
推理不依赖下载清单、固定文件大小或 SHA256 记录；模型格式由实际加载器检查。
DiT、文本编码器、VAE 分别按执行阶段加载：只执行文本编码时不要求另外两类文件。
可选下载助手按文件勾选，仅对其实际下载的字节做 SHA256 校验，不覆盖已有本地文件。
Original TE 使用完整的分片、配置和配套 tokenizer 目录，不能把单个分片当作整个编码器。

Base/SFT各一份FP8 row权重，可选择FP8或16-bit计算。模型清单、来源、目录及离线导出见[MODEL_FILES.md](MODEL_FILES.md)。

## 兼容要求

保留正在使用的 ComfyUI、Torch、CUDA 和核心组件，不要求它们等于某个历史版本。
SM89 / RTX40 与 SM120 / RTX50 根据 Torch 小版本选择 Triton 的兼容范围；已经满足范围的补丁版和 post 版直接复用。
例如 Torch 2.14 对应 `triton-windows>=3.8.0.post13,<3.9`，范围内版本直接复用。
SM120 需要支持 Blackwell 的 Torch/CUDA；不会安装 Turing 的旧版扩展。

SM75 / RTX20 使用 `triton-windows>=3.2.0.post13,<3.3`，采用 standalone Triton JIT 与 PyTorch eager，
不用本包的 torch.compile 路线，也不强制把 Torch 降到 2.10。BF16/FP8 不在 SM75 支持范围内。
普通 CPython 的安装范围为 `>=3.12,<3.15`。便携版按实际 ABI 补齐开发文件；3.14 的开发文件来自 Python Software Foundation 发布的 NuGet 包。
已有 `HTTP_PROXY` / `HTTPS_PROXY` 设置优先；Windows 系统代理会在准备进程中自动继承。

Diffusers 接受 `>=0.41,<1.0`，不再因人为的 `<0.42` 上限排除新版。安装器使用兼容范围和按需依赖解析，已满足范围的版本直接复用。
Sage 使用上游稳定 ABI wheel，按 CUDA12/13 选择发行物；支持普通 Python>=3.12,<3.15、Torch>=2.10,<3、SM80+。
已有全局 Sage 优先复用。PV16 CUDA wheel使用SM8x内核；SM120选择Sage QK8/PV FP8（per-warp）或CK INT8，三条默认/加速路径已在5060Ti出图。公共Flash使用PyTorch支持的SM80+实现。
可选 Turing Flash 仍保留其独立的历史 ABI 限制。两者不改变默认 auto 数值路线。
完整策略、实测环境和未做实卡验收的范围见[环境支持](docs/ENVIRONMENT_SUPPORT.md)。

显式A8 TE要求SM80+，FP8计算要求SM89+；完整范围见[环境支持](docs/ENVIRONMENT_SUPPORT.md)。

Manager 的“冲突”需要看具体原因。若显示“安全验证待定 / 无法验证兼容性”，这是 Registry 审核状态，
运行组件兼容性由准备窗口独立检查。已有版本因 Registry policy-v0.7 等待复核，跟踪见[上游复核记录](https://github.com/Comfy-Org/registry-backend/issues/311)。
遇到“没有可用发布版本”时，用上方完整包安装器更新；不要求降低 Manager 安全等级，不应为此降级 ComfyUI。

升级时保留一份插件安装，避免旧 `moss-optimized`、`ComfyUI-KohakuMOSS` 与新包重复启用。

## 实机验收

[RTX5060Ti / SM120 的0.1.7安装与生成记录](docs/ACCEPTANCE_0.1.7.md)包含实际默认SDPA、Sage FP8和Kitchen INT8的输出及日志摘要。

## 分享与许可

分享00 JSON或带工作流metadata的输出PNG。接收者安装节点后选择自己的模型；只下载所缺的组件，不要求同步整套权重。

正式包名为 `kohaku-moss-lowvram-optimized`。代码遵循[KohakuMoss License1.0](LICENSE)，保留Kohaku品牌与组件归属。
权重及依赖采用各自许可，见[NOTICE](NOTICE)、[组件说明](THIRD_PARTY_NOTICES.md)和[来源清单](LICENSE_PROVENANCE.json)。
