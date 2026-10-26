# MOSS LowVRAM Optimized

使用 **KohakuMoss** 推理组件的 ComfyUI 节点包，提供 MOSS Base/SFT 多精度推理、显存预算管理和模型资源准备。

## 安装与首次运行

### 通过工作流安装

1. 在已启用官方ComfyUI Manager的ComfyUI中导入唯一工作流：[00_moss_lowvram.json](example_workflows/00_moss_lowvram.json)。
2. 出现缺失节点提示时，点击 **查看详情 → 在管理器中查看**，找到 **MOSS LowVRAM Optimized**。
3. 打开版本选择器，选择最新版本并安装，然后点击 **应用更改 / 重启**。
4. 在 **MOSS · 配置与准备** 节点选择模型、计算方式与显存预算，点击 **Prepare / check selected configuration**。
5. 查看资源计划并确认准备。模型就绪后点击ComfyUI的运行按钮。首次运行会编译内核，请等待完成。

### 命令行安装

已安装[comfy-cli](https://docs.comfy.org/comfy-cli/getting-started)和[ComfyUI Manager](https://docs.comfy.org/manager/install)时，先激活ComfyUI使用的Python环境，再运行：

```sh
comfy --workspace "C:\ComfyUI" node install https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized
```

将路径替换为自己的ComfyUI根目录。安装完成后重启ComfyUI，导入工作流，点击 **Prepare** 准备模型。
详细步骤和依赖要求见[安装指南](docs/INSTALLATION_GUIDE.md)。

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

当前主要环境为 **Windows x64、NVIDIA RTX40 / SM89、普通CPython3.12/3.13、Torch2.10.0+cu130、Triton3.6.0.post26、ComfyUI≥0.41**。
固定依赖自动准备覆盖精确匹配的宿主组合；CPU-only不可用。RTX20保留历史W5 auto范围，RTX50与Linux尚待验证。
显式A8 TE要求SM80+，FP8计算要求SM89+；完整范围见[环境支持](docs/ENVIRONMENT_SUPPORT.md)。

ComfyUI前端1.55.17的Manager详情可能在CUDA宿主上误报CPU/CUDA冲突；已有CUDA宿主可继续正常安装确认，
以配置节点Prepare的实际 `torch.cuda` / SM检查为准。

升级时保留一份插件安装，避免旧 `moss-optimized`、`ComfyUI-KohakuMOSS` 与新包重复启用。

## 分享与许可

分享00 JSON或带工作流metadata的输出PNG，接收者导入后通过Manager安装节点，再准备模型。

正式包名为 `kohaku-moss-lowvram-optimized`。代码遵循[KohakuMoss License1.0](LICENSE)，保留Kohaku品牌与组件归属。
权重及依赖采用各自许可，见[NOTICE](NOTICE)、[组件说明](THIRD_PARTY_NOTICES.md)和[来源清单](LICENSE_PROVENANCE.json)。
