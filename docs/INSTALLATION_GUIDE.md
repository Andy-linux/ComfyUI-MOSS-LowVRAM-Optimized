# 安装与使用

## ComfyUI Manager安装

1. 在ComfyUI导入 [00_moss_lowvram.json](../example_workflows/00_moss_lowvram.json)。
2. 缺失节点提示中点击 **查看详情**，再点击 **在管理器中查看**。
3. 找到 **MOSS LowVRAM Optimized**，打开版本选择器，选择最新版本并点击安装。
4. 点击 **应用更改 / 重启**，等待启动依赖处理完成。
5. 打开 **MOSS · 配置与准备**，选择所需配置，点击 **Prepare / check selected configuration**。
6. 确认模型来源、大小、位置和准备计划；资源就绪后运行工作流。首次运行会编译内核，请等待完成。

Manager在安装阶段处理requirements并执行install.py。Windows可能把依赖步骤安排到下一次启动，安装后应完成重启。
Registry索引更新期间，卡片与版本选择器显示可能不同；以选择器中的目标版本为准。

## 模型准备

默认配置是SFT W5、Sumomo INT8 TE v2 / W8A16、auto注意力与原MOSS FP32 VAE。
默认SFT采样为1024×1024、16步；Base建议50步，建议参数通过按钮显式应用。

模型写入ComfyUI标准目录，也可复用已注册额外目录中的匹配文件。Prepare后台核验完整SHA256，
支持识别子目录和改名文件。文件冲突时先查看提示，再明确选择处理方式。
Original TE下载计划包含完整分片、index/config及tokenizer。切换配置后刷新准备计划。

## 环境与可选注意力

| 宿主 | 安装准备 |
| --- | --- |
| Windows x64 / 普通CPython3.12或3.13 / Torch2.10.0+cu130 / CUDA build13.0 / SM89 | 固定Triton3.6.0.post26；默认auto注意力 |
| Windows x64 / Python3.12 / Torch2.10.0+cu130 / CUDA build13.0 / SM75 | 历史配对Triton3.2.0.post21；严格ABI匹配时准备独立Turing资产 |
| Linux NVIDIA | 按现有Torch元数据中适用的精确Triton版本准备 |

普通依赖由Manager处理，包括diffusers、transformers、accelerate、comfy-kitchen等。
安装器保持现有Torch/CUDA/Python版本；其他组合由环境检查给出诊断。便携Python缺少开发文件时，
精确匹配的安装策略会准备固定来源的include/libs文件。

默认auto使用PyTorch SDPA。匹配的Windows Python3.12 / SM89宿主可准备固定Sage wheel。Sage使用[固定wheel发行源](https://github.com/woct0rdho/SageAttention/releases/tag/v2.2.0-windows.post6)，Turing资产记录于
[optional_binary_manifest.json](../optional_binary_manifest.json)。可选组件按安装策略准备到插件隔离目录；准备失败时使用auto。
Turing仅适用Windows x64、Python3.12、Torch2.10+cu130、SM75及FP16限定路径，来源与许可见[组件说明](../THIRD_PARTY_NOTICES.md)。

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

- **节点安装后依赖仍缺：** 先完成应用更改/重启，再查看Manager安装日志。维护入口随界面而异，旧界面可能使用Try fix / Reinstall；宿主问题使用官方ComfyUI Desktop或宿主维护流程处理。
- **Manager显示CPU/CUDA冲突：** ComfyUI前端1.55.17存在详情兼容性误报。CUDA宿主可继续正常安装确认，以Prepare检查的实际CUDA/SM为准；CPU-only不可用。
- **已有模型再次出现在计划中：** 等待后台校验。匹配文件会复用，名称相同但字节不同的文件会提示冲突。
- **升级后重复节点：** 检查是否同时启用了旧包和新包，保留一份安装并备份私人模型映射。

GPU和平台要求见[环境支持](ENVIRONMENT_SUPPORT.md)。
