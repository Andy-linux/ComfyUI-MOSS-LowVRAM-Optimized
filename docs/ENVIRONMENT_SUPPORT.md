# 环境支持

## 基本要求

- ComfyUI≥0.41、Python≥3.12、NVIDIA CUDA宿主。
- 当前主要运行环境：Windows x64、普通CPython3.12/3.13、Torch2.10.0+cu130、Triton3.6.0.post26、RTX4070TiSUPER / SM89。
- Manager与准备向导共用安装器；按实际Python ABI和Torch/Triton小版本关系选择固定版本，详情见[安装指南](INSTALLATION_GUIDE.md)。

## 安装检查与完整推理分开记录

安装器支持为SM80+上的Torch2.10–2.14准备匹配Triton；这不是对所有GPU的生成质量或速度承诺。
当前官方Portable的Torch2.14.1不再被旧的Torch2.10精确匹配规则挡住。

## GPU与平台范围

| 环境 | 当前范围 |
| --- | --- |
| RTX40 / Ada、SM89 | 当前主要运行平台，已使用RTX4070TiSUPER 16GB；其他卡按资源与内核条件检查 |
| RTX20 / Turing、SM75 | 历史W5 auto兼容范围，历史设备为2080Ti 22GB；当前版本待复验 |
| RTX30 | 整工作流支持尚未承诺；按各计算模式的SM要求选择 |
| RTX50 / Blackwell、SM120 | Python3.12/3.13 + Torch2.14.x+cu130 的依赖准备可用；完整推理待实卡验收 |
| Linux NVIDIA | 依赖准备支持读取现有Torch的精确Triton要求；整工作流待验证 |
| CPU-only、AMD、Apple Metal、Intel加速器 | 当前不可用 |

显存profile表示进程分配预算。实际运行还需为驱动、上下文和其他进程保留显存。

## 计算模式要求

| 模式 | 条件 |
| --- | --- |
| 显式INT8 W8A8 / W4A8文本编码 | SM80+ |
| Kohaku FP8 row计算与共享16-bit模式 | 当前实现要求SM89+ |
| Original BF16 / FP16 TE | 使用对应浮点权重与计算选项 |
| auto注意力 | PyTorch SDPA |
| Sage | 固定wheel与精确匹配的Windows Python3.12 / SM89宿主 |
| Turing Flash | 固定独立资产与精确Windows SM75 ABI，FP16限定路径 |
| VAE | FP32精度边界 |

Prepare根据实际CUDA设备、SM、依赖和所选模型检查配置。未知宿主组合会显示具体诊断。
可选注意力的来源、许可与固定资产见[组件说明](../THIRD_PARTY_NOTICES.md)和[optional_binary_manifest.json](../optional_binary_manifest.json)。
