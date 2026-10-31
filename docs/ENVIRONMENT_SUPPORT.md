# 环境支持

## 基本要求

- ComfyUI≥0.41、Python≥3.12、NVIDIA CUDA宿主。
- 历史记录与参考环境（不是安装门槛）：Windows x64、普通CPython3.12/3.13、Torch2.10.0+cu130、Triton3.6.0.post26、RTX4070TiSUPER / SM89。
- Manager与准备向导共用安装器；按实际Python ABI和Torch/Triton小版本关系选择兼容范围，详情见[安装指南](INSTALLATION_GUIDE.md)。

## 安装检查与完整推理分开记录

安装器支持为SM80+上的Torch2.7–2.15准备匹配Triton；这不是对所有GPU的生成质量或速度承诺。
当前官方Portable的Torch2.14.1不再被旧的Torch2.10精确匹配规则挡住。

## GPU与平台范围

| 环境 | 当前范围 |
| --- | --- |
| RTX40 / Ada、SM89 | 当前主要运行平台，已使用RTX4070TiSUPER 16GB；其他卡按资源与内核条件检查 |
| RTX20 / Turing、SM75 | Triton 3.2 系列，standalone JIT + eager，不锁 Torch 2.10；历史设备为2080Ti 22GB；当前新宿主组合未实卡复验 |
| RTX30 | 整工作流支持尚未承诺；按各计算模式的SM要求选择 |
| RTX50 / Blackwell、SM120 | 跟随宿主 Torch 小版本匹配 Triton；Blackwell 要求 Torch>=2.7、CUDA build>=12.8；完整推理待实卡验收 |
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
| Sage | SM89、Windows、Python3.12/3.13；上游稳定ABI wheel支持Torch>=2.10,<3与CUDA13，安装后检查原生扩展导入 |
| Turing Flash | 固定独立资产与精确Windows SM75 ABI，FP16限定路径 |
| VAE | FP32精度边界 |

Prepare根据实际CUDA设备、SM、依赖和所选模型检查配置。未知宿主组合会显示具体诊断。
可选注意力的来源、许可与固定资产见[组件说明](../THIRD_PARTY_NOTICES.md)和[optional_binary_manifest.json](../optional_binary_manifest.json)。

## 0.1.5 首次安装验收记录

2026-10-30：Windows 10、RTX4070Ti SUPER / SM89、官方 ComfyUI Portable 0.43.0、前端1.57.4、Manager4.2.2、
嵌入式Python3.13.14、Torch2.14.1+cu130、Triton3.8.0.post29、comfy-kitchen0.2.40。
独立应用环境中通过准备界面补齐运行组件，从空的MOSS模型目录下载并校验所选模型，
然后实际完成1024×1024、SFT W5、16步、默认6GB进程预算的生成，后端返回success并保存PNG。
首次执行146.308秒，包含冷启动编译；它不是稳定态速度基准，也不是物理6GB显卡验收。

这项记录不等于RTX50实卡、所有模型精度、Desktop以及所有源码部署组合都经过独立全新安装测试。
安装器使用实际运行的ComfyUI解释器；Windows完整包安装器的实例选择、安装位置确认与依赖准备已在隔离实例验证。

## 本次兼容更新的验收记录

实际运行数据以 `docs/ACCEPTANCE_0.1.6.md` 为准。只有 SM89 在维护机器上执行完整生成；SM75/SM120 的策略和硬件限制已实现，但不把静态检查冒称为实卡生成验收。

SM75 的编译组合算子已在本包 vendored dispatch 中明确排除现代 Inductor，保持原 dtype 的 eager 组合；独立 Triton3.2 和 INT8 GEMM 不关闭，也不全局修改其它 ComfyUI 节点的 torch.compile。该架构的新宿主组合仍需要实卡验收。
