# 0.1.7 · Windows SM120 安装与实机验收

## 已完成

在用户实际的 RTX 5060 Ti 16GB / Windows 11 企业版 LTSC（26100）上，修复安装入口、完成依赖准备，并通过界面执行三条完整生成。
原安装器把 Windows venv launcher 的基础 Python 子进程也作为候选；该子进程缺少 psutil，PowerShell 收到标准错误后中断发现流程。目标 Portable 仍保留 0.1.3，已下载的 0.1.6 没有覆盖进去。
新安装器合并重复子进程、独立捕获各实例输出，多实例时明确选择端口与目录，并记录持久化日志。支持 Portable、正在运行的 Desktop/源码环境，以及外置 venv/Conda 解释器选择。

## 本机环境

ComfyUI 0.42.1、前端1.55.18、Python3.13.14、Torch2.14.0+cu130、comfy-kitchen0.2.39、Manager4.2.2；GPU为RTX5060Ti / SM120、驱动617.42。
修复准备了Triton3.8.0.post29及相应Python开发文件。现有宿主核心版本保持原样。
软件依赖及版本兼容判断使用范围，移除了插件生成的版本等值约束。Python准备范围为普通CPython >=3.12,<3.15。

## 完整生成

数据来自用户在准备窗口选择并下载的模型文件；推理输入为同一提示词、seed314159265、1024×1024 latent、batch1、16步。
DiT使用SFT Packed W5存储及INT8×INT8→INT32计算，TE为INT8存储/W8A16计算，非量化边界FP16-fixed，VAE解码FP32。
执行顺序为文本编码、DiT采样、VAE解码，输出1024×1024 PNG。显存配置6GB为进程分配预算，实卡显存16GB。

| 路径 | 后端执行秒数 | 状态 |
| --- | ---: | --- |
| auto | 89.042 | success |
| Sage QK8 / PV FP8 | 55.283 | success |
| CK INT8 | 34.715 | success |

时间取自ComfyUI history的execution_start到execution_success。各次缓存和编译条件不同，不计算加速倍数。
三张图均已打开检查，图像包含河道、建筑、柳树和日出，无黑图或纯噪声结果；此处未执行广泛图像质量评测。

## 算子适配

实机执行暴露了SM120的INT8 FFN架构门禁和Sage PV16原生内核崩溃。FFN按真实INT8 MMA能力适配，保留存储与舍入规则。
上游PV16 CUDA wheel的内核属于SM8x；SM120选择其原生支持的Sage QK8/PV FP8，使用per-warp量化。Kitchen INT8也完成整图验收。
SM89保留既有PV16加速，SM120保留FP8、Kitchen和公共Flash入口。硬件不适配选项在界面给出具体选择，不触发重复依赖安装。

## 依赖与资源

没有执行单元测试或架构参数矩阵。依赖审计没有新增冲突；宿主原先缺少的三个workflow-templates媒体包保持记录，未宣称全局pip check通过。
模型不随插件安装；DiT、TE、VAE仍按ComfyUI标准目录与extra_model_paths.yaml读取，可选下载按文件勾选。
SM75和Python3.14的准备策略已实现及审查，本轮实卡是SM120；此前SM89记录见[0.1.6验收](ACCEPTANCE_0.1.6.md)。

## 输出核对

**auto**

Prompt ID：`39b35d14-cfeb-413c-ac16-b16da0c0554e`

PNG SHA256：`d4e37ce30d8e4e81637af87c732d844f17745191049716808e8b875bd3639ebd`

**Sage QK8 / PV FP8**

Prompt ID：`45e78a97-3c3a-4ee1-a9da-74bb78722834`

PNG SHA256：`dffbe87c9c0befe80f2404f4667a9532695611332bd4f692559df3fba005cad9`

**CK INT8**

Prompt ID：`9f3878c5-3bd4-4eab-9759-3e548665b023`

PNG SHA256：`dd9ed09267f868a19903250d442d7bb07e38ef51067bdc0ddbf5180268fdbaf1`

完整离线图像报告和结构化回执随[0.1.7发行版](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/tag/v0.1.7)提供。
