# 0.1.6 交付验收

## 已实际通过的流程

完整包安装器自动识别正在运行的独立 ComfyUI 解释器和节点目录，经图形界面确认后准备依赖；没有安装第二个 Manager，没有替换宿主 Torch。
模型选择与环境准备已分开：DiT、TE 使用标准目录子文件夹内的自定义文件名；仅在可选下载助手勾选 VAE，实际请求只含 `vae/moss-image-vae.safetensors`，下载 249,521,372 bytes。
生成前移除了下载校验记录。随后从界面点击运行，真实完成默认 auto 和可选 Sage QK8 PV16 两条 1024×1024、16步生成，后端均返回 success，并保存可打开的 PNG。

## 环境与范围

ComfyUI 0.43.0，官方核心 commit `6ea48ffc21d2a0c9be0df79f08d72b4ecf2d678f`，前端1.57.6，Python 3.13.14，Torch 2.14.1+cu130，Triton 3.8.0.post29，comfy-kitchen0.2.40，Windows10，RTX4070Ti SUPER / SM89。
默认组合为SFT W5、DiT INT8×INT8→INT32、TE W8A16、FP16-fixed边界、FP32 VAE；配置6GB为进程预算，实卡是16GB。
`pip check`：No broken requirements found。

| 实际运行 | 后端执行秒数 | 状态 |
| --- | ---: | --- |
| auto / SDPA | 149.653 | success |
| Sage QK8 PV16 | 26.382 | success |

前者含冷编译，后者复用了编译和文件缓存，不能据此计算Sage加速倍数，也不宣称量化注意力逐像素无损。

本轮是一次有针对性的本地模型/单项下载验收，加一次实际可选加速路径验收；没有运行单元测试或架构参数矩阵。
DiT和TE复用了已下载文件，未重复下载整套权重。模型本身及默认采样参数未改写。

## 未冒充已验收的范围

SM75使用Triton3.2范围，独立JIT和INT8 GEMM保留；组合算子不进入现代Inductor，不全局禁用其它插件的编译。
SM89/SM120按Torch小版本匹配Triton范围；SM120还要求Blackwell所需的Torch/CUDA能力。
SM75和SM120的新宿主组合本轮没有实卡生成；Desktop与源码部署共用实际解释器检测，但未把Portable验收表述成全部宿主逐一验收。

Registry“安全验证待定”与核心依赖冲突分开记录。未降低Manager安全等级。若市场没有活跃版本，使用本次GitHub完整ZIP中的Install-MOSS.cmd，按同一向导完成安装与模型选择。

## 可核验结果

默认PNG SHA256：`4ff9974f594edab9cd7ccf2f08e39d6a333fa8838951f17ac1d96e7971a48293`。Sage PNG SHA256：`ba02e997d9b092dc5d980223be1b69e147c86cb322143577ede4a88fae0cd527`。
Prompt ID分别为 `d55330ab-71fc-4c26-9547-97069f924584`、`c2582260-8577-4f1c-b44a-671f100cc5fd`。
[离线HTML与结构化回执](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/tag/v0.1.6)随发行资产提供。
