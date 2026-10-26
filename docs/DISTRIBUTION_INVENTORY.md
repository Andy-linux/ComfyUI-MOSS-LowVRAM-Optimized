# 分发清单与来源边界

主展示：**MOSS LowVRAM Optimized**，使用KohakuMoss推理组件。
正式包名 `kohaku-moss-lowvram-optimized` 保留Kohaku命名；根LICENSE及原组件
LICENSE/NOTICE字节不变。此清单记录工程分发决定，不作为额外授权或法律认证。

## 分发决定

| 类别 | 当前范围 | 来源与验证 |
| --- | --- | --- |
| 节点、独立adapter、生命周期/下载工具 | GitHub / Registry源码包 | 保留原许可、来源及修改说明；六个旧class IDs兼容，新增统一配置入口 |
| Kohaku / OpenMOSS源码、必要tokenizer/config | 随源码保留必要文件 | `runtime/vendor/SOURCES.json`、`LICENSE_PROVENANCE.json`及原LICENSE/NOTICE |
| 用户模板 | 仅 `example_workflows/00_moss_lowvram.json` | GUI选择系列/格式/计算并准备所选模型；旧01–07只在tests/historical_workflows，不进Registry |
| 原有Base/SFT/TE/VAE模型 | 固定上游来源索引，不复制上传 | model_manifest.json内真实revision、bytes、SHA256与各模型许可 |
| 原有W5/W6 | 使用现有Andylinux公开仓库，不重传 | 下列已核对revision/hash；不与新FP8目标混同 |
| 新Base/SFT FP8 row | 两份文件已在Andylinux/MOSS-Image-Kohaku-FP8的固定R1公开核验 | 同一row文件切换FP8/16-bit计算；使用下列真实R1/bytes/hash |
| 旧独立Base weight-only导出 | 仅legacy local，不是第三份公开模型 | 保留历史conversion/hash记录，不重复上传 |
| 固定公开Sage wheel | 严格SM89宿主由Manager安装阶段准备到插件隔离目录 | 原发行URL、精确bytes/SHA256及原组件许可；源码包不内嵌wheel |
| 自编SM75 Turing bundle | 用户已授权且独立GitHub Release已公开；匿名完整下载大小/hash及远端digest确认 | optional_binary_manifest.json固定URL/整体及逐文件hash/严格ABI，保留源码、构建、原许可和NOTICE；上游仓库许可未确认事实不变 |
| Torch/CUDA/Python | ComfyUI宿主拥有，不随源码包分发或自动替换 | Manager仅按窄策略补Triton/开发文件；未知组合给诊断 |
| prompt预编码张量、校准cache、latent、编译cache | 不作为公开默认运行依赖 | 普通工作流实时运行TE，不要求benchmark缓存 |

## 已公开W5/W6

仓库：[Andylinux/MOSS-Image-SFT-W5A8-W6A8](https://huggingface.co/Andylinux/MOSS-Image-SFT-W5A8-W6A8)。
已核对revision：`bfb77cb73ae7acb4f788d50d4043c2d71989b0a7`。
以下公开hash与实测本地产物匹配；这不自动证明所有运行平台或GPU组合通过。

| 文件 | SHA256 |
| --- | --- |
| `q5_ridge_gptq.safetensors` | `3f8a775e179a63a7511d469f08430b89f5807ebd181f37f4bbaea7d039096f82` |
| `q6_direct.safetensors` | `dc5af59f22eb89320e457536ae90b170c6548c74a9fca584684139cf88018f87` |

此revision不被挪用成更早BF16源模型的下载revision；原始来源按实际manifest记录。

## 两份新FP8 row与历史本地产物

两份Base/SFT FP8 row已公开并固定于[R1 `d8a0b2f46a79770fe711bf5322cf6cb3219c8d41`](https://huggingface.co/Andylinux/MOSS-Image-Kohaku-FP8/tree/d8a0b2f46a79770fe711bf5322cf6cb3219c8d41)，合计10,349,759,704 bytes。已核对两文件精确大小、远端LFS SHA256与本地完整文件SHA256，并读回小型配套文件；没有重新下载约10GB远端权重做全流SHA256核验。
精确文件大小与固定revision表见[模型文件](../MODEL_FILES.md)；Registry0.1.0已提交并观察为Pending，Active与实际Manager安装仍待验证。

| 文件 | 分发范围 | SHA256 |
| --- | --- | --- |
| `moss_image_base_kohaku_fp8_row.safetensors` | Base公开row，5,174,879,864 bytes；共享FP8/16-bit计算 | `4f18284539c84c804919e7c343ee33b1550dfcd1c4cd5dc5dd7f4569905d0f14` |
| `moss_image_sft_kohaku_fp8_row.safetensors` | SFT公开row，5,174,879,840 bytes；共享FP8/16-bit计算 | `06c58d186852abd0fd073fba6486a82543a6006a8c17a3eac7d62828b9c03862` |
| `moss_image_base_kohaku_fp8_weight_only.safetensors` | legacy local，非本次第三份公开下载 | `03a31fb8a7c2e196b3b0c3b02849442d40503ead10cdf9f9f4bce543525c1c54` |

导出为格式转换/量化，没有训练。已验证共享路径的8个512²/2步/解码请求及263层
q/scale不变范围，不扩大为全步数质量/速度benchmark。历史conversion记录及未确认事实
保持原样，不用猜测日期、设备或source revision补齐。

## 可选二进制与真实许可状态

Sage原发行页：
<https://github.com/woct0rdho/SageAttention/releases/tag/v2.2.0-windows.post6>。
SM89使用固定wheel，不镜像成新的自有构建，不把一种wheel的兼容性推广到所有后端。

Turing源commit：`9ef98fcb506bb1e2fe3cece50935e2935bf6b124`；
CUTLASS：`df18f5e4f5de76bed8be1de8e4c245f2f5ec3020`。
真实记录未发现该Turing HEAD的LICENSE/仓库许可声明；可用CUTLASS/NVIDIA原文只覆盖
各自组件，不被描述为整个实现/组合.pyd的开放许可。用户明确授权分发不改变此事实。
独立Release保留完整源归档、无源码patch的记录、构建环境与参数、可移植manifest、
二进制/源文件hash、全部实际组件LICENSE/NOTICE，排除本地绝对路径、GPU UUID、私有凭证和cache。

代码包仍不内嵌.pyd/.dll/.whl。Manager仅在固定清单完整且严格Winx64/Python3.12/
Torch2.10+cu130/SM75时准备Turing资产；SM89跳过。缺清单或失败保留auto，不伪称Flash就绪。
上游源码许可未确认、用户分发授权、文件完整性、原生导入和GPU工作流资格是不同事实。
