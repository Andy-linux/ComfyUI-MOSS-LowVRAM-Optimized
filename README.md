# MOSS LowVRAM Optimized

使用 **KohakuMoss** 推理组件

本项目面向 **ComfyUI**，提供 MOSS 多精度推理、显存预算管理、权重搬运调度及工作流集成。

当前实现复用了 **KohakuMoss** 的模型与文本编码等组件，并使用 **OpenMOSS**、**ComfyUI**、**ComfyKitchen** 等项目的相关实现。原始许可证、版权和修改记录继续保留。

正式分发包名为 `kohaku-moss-lowvram-optimized`；`Kohaku` 前缀保留上游许可第10节要求。内部节点 ID 保持不变，节点显示采用 `MOSS · …`，分类为 `MOSS/Optimized`。

本项目执行量化权重加载、文本编码与图像推理；离线 FP8 导出是格式转换／量化，不进行训练。

代码包不包含模型权重、Torch/CUDA、编译器或可选注意力二进制。当前发布候选的实际验证环境为 Windows、RTX 4070 Ti SUPER（SM89，16 GB）、Python 3.12、Torch 2.10.0+cu130、ComfyUI 0.41.0。具体通过的组合和限制见 `RELEASE_VALIDATION.md`；显存 profile 是同一张卡上的进程预算，不代表已经在每种物理小显存卡上验收。

## 模型与计算模式

| 部分 | 选择 | 实际含义 |
|---|---|---|
| DiT | SFT W5 / W6 | 原 packed 整数码，解包为 INT8，动态 A8，INT32 GEMM |
| DiT | SFT W8 | SFT ConvRot INT8，动态 A8，INT32 GEMM |
| DiT | Base W8A8 INT8 ConvRot | Base 文件自身的完整参数；兼容 separate Q/K/V 与 Comfy fused QKV |
| DiT | Base / SFT Kohaku FP8 | 识别 Kohaku stored FP8 recipe；也识别明确带 metadata 的 weight-only FP8 |
| TE | INT8 ConvRot · W8A16 | 保留原 INT8 codes、scale、ConvRot，16 位激活 |
| TE | INT8 ConvRot · W8A8 | 旋转后按 token 动态量化激活；INT8 × INT8 → INT32 |
| TE | W4 GPTQ · W4A8 | 原 W4 格式解码到原 INT8 grid，再执行真实 A8 GEMM |
| TE | Original BF16 / FP16 | 原始浮点 checkpoint，按选择的 dtype 计算；支持 HF 分片目录和单文件 |

旧工作流的 `INT8 ConvRot` 和 `W4A8 GPTQ` 字符串仍有效，保留旧的 **A16** 计算路径。旧 W4 文件名中的 W4A8 描述存储协议，不能证明旧运行器执行了 A8。新增带精确位宽标签的选项才切换到显式 A8。这个变化可能影响 conditioning、文字、细节和构图，不是无损提速保证。

TE 的 conditioning 输出边界仍为 CPU FP16，与原始权重 dtype、线性层激活 dtype 分开记录。Original 的 `te_memory=auto` 使用 ring；不会一次将整个原始 TE、视觉分支和 LM head 放入 GPU。

Kohaku 的 `fp8_compute` 提供激活量化的 FP8 路径；`fp8_weight_only` 保持 A16。实际计算仍由上游按形状选择，也可能走反量化后的 16 位分支。一个 FP8 checkpoint 的存储 dtype 不能单独证明每个形状都执行 FP8 Tensor Core。诊断见 `tools/probe_fp8_dispatch.py`。不支持没有格式说明、随意改 dtype 的“FP8 文件”。

## 安装与依赖

1. 将本目录放入 `ComfyUI/custom_nodes/kohaku-moss-lowvram-optimized/`，或者在真实 Registry 发布后通过 Manager 安装。
2. 使用 **启动 ComfyUI 的同一个 Python** 安装 `requirements.txt`。
3. 使用 Python 3.12 或更高版本，按显卡、平台和 Torch/CUDA 组合准备 Triton。RTX20 的历史实测组合使用 Triton 3.2；RTX40 的当前实测组合使用 Triton 3.6，不能给两者统一安装最新版。具体状态见 [环境支持表](docs/ENVIRONMENT_SUPPORT.md)。
4. 准备模型文件，重启 ComfyUI，在模板浏览器打开本节点的示例。

**升级已有安装时先检查重复包。** 六个内部node class IDs保持不变；不要同时启用
旧`moss-optimized`、`ComfyUI-KohakuMOSS`目录和新的Registry包，否则可能重复注册
同一组class IDs。更新或迁移现有一份安装即可。本机已有的`moss-optimized`目录可
为启动器兼容继续使用，目录名不必等于新的仓库/Registry品牌。需要迁移时先私下
备份`models.local.json`，再按需要恢复到唯一启用的安装；不要把私人映射提交发布。

```bash
python -m pip install -r requirements.txt
python tools/check_environment.py
```

当前包要求 **Python ≥3.12**。普通依赖包含`accelerate`、transformers、safetensors、
einops、NumPy、Pillow和comfy-kitchen；`requirements.txt`与pyproject保持一致。
Python版本下界是安装条件，当前实际验收版本仍为3.12，不代表未来所有Python/
Torch组合已测试。Triton按平台显式准备，不会把现有Torch/CUDA替换成另一套。

显式 A8 TE 需要 CUDA SM80+；本候选的 FP8 计算要求 SM89+。低预算双缓冲及 Sage 等新增路径实际验收以 SM89 为准。可选 Flash/Sage 不会在导入或运行节点时自动安装。`auto` 使用 PyTorch SDPA；`Flash` 只允许实际 Flash 内核，不会悄悄换成其他注意力后端。详见 `docs/portable_attention_dependencies.md`。

## 模型下载与路径

标准目录：

```text
ComfyUI/models/diffusion_models/
ComfyUI/models/text_encoders/
ComfyUI/models/vae/
```

节点通过 ComfyUI 的 `folder_paths` 查找文件，支持 `extra_model_paths.yaml`。代码没有固定 F 盘路径。已有模型可在节点的 checkpoint 下拉框中选择；也可在不分发的 `models.local.json` 中为 asset ID 配置本机路径。文件模式、Base/SFT 来源要与所选模式一致，两者相同的 shape 不能用来判定权重身份。

`model_manifest.json` 为公开文件记录完整 HF revision、精确字节数、SHA256、目标位置、格式和来源许可。下载器是独立的显式操作：

```bash
# 只查看需要哪些文件以及大小；不下载、不写入
python tools/download_models.py --models-dir /path/to/ComfyUI/models --dry-run

# Base INT8 + INT8 TE + 原 MOSS VAE
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets base_w8 te_int8 vae

# 原始 TE：四个分片、index/config 和 tokenizer 一起准备
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets te_original

# 对已经下载的文件只做本地校验
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets base_w8 te_int8 vae --verify-only
```

默认公开组合为 14,116,205,080 bytes（约 14.12 GB 十进制），不是所有可选权重的总大小。现有相同文件会验证后复用；不匹配的已有文件不会被覆盖。下载先写临时文件，验证大小和 hash 后发布到目标路径。缺少 revision/hash 的待发布自有权重不能通过此工具伪装成可下载文件。

Original HF TE 目录是 `models/text_encoders/MOSS-Image-Base/text_encoder/`，同级为 `tokenizer/`。完整映射见 `MODEL_FILES.md`。

## FP8 本地导出

有原始 BF16 DiT 时，可以离线生成 Kohaku stored row checkpoint：

```bash
python tools/export_fp8.py --source /path/to/moss_image_base_bf16.safetensors --output /path/to/ComfyUI/models/diffusion_models/moss_image_base_kohaku_fp8_row.safetensors --variant base --precision fp8_compute --recipe row --device cuda:0
```

SFT 使用其自身原始文件或 HF transformer 目录，并指定 `--variant sft`。weight-only 格式使用 `--precision fp8_weight_only`。导出不训练、不下载，不覆盖已有输出，也不把已经量化的文件当作原始 BF16 再量化。输出附带 recipe 和转换 receipt；发布前还应记录源与目标 hash。

## CUDA 内存分配器

优先用 ComfyUI 自带的 `--cuda-malloc`，通过 PyTorch 接入 CUDA 的 `cudaMallocAsync` 内存池。在任何 CUDA 初始化前选择后端：

```bash
python tools/launch_comfy_allocator.py --comfy-main /path/to/ComfyUI/main.py -- --listen 127.0.0.1
```

启动器先在一次性进程里验证后端和基本生命周期。环境不支持时，在新进程启用 native 和已有补丁；不会因为一次采样失败就自动重跑工作流。它保持两种 allocator 环境变量一致，避免外部的 `backend:native` 抵消 Comfy 参数。显式故障恢复可选择 native，详见 `ALLOCATOR_SUPPORT.md`。

4/6/8/12/16 GB profile 分别设置 3.5/5.5/7.5/11.5/15.5 GiB 的进程分配预算。native 与 async 的统计语义不同：async 内存池 `reserved` 可以高于申请限额，不能因此直接判超预算；实际物理显存和外部分配仍需观察。本候选修复了 Torch 2.10 async 初始未限额 getter 返回大于 1 的值、导致 finally 恢复失败的问题。

offload、VAE 预估、原始 payload 双缓冲、锁页 chunk 与卸载清理均保留。GPU 内存池与 CPU pinned pool 是两套分配机制，切换 CUDA 后端不会自动替代 host-cache 生命周期管理。

## 分享工作流

可以分享 `example_workflows/*.json`，或者分享带 Comfy 工作流 metadata 的输出 PNG。它们携带图和参数，**不携带模型、节点源码或 Python 环境**。

本包当前提供**七个UI模板及对应API图**。普通生成同时连接PreviewImage和SaveImage，
Controls默认为idle。SFT示例16步，Base示例50步；FP8模板明确要求本地导出或等待
真实HF发布。W5/W6直接索引现有公开权重，不重复上传。

**本轮七份精确JSON均已通过正常步数HTTP/API执行**，包括01精确公开SFT fused文件、
04显式Original TE目录、06/07新W5/W6图及版本化TE v2别名。API文件SHA保持与执行
回执一致；UI模板随后仅更新了不执行的Note说明文字，前后SHA链见
[Note文本变更记录](docs/validation/workflow-note-lineage.json)，节点ID、链接、设置及
模型metadata完全不变，原执行回执SHA保留。请求携带UI工作流供PNG metadata保存。
另已在实际Windows/Chrome **v4最终前端检查中通过06导入及真实按钮/dialog操作**：9节点、
20个原输入持久化widget及两个临时widget，状态settled；按钮callback刷新并打开报告，
显示4070 Ti SUPER和cudaMallocAsync，六项dialog检查全部通过，关闭后dialog移除，
序列化的20个输入值不变。无console errors、无`POST /prompt`。使用最终Note版06，
环境label/displayName均为“环境状态：依赖检查通过”，实际2×截图确认可见；全部旧v3
检查及新label检查通过。v3虽通过数据/dialog检查，但disabled文本value被前端隐藏，
本次仅`web/environment.js`改用官方支持的label，readonly/name/value和20输入不变，
06 UI SHA不变，无GPU/runtime改动，不重算62测试或7图。
详见[最终前端回执](docs/validation/frontend-final-20261025.json)，
[v3回执](docs/validation/frontend-dialog-20261025.json)保留先前按钮验证及显示问题。
这不是生成测试，
也不是其余六图逐一浏览器认证；七图GPU执行是独立证据。本轮新图没有新增
人工画质认证。以下旧PNG继续作为带原图的历史样例，不冒充
本轮新生成图片。完整数据见[本轮验证](docs/PUBLICATION_VALIDATION.md)。

| 模板 | 当前JSON | 样例及验证边界 |
|---|---|---|
| SFT W8，显式公开fused文件 | [工作流](example_workflows/01_sft_w8_legacy_1024.json) | 当前16步PASS；[旧auto历史PNG](docs/samples/01_sft_w8_legacy_1024.png) |
| Base W8 | [工作流](example_workflows/02_base_w8_standard_1024.json) | 当前50步PASS；[历史PNG](docs/samples/02_base_w8_standard_1024.png) |
| Base Kohaku FP8 row | [工作流](example_workflows/03_base_fp8_row_standard_1024.json) | 当前50步PASS；[历史PNG](docs/samples/03_base_fp8_row_standard_1024.png) |
| Base + Original BF16 TE目录 | [工作流](example_workflows/04_base_w8_original_bf16_te_1024.json) | 当前50步PASS；[旧auto目录历史PNG](docs/samples/04_base_w8_original_bf16_te_1024.png) |
| Base + W8A8 TE | [工作流](example_workflows/05_base_w8_real_w8a8_te_1024.json) | 当前50步PASS；[历史PNG](docs/samples/05_base_w8_real_w8a8_te_1024.png) |
| SFT W5低显存，4 GB进程profile | [工作流](example_workflows/06_sft_w5_lowvram_1024.json) | 当前16步PASS并在格式化后复跑；FP32 VAE |
| SFT W6低显存，6 GB进程profile | [工作流](example_workflows/07_sft_w6_lowvram_1024.json) | 当前16步PASS；FP16-fixed、W8A16 TE、FP32 VAE |

本轮Windows单元测试**62/62通过**；官方Comfy CLI **1.23.0**验证通过，E702/S102/
S307选定检查零发现。模型清单先核验14个已有公开文件38,776,396,306字节，再下载并
核验公开SFT W8的5,413,698,958字节，共15个公开文件44,190,095,264字节。执行使用
既有Windows Python环境，但节点/模型目录隔离、无`models.local.json`及私人注意力包；
这不等于新装OS/Python依赖验收。32文件格式化前后AST完全相同，01与06另有格式化后
运行记录。集成耗时只保留观察，不作为速度benchmark。

公开模型的 `properties.models` 包含下载 URL、目标目录及 hash，帮助支持该功能的客户端显示缺失模型。ComfyUI Desktop 与普通网页部署的下载、放置行为不完全相同；独立下载器提供确定的文件路径和验证流程。JSON 无法替别人保证驱动、Triton、可选二进制和模型全部自动安装。

当前七图的单文件metadata均匹配实际widget文件名。Original TE须用
`--assets te_original`完整准备分片、index/config及tokenizer；04的模型metadata
仅列DiT和VAE，不能依靠前端Download all安装整个TE目录。详见
[工作流分享说明](docs/SHARING_WORKFLOWS.md)。

INT8 TE v1与v2在上游使用同一basename，仅按文件名检查可能把旧v1误当成v2。
本包将精确v2文件下载/选择为
`text_encoders/moss-image-qwen35-int8-convrot-v2.safetensors`，metadata和widget均
使用这个本地别名。上游URL中的原始文件名、revision及SHA不变；普通浏览器下载
可能仍使用上游原名，需按此别名改名并用下载器`--verify-only`核验。该改名不修改
权重字节。此前精确v2的独立API PASS保持原范围；本轮七图另有各自精确JSON执行回执。

尚未注册的节点不会写假的 `cnr_id`。正式注册后，从已安装 Registry 版本导出最终模板，以携带真实节点包 ID 和版本。

## 发布状态

```bash
python tools/validate_release.py
python tools/validate_release.py --publication
```

本地检查与公开发布门槛分开。GitHub目标为
[Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized)，
Registry Publisher `xianbei`的公开状态已确认为**Active**。本地加密凭证读取/
解密检查已成功；**PAT在Registry服务上的认证仍未验证**，不能据此宣称已可发布。
尚无用于这项PAT认证检查的官方只读验证端点，不探测可能产生写入的发布接口，
文档也不包含凭证内容。三个自制FP8文件的目标HF repo仍待用户选择，不能将GitHub
URL当作HF模型仓库。此说明不声称本轮已经推送、上传或注册。

准备内容及本轮执行验收已完成。源码上传进度、仓库可见性和Registry实际发布结果见
[发布计划](docs/PUBLISHING_PLAN_ZH.md)及操作者后续的真实回执；不从本地验证通过推断
这些外部写入已完成。当前验证数据记录Registry发布未尝试。

20系仅保留历史2080 Ti **22 GB物理显存卡**的W5 auto端到端回执，4g/6g是软件
预算，不是物理4/6GB卡。当前版本未实卡复测；40系是当前SM89验收，50系待补，
Linux未测。详见[分代支持与历史证据](docs/ENVIRONMENT_SUPPORT.md)。

根协议为 `LICENSE` 中的 KohakuMoss License 1.0；必须保留 Kohaku 品牌及组件许可。原始模型、转换产物、tokenizer 和可选依赖的许可范围见 `NOTICE`、`THIRD_PARTY_NOTICES.md` 和 `LICENSE_PROVENANCE.json`。发布应使用本轮整理的干净源码历史，不公开私有旧 baseline。发布步骤见 `RELEASE_CHECKLIST.md`。

官方一手规则：[Registry发布](https://docs.comfy.org/registry/publishing)、
[依赖与兼容声明](https://docs.comfy.org/registry/specifications)、
[模板下载行为](https://docs.comfy.org/interface/features/template)、
[禁止运行期安装依赖](https://docs.comfy.org/registry/standards)。
本项目的实际待发布顺序见[发布计划](docs/PUBLISHING_PLAN_ZH.md)。

## 使用限制

- 当前 batch 为 1，不提供 LoRA 注入。
- 不截断长提示词；超过最大完整 token 长度会报错。
- 超过预算仍可能需要 Comfy 的 VAE 分块或卸载。未知 shape 不套用旧测量估计。
- 不承诺 FP8、W8A8、低精度注意力比其他模式更快，也不承诺画质无变化。
- 旧 R4 报告中的 native 分配器结果保留原口径，不作为 async 和新格式的实测结果。
