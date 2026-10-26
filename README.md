# MOSS LowVRAM Optimized

使用 **KohakuMoss** 推理组件

本项目面向 **ComfyUI**，提供 MOSS 多精度推理、显存预算管理、权重搬运调度及工作流集成。

当前实现复用了 **KohakuMoss** 的模型与文本编码等组件，并使用 **OpenMOSS**、**ComfyUI**、**ComfyKitchen** 等项目的相关实现。原始许可证、版权和修改记录继续保留。

正式分发包名为 `kohaku-moss-lowvram-optimized`；`Kohaku` 前缀保留上游许可第10节要求。内部节点 ID 保持不变，节点显示采用 `MOSS · …`，分类为 `MOSS/Optimized`。

本项目执行量化权重加载、文本编码与图像推理；离线 FP8 导出是格式转换／量化，不进行训练。

代码包不包含模型权重、Torch/CUDA 或可选注意力二进制。Manager 的安装阶段仅为明确匹配的宿主准备固定 Triton、便携 Python 开发文件及独立可选注意力资产；导入节点、环境检查和推理不会安装依赖。当前发布候选的实际验证环境为 Windows、RTX 4070 Ti SUPER（SM89，16 GB）、Python 3.12、Torch 2.10.0+cu130、ComfyUI 0.41.0。具体通过的组合和限制见 `RELEASE_VALIDATION.md`；显存 profile 是同一张卡上的进程预算，不代表已经在每种物理小显存卡上验收。

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

## 默认使用：一个工作流完成配置与模型准备

1. 在已启用官方 ComfyUI Manager 的 ComfyUI 中导入
   [00 · MOSS 统一入口](example_workflows/00_moss_lowvram.json)，或从模板浏览器打开它。
2. 如果提示缺少节点，通过 Manager 安装真实已收录/发布的本节点包。
   仅 GitHub 公开并不保证默认新 Manager 能发现它；发布结果以实际 Registry/Manager 收录为准。
3. 点击 Manager 的 **Restart / 重启**，等待启动依赖准备完成。
   Windows Manager 4.2.2 会把安装步骤安排到下次启动，UI 队列成功不代表依赖已经安装。
4. 在 **MOSS · 配置与准备** 配置节点选择 Base/SFT、DiT 存储和计算、TE 格式和计算、
   显存 profile、注意力与 VAE。点击 **Prepare / check selected configuration**。
5. 在弹窗查看环境、所选模型的路径/大小/校验状态与下载来源，确认后点击下载。
   只准备当前所选组合；已有文件按状态复用，冲突文件不会被覆盖。Original TE 会按完整资产准备分片、
   配置和 tokenizer。切换配置后先刷新计划，重新确认所需文件。
6. 检查通过后生成。Base 建议 50 步、SFT 建议 16 步；切换系列不会擅自修改提示词、seed 或采样设置。
   需要同步建议值时，显式点击 **Apply recommended sampling parameters**。

以上默认流程无需终端命令。弹窗不能在插件或 Manager 尚未安装时自举安装它们。
尚未上架的发布候选可由维护者隔离验证，但这不能算新用户 Manager 安装闭环已验收。

### 依赖准备范围

常规依赖包含 `diffusers>=0.41.0,<0.42`：上游编码器会直接导入其 `AutoencoderKLFlux2`，因此即使从TE加载开始也需要此依赖；Manager会与其余requirements一起准备，环境检查会提前报告缺失或版本不匹配。

当前自动准备的 Windows 精确组合为 **x64 Python 3.12、Torch 2.10.0+cu130、CUDA build 13.0、SM89**，
缺少 Triton 时由 Manager 安装 `triton-windows==3.6.0.post26`。SM75 仅提供历史 Python3.12 /
Torch2.10+cu130 / Triton3.2.0.post21 配对的准备策略，明确不是本次发布的实卡复验。
Linux 仅依据现有 Torch 包元数据中适用的精确 Triton 依赖准备，不声明工作流已验收。
其他宿主组合会显示诊断；本插件不会替换 Python、Torch 或 CUDA，也不安装“最新 Triton”。
Python ≥3.12 的声明是最低安装条件，不代表所有后续 Python/Torch/GPU 组合已测试或可自动准备。

仍缺依赖时，先重启执行待处理任务，再查看当前 Manager 实际可见的维护入口。
4.2.2 旧界面的名称为 **Try fix / Reinstall**；默认新界面由独立 frontend 提供，
不保证存在名为 Fix Dependencies 的按钮。安全策略拒绝维护操作时，请使用
[官方 ComfyUI Desktop](https://www.comfy.org/download) / 宿主图形维护流程，不降低安全级别。
完整步骤、固定版本、便携 Python 开发文件校验和故障边界见
[安装指南](docs/INSTALLATION_GUIDE.md)及[运行环境支持表](docs/ENVIRONMENT_SUPPORT.md)。
自动准备状态与运行时环境检查分开：自动安装策略不覆盖某个既有环境，不等于该环境不能运行。

**升级已有安装时先检查重复包。** 原六个内部 node class IDs 保持兼容，新统一配置节点为新增入口；
不要同时启用旧 `moss-optimized`、`ComfyUI-KohakuMOSS` 与新的 Registry 包。
更新或迁移唯一一份安装，既有目录名可为启动器兼容保留。迁移前私下备份
`models.local.json`，不要把私人映射提交发布。

显式 A8 TE 需要 CUDA SM80+；本候选的 FP8 计算要求 SM89+。低预算双缓冲及 Sage 等路径的
实际验收以 SM89 为准。可选 Flash/Sage 不在导入或运行节点时安装。
`auto` 使用 PyTorch SDPA；`Flash` 只允许实际 Flash 内核。
精确SM89组合会在Manager安装阶段尝试固定公开Sage wheel，失败保持auto。用户已授权将自编SM75 Turing包作为独立GitHub Release分发；上游固定commit没有已确认仓库许可的事实继续披露，实际组件LICENSE/NOTICE与来源构建记录一并保留，不伪造开放许可。Turing已作为[独立公开Release](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/tag/turing-sm75-cp312-cu130-v1)发布，匿名完整下载确认43,707,834 bytes及固定SHA256，与远端digest一致；独立清单已启用public，仅精确SM75宿主在Manager安装阶段准备。
见 [注意力依赖说明](docs/portable_attention_dependencies.md)。

## 模型下载与路径

标准目录：

```text
ComfyUI/models/diffusion_models/
ComfyUI/models/text_encoders/
ComfyUI/models/vae/
```

节点通过 ComfyUI 的 `folder_paths` 查找文件，支持 `extra_model_paths.yaml`。代码没有固定 F 盘路径。已有模型可在节点的 checkpoint 下拉框中选择；也可在不分发的 `models.local.json` 中为 asset ID 配置本机路径。文件模式、Base/SFT 来源要与所选模式一致，两者相同的 shape 不能用来判定权重身份。

`model_manifest.json` 为公开文件记录完整 HF revision、精确字节数、SHA256、目标位置、格式和来源许可。
默认请使用统一配置节点的准备弹窗；选择 SFT W5 / Sumomo INT8 TE v2 / 原 MOSS VAE 时只准备这组资产，
不下载所有精度。独立维护下载器的旧默认组仍为 Base INT8 + INT8 TE + 原 VAE，合计
14,116,205,080 bytes（约14.12 GB十进制），与新00工作流的默认配置应分别理解。

下载先写临时文件，验证大小与 SHA256 后发布到目标路径。Original HF TE 的目标目录为
`models/text_encoders/MOSS-Image-Base/text_encoder/`，同级为 `tokenizer/`。
完整映射、维护者 CLI 选项及资产状态见 [MODEL_FILES.md](MODEL_FILES.md)。

## FP8 本地导出

有原始 BF16 DiT 时，可以离线生成 Kohaku stored row checkpoint：

```bash
python tools/export_fp8.py --source /path/to/moss_image_base_bf16.safetensors --output /path/to/ComfyUI/models/diffusion_models/moss_image_base_kohaku_fp8_row.safetensors --variant base --precision fp8_compute --recipe row --device cuda:0
```

SFT 使用其自身原始文件或 HF transformer 目录，并指定 `--variant sft`。公开分发范围为 Base/SFT 两份 row checkpoint；统一节点对同一 row 文件选择 FP8 或 16-bit 计算，不为计算开关重复下载权重。旧的独立 weight-only 导出仅为 legacy local 兼容，不作为第三份公开模型。导出不训练、不下载，不覆盖已有输出，也不把已经量化的文件当作原始 BF16 再量化。输出附带 recipe 和转换 receipt；发布前还应记录源与目标 hash。

## CUDA 内存分配器

优先用 ComfyUI 自带的 `--cuda-malloc`，通过 PyTorch 接入 CUDA 的 `cudaMallocAsync` 内存池。在任何 CUDA 初始化前选择后端：

```bash
python tools/launch_comfy_allocator.py --comfy-main /path/to/ComfyUI/main.py -- --listen 127.0.0.1
```

启动器先在一次性进程里验证后端和基本生命周期。环境不支持时，在新进程启用 native 和已有补丁；不会因为一次采样失败就自动重跑工作流。它保持两种 allocator 环境变量一致，避免外部的 `backend:native` 抵消 Comfy 参数。显式故障恢复可选择 native，详见 `ALLOCATOR_SUPPORT.md`。

4/6/8/12/16 GB profile 分别设置 3.5/5.5/7.5/11.5/15.5 GiB 的进程分配预算。native 与 async 的统计语义不同：async 内存池 `reserved` 可以高于申请限额，不能因此直接判超预算；实际物理显存和外部分配仍需观察。本候选修复了 Torch 2.10 async 初始未限额 getter 返回大于 1 的值、导致 finally 恢复失败的问题。

offload、VAE 预估、原始 payload 双缓冲、锁页 chunk 与卸载清理均保留。GPU 内存池与 CPU pinned pool 是两套分配机制，切换 CUDA 后端不会自动替代 host-cache 生命周期管理。

## 分享唯一工作流

当前唯一用户模板为 [00 · MOSS统一配置与准备](example_workflows/00_moss_lowvram.json)。
一个配置节点涵盖Base/SFT、DiT存储与计算、TE格式与计算、显存预算、注意力及VAE，
无需为每种精度另选JSON。默认SFT W5、Sumomo v2 INT8 TE / W8A16、auto注意力、FP32 VAE，
1024²、16步；Base建议50步，建议采样参数仅经显式按钮应用。

分享该JSON或带Comfy工作流metadata的输出PNG会携带图与参数，**不会携带模型、节点源码或Python环境**。
安装节点后使用GUI准备当前所选资产；它会查找现有注册模型路径，后台核验完整SHA256，
匹配文件可原位复用，缺失才下载。改名或上游原名文件不凭名字自动视为正确版本；
错误目标文件需用户明确确认保留旧字节后再下载。Original TE按完整目录准备，不依赖普通Download all
重建其分片、config/index与tokenizer。详见[分享与文件准备](docs/SHARING_WORKFLOWS.md)。

INT8 TE v2的推荐目标别名为 `text_encoders/moss-image-qwen35-int8-convrot-v2.safetensors`，
避免与同上游basename的旧v1混淆。GUI也可核验并复用其他注册名称下的同一份精确v2字节，
不要求仅为改名再下载8.45GB文件。Base/SFT FP8各一份row文件共享FP8/16-bit计算，
只切换计算不重复下载权重；两份公开权重已固定于R1 `d8a0b2f46a79770fe711bf5322cf6cb3219c8d41`，精确大小/hash见[模型文件](MODEL_FILES.md)。

旧01–07图已移到 `tests/historical_workflows/`，仅为GitHub历史测试fixture，
由 `.comfyignore` 排除Registry包，不再作为用户模板或当前分享方案。
当前00的 `cnr_id` / `ver` 对应已提交的0.1.0 metadata，不表示版本已Active或Manager安装已验证；实际安装后仍需核对并重新导出。

## 验证范围

新增共享FP8路径已完成一次明确范围的Windows GPU验证：RTX4070TiSUPER / SM89、
cudaMallocAsync，**Base/SFT × FP8/16-bit × 两次，共8个请求**，每次512²、2步采样并解码。
8项CPU测试通过且无skip；263层的q/scale指针、哈希及字节保持不变。
这验证同一row文件切换计算模式和相关加载契约，**不是完整步数画质认证或速度benchmark**。

Windows候选服务上的通用00已通过 **19项GUI检查**：准备作业全部ready，原位复用三份预存模型
（子目录中改名的W5、子目录TE v2别名、注册extra-root VAE），**模型网络下载为0**。
随后默认SFT W5 / TE W8A16、1024²、16步、seed314159265的真实prompt成功，输出
`MOSS/00_moss_lowvram_00001_.png`。日志执行区间25.76秒仅为这次观察，不是通用速度承诺。

此前独立Chrome context run03通过16项联动、Original TE九文件、metadata和24值序列化检查；
该run03没有download/prompt。此前delta04的Windows137/137单元通过、0skip，GUI相关32项通过；
冻结Turing helper又完成独立8项focused测试及真实17成员ZIP纯校验，均通过，不等于重新运行全套137项，也不是SM75 GPU采样验证。
干净Python基线上的requirements/install.py及固定Sage隔离准备已通过；**候选服务复用已有模型的
生成成功与真正空Manager基线安装是独立证据**。公开Registry发现→安装→重启闭环仍待完成。
完整参数、作业/prompt标识及边界见[统一00验证](docs/GUIDED_SETUP_VALIDATION.md)。

历史01–07图的正常步数HTTP执行、62项测试及旧前端检查继续按各自记录范围保留，
不移用为新00或新增计算路径的验证。[历史发布候选验证](docs/PUBLICATION_VALIDATION.md)和
`docs/validation/` 原始回执、数值与SHA链不改写。旧PNG不作为本轮新图或当前推荐分享模板。

## 发布状态

```bash
python tools/validate_release.py
python tools/validate_release.py --publication
```

本地检查与公开发布门槛分开。GitHub源码仓库已公开：
[Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized)，
本次Registry 0.1.0包对应实现提交 `c71ead06bad6fb77002c81b5f6b442bb905cf271`。Turing独立Release也已公开并完成匿名文件大小/hash核验；
Registry 0.1.0已提交并上传完成；03:24:01 UTC观察版本状态为Pending，真正空基线Manager安装仍待完成。
Registry Publisher `xianbei`的公开状态已确认为**Active**。
既有PAT已通过本次实际Registry发布请求的服务端认证；secure bridge仅提交一次，无重试。
Pending表示平台仍待处理，不是要求用户重新登录、重复发布或触发扫描；尚不能宣称Active或Manager已安装。
文档不包含凭证内容。两份Base/SFT FP8 row已公开并固定于[R1 `d8a0b2f46a79770fe711bf5322cf6cb3219c8d41`](https://huggingface.co/Andylinux/MOSS-Image-Kohaku-FP8/tree/d8a0b2f46a79770fe711bf5322cf6cb3219c8d41)，合计10,349,759,704 bytes。已核对两文件精确大小、远端LFS SHA256与本地完整文件SHA256，并读回小型配套文件；没有重新下载约10GB远端权重做全流SHA256核验。旧weight-only不纳入本次公开分发。
GitHub、Turing与HF资产公开不代表Registry注册或实际Manager安装完成。

新增00的候选服务复用/生成已通过；此前01–07图证据保留原范围。真正新用户Manager安装闭环尚待完成。
已在Windows独立基线（Python3.12.13、Torch2.10+cu130、Manager4.2.2，初始缺少Triton与accelerate）
完成requirements与install.py直接执行，两者退出码均为0；这证明安装钩子可在该基线运行，
**不等于经过Manager UI发现、安装和重启的端到端验收**。源码上传进度、仓库可见性和Registry实际发布结果见
[发布计划](docs/PUBLISHING_PLAN_ZH.md)及操作者后续的真实回执；不从本地验证通过推断
这些外部写入已完成。Registry 0.1.0已提交并上传完成，平台Active状态与真实Manager全链条仍待验证。

20系仅保留历史2080 Ti **22 GB物理显存卡**的W5 auto端到端回执，4g/6g是软件
预算，不是物理4/6GB卡。当前版本未实卡复测；40系是当前SM89验收，50系待补，30系未承诺整流程支持，
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
