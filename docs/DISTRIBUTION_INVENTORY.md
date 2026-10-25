# 分发清单与来源边界

主展示：**MOSS LowVRAM Optimized**。副标题：**使用KohakuMoss推理组件**。
正式分发包名：`kohaku-moss-lowvram-optimized`。现有 GitHub 地址暂保留为
<https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized>；内部节点 ID 不变。

正式包名包含 Kohaku，是当前按 KohakuMoss License Section 10 采用的命名路径。
副标题说明组件来源，不被当成该条款的例外授权或法律认证。根 LICENSE 原文、
组件 LICENSE/NOTICE 与来源 hash 均保留。

## 各类文件的分发决定

| 文件类别 | 当前决定 | 来源与验证要求 |
|---|---|---|
| 本项目节点、独立 adapter、生命周期与下载工具 | GitHub / Registry 源码包 | 按根 LICENSE 及组件 scope 保留来源和修改说明；不改变旧 node ID |
| Kohaku / OpenMOSS 复用源码、tokenizer/config | 随源码包保留必要文件 | `runtime/vendor/SOURCES.json`、`LICENSE_PROVENANCE.json` 及各组件原 LICENSE/NOTICE；不能把 gzip tokenizer 当作无来源新输出 |
| 已在原仓库提供的 Base/SFT/TE/VAE 权重 | 直接索引原仓库，不复制上传 | 固定实际 revision、文件名、大小、SHA256；保留模型自己的许可，见 `model_manifest.json` |
| 已公开 W5/W6 | 索引现有 Andylinux 仓库，不重传 | 本节下方列明已核对来源与 hash；不要再将“尚未选择新 HF 仓库”套到这两份文件 |
| 自产 Base FP8 row、SFT FP8 row、Base FP8 weight-only | 三份本地导出已完成；待选择 HF 仓库 | `huggingface/conversion_manifest.json`、`huggingface/SHA256SUMS`；没有公开 URL/revision 前不宣称已发布 |
| 原发行方已有可选 Flash / Sage wheel | 直接索引原始发行页面，不另做镜像 | 保留精确 artifact/hash、平台/ABI 说明与适用 MIT/BSD/Apache 文本；不是任意 wheel 兼容保证 |
| Torch / CUDA / Triton 工具链 | 使用用户现有兼容环境，不随包捆绑 | 不自动覆盖 Torch/CUDA，不复制整套本地环境 |
| 固定 prompt 预编码 conditioning `.pt` | 不作为公开默认依赖分发 | 普通工作流实时运行 TE；不要求用户下载本地 benchmark prompt cache |
| 校准／性能测试 cache、latent、编译 cache | 不作为默认模型依赖 | 私有证据按用途保存；不是缺失后节点应自动下载的运行资产 |
| `ssiu/flash-attention-turing` bundle / 自编 `.pyd` | 不捆绑、不镜像、不上传 | 固定版本的项目授权未确认；不能仅以 CUTLASS/NVIDIA 文本视为整包许可 |

## 已公开 W5/W6

仓库：<https://huggingface.co/Andylinux/MOSS-Image-SFT-W5A8-W6A8>。
已核对 revision：`bfb77cb73ae7acb4f788d50d4043c2d71989b0a7`。
操作者已确认这些公开文件与本地文件 SHA256 一致，使用现有分发，不重复上传。
下表中的名字为本地已验证文件名，不在未核对时推断 HF 内部文件路径。

| 本地文件 | SHA256 |
|---|---|
| `q5_ridge_gptq.safetensors` | `3f8a775e179a63a7511d469f08430b89f5807ebd181f37f4bbaea7d039096f82` |
| `q6_direct.safetensors` | `dc5af59f22eb89320e457536ae90b170c6548c74a9fca584684139cf88018f87` |

这里的 revision 是上述公开 W5/W6 仓库的版本，不被挪用成原始 BF16 模型的下载
revision。模型的更早上游与量化说明以其真实 README / LOWBIT_MANIFEST 及原许可
为准；不添加未经证实的原始 source revision。

## 三份新 FP8 导出

| 导出文件 | SHA256 |
|---|---|
| `moss_image_base_kohaku_fp8_row.safetensors` | `4f18284539c84c804919e7c343ee33b1550dfcd1c4cd5dc5dd7f4569905d0f14` |
| `moss_image_sft_kohaku_fp8_row.safetensors` | `06c58d186852abd0fd073fba6486a82543a6006a8c17a3eac7d62828b9c03862` |
| `moss_image_base_kohaku_fp8_weight_only.safetensors` | `03a31fb8a7c2e196b3b0c3b02849442d40503ead10cdf9f9f4bce543525c1c54` |

这些是格式转换／量化导出，不是训练。导出完成不等于任意显存、精度、attention
或普通 UI 组合全部通过。原始来源按 conversion manifest 的实际 file hash 记录；
其中未确认的 source revision 保持未确认，不用当前 main 或 TE 的 revision 补齐。

## 可选 binary 的许可边界

可选 Sage 原发行页：
<https://github.com/woct0rdho/SageAttention/releases/tag/v2.2.0-windows.post6>。
其他 wheel 按 `portable_attention_dependencies.md` 和实际 artifact 记录索引原来源。
保留当前 `licenses/` 中适用的 MIT、BSD、Apache 及 NOTICE 文本，不把某个依赖的
开放许可推广为所有捆绑 binary 的许可。若未来自编并分发，须另列实际源码 commit、
修改、编译环境、ABI、目标 hash 和全部实际纳入组件的许可；本清单没有授权这种
尚未审查的新分发。

本机 Turing bundle 的来源 commit 为
`9ef98fcb506bb1e2fe3cece50935e2935bf6b124`。其 manifest 记录未发现该固定
Turing 仓库的 LICENSE／许可声明；CUTLASS/NVIDIA 文本不自动许可其新增实现和
组合 `.pyd`。公开包排除它，默认 auto 不依赖它。本项目独立 adapter 可以保留，
但不能随 adapter 携带上游未解决授权的代码／binary。本机已有显式可选使用与
公开再分发分别记录，不互相证明授权。
