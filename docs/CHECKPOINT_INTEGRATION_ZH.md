# DiT checkpoint 集成接口

`runtime/checkpoints.py` 提供加载与离线导出接口；节点和 Session 已接入这些
接口，Session 负责设备放置、资源调度和 GPU 准入。
Base与SFT架构相同，无法通过shape可靠辨认模型身份。调用方必须提供模型来源与
`variant`，不能仅凭文件名证明其确为Base。实际非量化张量全部来自传入checkpoint，
不再借用SFT retained sidecar。加载沿用vendor compute dtype和KEEP_FP32规则，
“来自同一个文件”不代表任何dtype转换都逐位保持原文件。

## Base/SFT INT8 ConvRot

```python
from .checkpoints import inspect_checkpoint, load_checkpoint

header = inspect_checkpoint(weights)  # 标准库；不加载大张量、不初始化CUDA
bb, receipt = load_checkpoint(
    weights, variant="base", dtype=torch.bfloat16, device="cpu",
    int8_backend="w8a8", keep_fp32=(),
)
```

`load_checkpoint`先调用打包vendor的`load_backbone(..., strict=True)`，其
`ModelKeyView`处理Comfy-Org fused qkv和Sumomo separate q/k/v。每个codes与
scale按输出行一起拆分；Comfy quant记录共享。随后将符合固定ConvRot256协议的
joint INT8层替换为原有`FrozenInt8Base`。GPU forward复用现有A8动态量化及原生
INT32 GEMM；CPU加载成功不代表CPU可以运行这个W8A8 backend。

W8A8入口要求显式`comfy_quant`、ConvRot group256、无bias、GEMM维度16对齐。
不会从文件名猜rotation，也不会暗中退到weight-only。普通weight-only参考加载可
显式选择`int8_backend="weight_only"`，但那不属于W8A8性能测试。

Session应继续负责后续的op binding、设备放置、FP16 online fix、attention/backend
安装及payload capture。不要同时执行旧的SFT retained加载分支。量化base是
`FrozenInt8Base`，可继续交给既有payload owner；保留load receipt。整模型CUDA
加载需要调用方事先做资源准入，默认`device="cpu"`。
`keep_fp32=None`保留upstream默认值；显式`keep_fp32=()`恢复旧R4的非量化参数
全部compute dtype、派生rope保持FP32的contract。该选择记录进receipt。

## Kohaku FP8

支持单文件safetensors metadata `kohakumoss_fp8`，内容是JSON字符串
`{"recipe":"row","layers":[...]}`。每层为E4M3 `.weight`和`.weight_scale`。
header检查要求量化权重与layers集合完全一致，不接受缺失scale或未知recipe。

| 模式 | weight scale | activation/compute语义 |
|---|---|---|
| `fp8_weight_only` | FP32逐输出行 | activation保留16位；weight-only GEMM |
| `fp8_compute` row | FP32 `[N]` | FP8计算路径按调用动态量化activation；vendor也可能选择不量化activation的w16路径 |
| `fp8_compute` tensor | FP32 `[1]` | tensor scale |
| `fp8_compute` block | FP32 `[ceil(N/128),K/128]` | block scale |
| `mxfp8` | UINT8 e8m0 `[N,K/32]` | MX block scale；不能按普通FP32行scale解读 |

MX vendor swizzle由打包的`ComputeWeight.from_stored`重建。推理不分配训练用
转置权重副本。放置完成后调用：

```python
from .fp8_support import bind_fp8_inference
binding = bind_fp8_inference(bb, dtype=compute_dtype, device=device)
```

默认要求CUDA SM89+。CPU/其他设备数值参考必须显式`backend="eager"`。vendor
本身可能按shape选择16位路径，因此receipt不声称每一层都执行FP8 tensor-core，
也不把设备能力检查当作速度或图像质量认证。
`fp8_compute`是存储/执行接口的语义族，不表示每次实际dispatch都量化activation。
上游w16路径反量化weight后执行16位matmul，activation保持16位。

`PayloadCache`现在可以接管42×6同构joint FP8层的原始q字节，复用已有双slot
调度。scale与MX aux保持普通model buffers，只在模型放置时上传，不随slot搬运。
placement完成后对joint层作forward-only rebind；额外text adapter FP8仍由Session
负责人单独place/rebind。training q_t/scale_t/aux_t一律拒绝。

默认joint scale+aux总预算256MiB，实际字节数写入owner receipt，与slot/prefix
payload字节分开。row/weight-only通常较小；MX包含自然scale与swizzled aux，可能
超过预算并明确拒绝，调用方需切换完整模型放置，或显式提高预算同时将它纳入显存
准入。可传`cache.capture(..., fp8_side_buffer_budget_bytes=...)`，不会自动回退。
旧INT8/W5/W6路径和pinned chunk生命周期保持原样。CPU fake调度测试不构成GPU
认证。本次已在同一张16GB SM89卡上完成Base FP8 row的4g profile两步出图，
以及W5、Base W8的4g profile短→长→短切换；这些是进程预算测试，不能写成
物理4GB显卡认证。实际峰值与逐组合范围见 `../RELEASE_VALIDATION.md`。

现有W5/W6 INT32特定投影patch不适用于FP8 base，必须跳过。各recipe可能有不同
的activation/workspace开销，side-buffer预算不是整模型显存预算。

## 从本地BF16离线导出

```powershell
python tools/export_fp8.py --source /path/to/ComfyUI/models/diffusion_models/moss_image_base_bf16.safetensors --output /path/to/ComfyUI/models/diffusion_models/moss_image_base_kohaku_fp8_row.safetensors --variant base --precision fp8_compute --recipe row --device cpu
```

也可传本地HF Base/SFT根目录或transformer目录。`--device cuda:0`仅决定逐层
量化工作设备，仍需调用方保障GPU准入。CPU是默认值，转换大模型需要足够主机RAM。
`--precision mxfp8`生成Kohaku MX格式；`fp8_weight_only`生成明确Comfy per-layer
format记录，不生成`kohakumoss_fp8`。普通Kohaku的`fp8`运行模式与其stored FP8
compute文件不可混用。

当前导出器的非量化tensor dtype需按格式区分：row-compute通过
`quantized_state(..., torch.bfloat16)`写出BF16浮点tensor，`rope.bands`保持FP32；
weight-only分支保留加载后的live dtype，因此除了BF16外，还保留默认
`KEEP_FP32`敏感参数及rope的FP32。这是文件存储口径，不能笼统写成所有FP8产物
的非量化参数全部BF16。加载时可显式传`keep_fp32=()`统一到运行时compute dtype；
这不会追溯修改已导出文件、其字节数或SHA256。

只接受dense源，拒绝已有量化codes和folded fp16-fix模型。量化使用vendor的
checkpoint source机制，不先将BF16源强制转FP16。不会训练、联网下载或覆盖已有
输出；输出safetensors及同名JSON receipt。导出本身不是GPU推理验证。

## 验证

```text
python -m unittest discover -s tests -p test_checkpoints.py -v
```

标准库测试检查各recipe、异常scale、遗漏layer、两种qkv布局与rotation metadata。
有torch/safetensors时另跑tiny backbone真实加载测试，核对原始codes、scale切片
以及同源dense video_embedder。没有torch时该测试明确skip，不伪装为数值PASS。
最终Windows环境已完成41项组件测试、17个直接执行案例和5个正常步数ComfyUI
HTTP工作流，包括Base W8、Base FP8 row与原始/真实A8 TE。Base/SFT FP8 row
及Base weight-only均有实际加载出图记录。上述记录不覆盖所有FP8 recipe、
backend或shape，也不等同于完整模型数值/画质等价认证。

## 当前独立解包实现与历史边界

当前 Session 使用 `runtime/packed_decode.py`。旧
`runtime/vendor/unpack_sm75.py` 及其不再适用的 GPL 文本不在分发候选中。
新实现按 little-endian B-bit two's-complement 数学布局独立编写：
`UINT8[ceil(K*B/8),N] -> INT8[N,K]`。操作者报告未阅读被移除实现，
并已在 SM89 进行 10 个 W5/W6 black-box case、169,939,328 个整数 code 的
逐值相等检查。详细 scope 与实际源码 hash 见 `../LICENSE_PROVENANCE.json`。
这不是正式法律 clean-room 认证，也不追溯改变历史代码许可。

旧私有 baseline 及失败尝试保留在私有证据中；公开发布应使用不包含该旧代码的
clean release history。根 LICENSE 保留 KohakuMoss 全文，其他组件保留各自
许可与 NOTICE。当前组件来源见 `../runtime/vendor/SOURCES.json`。

本文件是 checkpoint API 工程说明，不是完整当前 GPU 测试回执。新增 FP8
Base/SFT row 与 Base weight-only 已由操作者完成本地导出；这不自动证明每个
backend、shape、显存档或普通 UI 都通过。最终逐组合结果以实际
`../RELEASE_VALIDATION.md` 及对应回执为准。
