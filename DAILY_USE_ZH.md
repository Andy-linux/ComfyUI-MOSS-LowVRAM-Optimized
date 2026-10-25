# Kohaku MOSS Optimized 日常使用入口

当前便携发布候选以 [README.md](README.md) 为主使用说明，模型路径、
默认公开 Base INT8 组合、TE 精度、allocator 和依赖边界均在那里定义。

## 新安装

1. 在现有 ComfyUI 环境安装节点与声明的 Python 依赖；Torch/CUDA 不由本包替换。
2. 按 [MODEL_FILES.md](MODEL_FILES.md) dry-run 并显式下载所选公开模型，或选择本地权重。
3. 打开节点提供的 `example_workflows/` 示例。Base 与 SFT、INT8 与 FP8 是不同选择。
4. 填写完整提示词，按模板的实际采样参数运行。Controls 保持 idle；预热与卸载是显式动作。
5. SaveImage 保存最终图片，PreviewImage 仅提供预览。比较结果时保持真实权重、
   seed/噪声来源、尺寸、精度、attention、steps 与缓存状态一致。

W5/W6 与用户导出的 FP8 仍需本地文件或待提供的真实 HF 发布信息。
FP8 本地导出已完成不等于所有运行组合均已验收；参见实际 release validation。

## 历史 R4 的适用范围

R4 原性能矩阵在 RTX 4070 Ti SUPER、PCIe 4.0 ×4 上使用 SFT W5/W6/W8，
同卡五档 allocator 预算，并采用其对应的原始数值、fresh-TE 和 I/O 口径。
它不是当前所有 Base、FP8、A8 TE、async allocator 或其他硬件组合的性能证明。
历史本机启动器、工作流目录与 benchmark 绝对路径不属于公开安装要求。

当前硬件/组合、成功与失败范围须由 `RELEASE_VALIDATION.md` 的实际证据说明；
不能把核级检查、CPU 加载或同形状另一模型当作完整普通 UI 通过。
