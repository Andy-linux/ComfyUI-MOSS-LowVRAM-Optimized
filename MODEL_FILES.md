# 模型与离线导出

## 在界面准备

导入 [00工作流](example_workflows/00_moss_lowvram.json)，在配置节点选择系列、DiT、TE与VAE，点击
**Prepare / check selected configuration**，确认计划后准备资源。
默认选择SFT W5、Sumomo INT8 TE v2 / W8A16和原MOSS VAE。

| ComfyUI目录 | 用途 |
| --- | --- |
| `models/diffusion_models/` | DiT权重 |
| `models/text_encoders/` | 文本编码器 |
| `models/vae/` | VAE |

支持 `extra_model_paths.yaml` 注册目录。匹配文件经过完整SHA256校验后原位复用；Original TE按计划准备完整目录。
INT8 TE v2推荐别名为 `moss-image-qwen35-int8-convrot-v2.safetensors`，以区分旧v1。

## 可选资产

文件名、固定revision、精确大小、SHA256、目标路径和许可来源统一记录于 [model_manifest.json](model_manifest.json)。

| Asset ID | 模型 | 文件总字节数 |
| --- | --- | ---: |
| `base_w8` | Base INT8 ConvRot | 5,413,723,800 |
| `base_w8_comfy` | Base INT8 fused QKV | 5,413,698,958 |
| `sft_w8` | SFT INT8 ConvRot | 5,413,698,958 |
| `te_int8` | Sumomo INT8 TE v2（默认） | 8,452,959,908 |
| `te_int8_comfy` | INT8 TE Comfy封装 | 9,748,121,532 |
| `te_w4` | W4 GPTQ TE | 7,343,591,088 |
| `vae` | 原MOSS VAE（默认） | 249,521,372 |
| `vae_flux2_small` | Flux2 small VAE | 249,521,340 |
| `te_original` | Original TE完整目录 | 16,805,544,650 |
| `sft_w5` | SFT W5（默认） | 3,630,052,176 |
| `sft_w6` | SFT W6 | 4,224,594,400 |
| `base_fp8` | Base Kohaku FP8 row | 5,174,879,864 |
| `sft_fp8` | SFT Kohaku FP8 row | 5,174,879,840 |

Base/SFT各一份FP8 row文件，支持FP8或16-bit激活计算。
两份文件位于[Andylinux/MOSS-Image-Kohaku-FP8](https://huggingface.co/Andylinux/MOSS-Image-Kohaku-FP8/tree/64cc4abd4ec6548b935aa75a1dd22e9d43236aab)。
W5/W6来自[Andylinux/MOSS-Image-SFT-W5A8-W6A8](https://huggingface.co/Andylinux/MOSS-Image-SFT-W5A8-W6A8/tree/bfb77cb73ae7acb4f788d50d4043c2d71989b0a7)。
旧独立Base weight-only导出仅用于兼容本地已有文件。

## 可选命令行下载

使用ComfyUI同一Python，从插件目录执行并明确指定所需asset：

```sh
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets sft_w5 te_int8 vae --dry-run
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets sft_w5 te_int8 vae
```

`--verify-only`校验本地文件。下载先写临时文件，大小与SHA256匹配后写入目标；目标文件系统需支持硬链接，如NTFS或ext4。

## 离线FP8导出

导出接受本地原始dense BF16单文件或HF transformer目录，输出到新文件：

```sh
python tools/export_fp8.py --source /path/to/moss_image_base_bf16.safetensors --output /path/to/base_fp8_row.safetensors --variant base --precision fp8_compute --recipe row --device cuda:0
```

SFT使用其自身源文件或目录并指定 `--variant sft`。导出执行量化与格式转换，输出附带recipe和来源记录。
参数详情见 `python tools/export_fp8.py --help`。权重及转换来源见[conversion_manifest.json](docs/huggingface/conversion_manifest.json)，许可见[组件说明](THIRD_PARTY_NOTICES.md)。
