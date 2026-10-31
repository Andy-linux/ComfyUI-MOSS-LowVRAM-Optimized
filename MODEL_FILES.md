# 模型与离线导出

## 本地使用与可选下载是两条独立路径

模型文件不随节点包安装，不要求下载任何不使用的模型系列、精度或编码器。
在 [00工作流](example_workflows/00_moss_lowvram.json)中选择系列、存储格式和实际计算方式，然后从下拉框选择自己的文件。
也可保持 `auto`，按所选类型的常见文件名在标准目录及子目录中查找。

| ComfyUI目录 | 用途 |
| --- | --- |
| `models/diffusion_models/` | DiT权重 |
| `models/text_encoders/` | 文本编码器 |
| `models/vae/` | VAE |

支持 `extra_model_paths.yaml` 注册目录、子文件夹、自定义文件名。已有本地模型直接加载，
**不需要进入 Prepare，不需要下载凭证，不要求与本项目推荐文件的 SHA256 相等**。
相同后缀不代表模型格式相同；文件的架构、量化格式仍必须匹配所选加载方式。
各阶段按需加载；完整生成需要对应 DiT、TE 和 VAE，只运行编码节点不要求另外两类权重。

缺文件时可使用可选下载助手：在准备窗口展开下载区，勾选需要的文件，再点击“下载勾选项目”。
默认不勾选，支持单独下载 DiT、TE、VAE 或 Original TE bundle 内所需文件。
只校验通过助手下载的字节，不覆盖已有文件；运行环境准备不阻塞独立模型下载。
Original TE 真正执行时仍需要完整分片、index/config 与配套 tokenizer。

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
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets vae --dry-run
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets vae
```

`--verify-only`校验本地文件。下载先写临时文件，大小与SHA256匹配后写入目标；目标文件系统需支持硬链接，如NTFS或ext4。

## 离线FP8导出

导出接受本地原始dense BF16单文件或HF transformer目录，输出到新文件：

```sh
python tools/export_fp8.py --source /path/to/moss_image_base_bf16.safetensors --output /path/to/base_fp8_row.safetensors --variant base --precision fp8_compute --recipe row --device cuda:0
```

SFT使用其自身源文件或目录并指定 `--variant sft`。导出执行量化与格式转换，输出附带recipe和来源记录。
参数详情见 `python tools/export_fp8.py --help`。权重及转换来源见[conversion_manifest.json](docs/huggingface/conversion_manifest.json)，许可见[组件说明](THIRD_PARTY_NOTICES.md)。
