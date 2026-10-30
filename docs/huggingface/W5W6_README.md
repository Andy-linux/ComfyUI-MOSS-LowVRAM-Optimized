---
license: apache-2.0
base_model: OpenMOSS-Team/MOSS-Image-SFT
base_model_relation: quantized
pipeline_tag: text-to-image
tags:
  - moss-image
  - comfyui
  - quantization
  - low-bit
  - w5
  - w6
  - safetensors
---

# MOSS-Image-SFT W5/W6 packed low-bit DiT

Packed 5-bit and 6-bit diffusion transformers for
[MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), for use
in ComfyUI through the public [MOSS LowVRAM Optimized](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized) node pack. The former private preview plugin is no longer required for this installation path.

The 252 joint-block projections of the 42-layer DiT are stored as packed
integer codes with one FP32 scale per output row. At inference the activation
is rotated and quantized by ConvRot256 to INT8 with one row scale, the main
matrix runs INT8 x INT8 -> INT32 over the full K reduction, and the result is
converted to FP32 once, scaled, and stored as BF16. No dequantized copy of the
main weights is kept resident on the GPU. All other DiT tensors are retained
BF16.

These are research artifacts, not a reproduction of the BF16 model.

## Files

| File | Bits | Physical parameter bytes | DiT parameters |
| --- | ---: | ---: | ---: |
| `q5_ridge_gptq.safetensors` | 5 | 3,629,981,696 | 3.381 GiB |
| `q6_direct.safetensors` | 6 | 4,224,524,288 | 3.934 GiB |

For reference, the same DiT in BF16 occupies 9.467 GiB of parameters.

Each file contains 715 tensors: 252 packed `qweight` matrices, 252 FP32
`wscales` and 211 retained BF16 tensors (including one small rope buffer).

## Format

- `layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}.qweight`: uint8
  `[ceil(K*bits/8), N]`. Signed codes are packed along K and contiguous along
  N, with no padding inside a packing period: W5 stores 5 bytes per 8 weights,
  W6 stores 3 bytes per 4.
- `.wscales`: FP32 `[N, 1]`, one scale per output row.
- Safetensors metadata: `model=MOSS-Image-SFT`, `layout=k_bitstream_n_v1`,
  `convrot_groupsize=256`, `boundary_dtype=BF16`, `scale_dtype=FP32`,
  `scale_axis=output_row`, `weight_zero_point=0`, `low_rank=0`, `bits`.

This is this project's own layout. It is not GGUF and not a standard GPTQ
export, and the files cannot be loaded as a dense checkpoint.

The offline W5 selection (target-aware ridge/GPTQ, dual block128, damping 0.1)
only changes the integer codes and keeps the W5 direct FP32 row scales, so the
runtime operator and the parameter count do not change. W6 uses direct
round-to-nearest. Both were built from the original BF16 projection with the
FP32 ConvRot256 weight rotation; the calibration prompts and code are not part
of this repository.

## Usage / 首次使用

Use the public [MOSS LowVRAM Optimized node pack](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized), version 0.1.5 or newer.
Download [00_moss_lowvram.json](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/latest/download/00_moss_lowvram.json) and drag it into ComfyUI.
Install the single MOSS package from the missing-node prompt, apply changes/restart, then press **F5 / Ctrl+R** to load the new frontend extension.
The preparation dialog opens automatically. Follow **Prepare runtime → Restart and continue** when needed, then confirm the selected model download.
Wait for **Ready**, close the dialog and click **Run**. The plugin uses the Python running that ComfyUI, without replacing Torch or requiring a separately installed system Python.

没有 Manager 或插件市场连接失败时，从 [Releases](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/latest) 下载完整 ZIP。
保持 ComfyUI 打开，解压并双击 `Install-MOSS.cmd`，确认安装位置。完成后关闭并重新启动 ComfyUI、刷新网页，再拖入同一个工作流。
这条入口不依赖 Registry，不安装第二个 Manager；Python 依赖和模型仍需要网络下载。

For the default W5 workflow, keep **SFT / Packed W5 / INT8 × INT8 → INT32**. Choose **Packed W6** to prepare the W6 file instead.
The unified workflow uses 1024×1024, SFT 16 steps, CFG 3, Euler / sgm_uniform, shift 5 and SLG layer 22.
The text encoder and VAE are separate assets selected and downloaded by the same preparation dialog.
Do not load these packed files with an ordinary dense-weight loader.

The format and quantization description above concerns the unchanged stored checkpoints. The current plugin offers configurable FP16-fixed/BF16 boundaries;
that is distinct from the historical preview runtime and cached-conditioning measurements below.

## Historical preview measurements

The following retained results describe the original preview runtime, its 24-step procedure and archived conditioning. They are not a benchmark of the 0.1.5 unified 16-step workflow.

### Measured performance

RTX 4070 Ti SUPER (sm89), 1024x1024, batch 1, 24 steps, CFG 3, shift 5, SLG on
single layer 22, fixed archived prompts p01/p02/p03. KSampler interval includes
the DiT upload; text encoding and VAE decode are excluded. One measurement per
prompt; the first image may include JIT compilation. The six bundled kernel
configurations were selected on this GPU; other sm80+ parts are targets, not a
validated range.

| Path | DiT parameters | p01 s | p02 s | p03 s |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 9.467 GiB | 45.00 | 37.09 | 37.15 |
| W8A8 | 5.042 GiB | 27.97 | 25.81 | 25.90 |
| W5 ridge/GPTQ (packed) | 3.381 GiB | 32.62 | 29.52 | 29.58 |
| W6 direct (packed) | 3.934 GiB | 32.92 | 29.55 | 29.45 |

Full-RGB pixel RMSE against BF16 on the same prompts:

| Path | p01 | p02 | p03 |
| --- | ---: | ---: | ---: |
| W5 ridge/GPTQ | 0.057030 | 0.124148 | 0.127633 |
| W6 direct | 0.046576 | 0.115860 | 0.104264 |

In the real ComfyUI workflow (cached text conditioning, `--fp32-vae`) the peak
allocated VRAM was 3992.117 MiB for W5 and 4559.117 MiB for W6, with KSampler
at 32.1305 s and 33.2423 s for p01.

## Validation

Fixed conditioning (the archived r4 cache), same seed, 24 steps, W5:

- The release loader and the original loader produce the same latent bit for
  bit, in a plain process and inside the ComfyUI server (relative difference
  0.0). This is the equivalence claim for these weights.
- The server's live text encoding differs from the archived offline encoding by
  about 1e-6 (max absolute 5.05e-5 on the positive conditioning, relative
  9.81e-7). The archived conditioning was encoded in a `--cpu` process (fp32
  text encoder); a GPU-enabled process runs the same INT8 encoder in bf16.
  `inference_mode` does not change the encode.
- That 1e-6 difference is amplified by the INT8 activation quantizer over 24
  steps: the live-encoder smoke differs from the archived reference by relative
  latent 0.22 and PNG RMSE 0.0375 (W5) / 0.0305 (W6).
- Consequence: using the official text-encoder pipeline does not guarantee
  pixel-identical output to the archived offline conditioning. Use the cached
  conditioning if bit-exact reproduction is required.

## Limitations

- The three regression prompts are archived, previously seen prompts; they are
  not a fresh evaluation set. Pixel RMSE is not a perceptual or semantic score.
- Quality still drifts from BF16 in composition, object count and local shape;
  no broad BF16 fidelity is claimed.
- Only Windows with the tested Triton build and CUDA 13 PyTorch was validated.
- The text encoder used in the example is the local INT8 ConvRot Qwen3.5
  encoder; the archived measurements used cached SFT text conditioning.
- Using the official text-encoder pipeline does not guarantee pixel-identical
  output to the archived offline conditioning; see Validation above.

## License and attribution

MOSS-Image-SFT is released by the OpenMOSS Team under the Apache License 2.0.
These files are a modified (quantized) derivative of
`OpenMOSS-Team/MOSS-Image-SFT`, revision
`daf706d2e1c7a6369c17b6d1658671aee6db1ebe`. A copy of the license is included
as `LICENSE`. All credit for the model belongs to the OpenMOSS Team.

The ConvRot rotation and INT8 activation quantization come from
`comfy-kitchen` (Comfy Org, GPL-3.0) and are used at runtime. This is an
unofficial community quantization, not affiliated with or endorsed by OpenMOSS.
