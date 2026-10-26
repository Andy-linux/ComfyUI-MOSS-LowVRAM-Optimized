---
license: apache-2.0
base_model:
  - OpenMOSS-Team/MOSS-Image-Base
  - OpenMOSS-Team/MOSS-Image-SFT
base_model_relation: quantized
pipeline_tag: text-to-image
language: [en, zh]
tags: [moss-image, kohaku, fp8, quantization, comfyui]
---

# MOSS-Image — Kohaku FP8 row weights

Base and SFT diffusion-transformer checkpoints for [MOSS LowVRAM Optimized](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized).
Each row file supports FP8 or 16-bit activation arithmetic in the node pack. Text encoder, tokenizer and VAE are separate assets.

## Files

| File | Model | Bytes |
| --- | --- | ---: |
| `moss_image_base_kohaku_fp8_row.safetensors` | Base | 5,174,879,864 |
| `moss_image_sft_kohaku_fp8_row.safetensors` | SFT | 5,174,879,840 |

Total: 10,349,759,704 bytes (about 9.64 GiB). Pipeline memory also includes the text encoder, VAE, activations and runtime buffers.

## Usage

Import the node pack's `00_moss_lowvram.json` into ComfyUI. In the configuration node, choose Base or SFT, **Kohaku FP8 row** storage,
and FP8 or 16-bit compute. Click **Prepare / check selected configuration** to prepare the selected assets.

FP8 mode uses activation quantization; kernel dispatch may select a 16-bit path for particular shapes.
16-bit mode reuses the stored quantized weights with FP16/BF16 activations. Both modes currently require SM89+ in the node pack.
Current primary platform is Windows/NVIDIA RTX40; see the node pack's [environment support](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/blob/main/docs/ENVIRONMENT_SUPPORT.md) for other platforms.

## Sources and conversion

These are quantized conversions of existing model weights. Row exports store E4M3 weight codes with an FP32 scale per output row;
nonquantized floating tensors use BF16 except `rope.bands` in FP32. Conversion uses the node pack's `tools/export_fp8.py` and Kohaku model IO/quantization.

- Base source: [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image/tree/24147db865fcdc207270f844777587223fed8d6e), `diffusion_models/moss_image_base_bf16.safetensors`.
- SFT source: [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9), three transformer shards.

Exact input/output hashes, revisions, conversion recipe and source facts are preserved in [conversion_manifest.json](conversion_manifest.json) and [SHA256SUMS](SHA256SUMS).

## License and attribution

Upstream model weights declare Apache-2.0; the original terms are preserved in [LICENSE](LICENSE). These files are quantized derivatives of OpenMOSS weights.
The converter reuses [KohakuMoss](https://github.com/KohakuBlueleaf/KohakuMossTrainer), whose code uses the KohakuMoss License with its naming condition and output exception.
Code and weight licenses apply to their respective components. OpenMOSS, KohakuMoss, ComfyUI and ComfyKitchen attribution remains in the node repository's NOTICE and provenance files.
