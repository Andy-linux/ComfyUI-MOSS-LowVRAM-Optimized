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

The node pack and model weights are installed independently. Download only the checkpoint(s) you need; the plugin does not require the other files in this repository.

Place DiT files in `ComfyUI/models/diffusion_models/`, text encoders in `models/text_encoders/`, and VAEs in `models/vae/`. Subfolders and directories registered through `extra_model_paths.yaml` are supported. Select your files in the node dropdowns or preparation dialog. Existing compatible local weights load directly: there is no requirement to use the download helper, match a recommended filename/hash, or possess a preparation receipt.

The optional download helper starts with no files selected and downloads only explicitly checked files. For example, you can use your existing DiT and text encoder and download only a VAE. A complete image-generation graph requires its selected DiT, text encoder and VAE; executing only one stage does not require unused components.

Install [MOSS LowVRAM Optimized 0.1.6](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/tag/v0.1.6) and import [00_moss_lowvram.json](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/download/v0.1.6/00_moss_lowvram.json). If Registry review or marketplace access prevents Manager installation, the complete release ZIP includes `Install-MOSS.cmd`: keep ComfyUI running, double-click the installer, confirm the detected instance, then restart ComfyUI and refresh the page. It uses that instance's Python, not an unrelated system interpreter.

Runtime dependencies use compatible ranges and architecture-aware Triton selection. Only the documented SM89 configuration has the current full-image acceptance; SM75 and SM120 policies are not claims of physical testing on those cards. See [environment support](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/blob/main/docs/ENVIRONMENT_SUPPORT.md).

Choose Base or SFT, **Kohaku FP8 row** storage, and FP8 or 16-bit arithmetic. Each checkpoint supports both activation modes, so a second copy of the weight file is not needed. FP8 activation quantization changes numerics; 16-bit mode reuses the stored quantized weights rather than recovering the original dense weights. Both modes currently require SM89+ in the node pack.

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
