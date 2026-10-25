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

# MOSS-Image — shared Kohaku FP8 weights

Two quantized diffusion-transformer checkpoints, **Base** and **SFT**, for
[MOSS LowVRAM Optimized](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized).
Each file can supply both the FP8-activation and 16-bit-activation inference modes
implemented by the node pack. Text encoder, tokenizer and VAE are separate assets.

These are conversions of existing model weights. No training, fine-tuning,
LoRA merge or distillation was performed, and the source files were not modified.
Each output contains **263 quantized Linear layers**, including the text adapter,
refiner and 252 joint projections. Training transpose/dgrad copies are not exported.

## Files

| File | Model | Exact bytes | SHA256 |
|---|---|---:|---|
| `moss_image_base_kohaku_fp8_row.safetensors` | Base | 5,174,879,864 | `4f18284539c84c804919e7c343ee33b1550dfcd1c4cd5dc5dd7f4569905d0f14` |
| `moss_image_sft_kohaku_fp8_row.safetensors` | SFT | 5,174,879,840 | `06c58d186852abd0fd073fba6486a82543a6006a8c17a3eac7d62828b9c03862` |

Total: **10,349,759,704 bytes** (about 9.64 GiB). File size is not pipeline RAM or
VRAM usage. The text encoder, VAE, activations, runtime buffers and operating system
consume additional memory.

## Same stored weights, selectable inference arithmetic

The stored row format contains an E4M3 FP8 weight-code tensor `q[N,K]` and a
float32 scale for each output row. Dequantization represents approximately
`W[n,k] = q[n,k] * scale[n]`. The checkpoint records the Kohaku row recipe and
its quantized layer list in `kohakumoss_fp8` metadata.

- **FP8 activations:** the runtime requests Kohaku FP8 compute, with dynamic
  activation quantization and row scales. Vendor dispatch can choose a 16-bit
  fallback for a particular shape; the FP8 option does not claim every matrix
  multiplication always uses native FP8 instructions.
- **16-bit activations (weight-only):** the runtime reuses the same `q` and scale
  tensors through a weight-only wrapper. Activations remain FP16 or BF16 according
  to the selected floating precision; weights are dequantized for the calculation.
  It does not quantize the checkpoint again or materialize a second full dense
  model merely to switch this option.

Changing the activation mode creates a distinct runtime session/cache identity.
The compatible node pack is required: arbitrary readers of the file's stored
FP8-compute metadata will not automatically implement this alternate mode.

An older, separate local `moss_image_base_kohaku_fp8_weight_only.safetensors`
export is **not required and is not distributed here**. Its nonquantized tensors
retain some FP32 values that the row exports store in BF16. Therefore this shared
row weight-only mode is not asserted to be byte-identical or image-identical to
that older export. Switching activation arithmetic cannot recover precision
already lost during weight conversion.

## Sources and reproducibility

The exact source byte hashes are recorded in
[conversion_manifest.json](conversion_manifest.json). Immutable public snapshots
containing matching bytes have been verified:

- Base: [Comfy-Org/MOSS-Image at 24147db865fcdc207270f844777587223fed8d6e](https://huggingface.co/Comfy-Org/MOSS-Image/tree/24147db865fcdc207270f844777587223fed8d6e),
  `diffusion_models/moss_image_base_bf16.safetensors`, 10,164,655,808 bytes,
  SHA256 `cacd158728a1fb9e57c408a922b54752ec71fceb566b3821269927964e62b41f`.
- SFT: [OpenMOSS-Team/MOSS-Image-SFT at f972bd4b4b9e3852133cc5da954d87bc32cd8fb9](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9),
  all three `transformer/diffusion_pytorch_model-0000*-of-00003.safetensors`
  shards. Each shard's bytes and hash are in the conversion manifest.

These matching public revisions identify reproducible input bytes. The revision
used by the historical download session, exact conversion timestamp and conversion
device were not retained in the available receipt; they are not inferred from the
filenames. The manifest uses null for these unconfirmed historical facts.

The converter is `tools/export_fp8.py` in the node repository, using its preserved
Kohaku model IO and quantization implementation. Row exports store nonquantized
floating tensors in BF16 except `rope.bands` in FP32. Conversion code and model
weights have separate licenses; see below.

## Installation and current validation scope

The ComfyUI node pack's model preparation interface resolves these fixed filenames,
checks exact byte sizes and SHA256, and downloads only the selected model.
Choose model family Base or SFT, storage **Kohaku FP8 row**, then FP8 or 16-bit
activation arithmetic. Use Base's or SFT's appropriate sampling preset; these are
different pretrained model variants, not interchangeable precision settings.

Existing limited Windows SM89 FP8-compute inference and isolated dispatch probes
are recorded in the node repository's release evidence. The new explicit row-to-A16
switch is being integration-tested for this release; no completed result for that
new path is claimed by this initial card. The node pack currently keeps the SM89+
hardware gate for both shared modes. A16 weight-only storage alone is not a claim
of support for an untested GPU architecture.

No lossless quality, speedup, full-shape native-FP8 coverage, or physical 4/8 GB card
certification is claimed. A VRAM allocation profile is a software budget, not the
physical capacity of the test GPU. Linux, 30-series and 50-series qualification
must be checked against the node pack's current compatibility table.

## Licenses and attribution

Upstream model weights declare Apache-2.0. The original license is preserved as
[LICENSE](LICENSE); no new ownership of OpenMOSS weights is asserted. These files
are quantized derivatives, with the changes described above and in the manifest.

The converter reuses [KohakuMoss](https://github.com/KohakuBlueleaf/KohakuMossTrainer),
whose code is under its preserved KohakuMoss License (Apache terms plus a naming
condition). That code license and its naming clause are not relabeled as the
model-weight license; its output exception is recorded in the manifest.
OpenMOSS, KohakuMoss, ComfyUI and ComfyKitchen attribution and component licenses
remain in the node repository's NOTICE and provenance files.
