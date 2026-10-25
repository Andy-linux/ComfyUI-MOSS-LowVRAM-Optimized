---
license: apache-2.0
base_model:
  - OpenMOSS-Team/MOSS-Image-Base
  - OpenMOSS-Team/MOSS-Image-SFT
base_model_relation: quantized
pipeline_tag: text-to-image
language:
  - en
  - zh
tags:
  - moss-image
  - kohaku
  - fp8
  - quantization
  - comfyui
---

# MOSS-Image — Kohaku FP8 DiT conversions

**Publication draft.** These three files were produced and hashed locally. No
Hugging Face destination repository or public download links have been assigned.
This card describes the exact artifacts below; it does not announce an upload.

These are quantized **diffusion-transformer components**, derived from the
MOSS-Image Base and SFT weights. They are not complete pipelines: the compatible
text encoder, tokenizer, VAE and scheduler must be supplied separately. No training,
fine-tuning, LoRA merge, distillation or changes to the source files were performed
for these conversions. Each output contains **263 quantized Linear layers**,
including the text adapter/refiner layers as well as the 252 joint projections.
Training transpose/dgrad copies are not exported (`dgrad=false`).

## Files

| File | Variant and format | Exact bytes | GiB |
|---|---|---:|---:|
| `moss_image_base_kohaku_fp8_row.safetensors` | Base, Kohaku stored FP8 compute, row recipe | 5,174,879,864 | 4.8195 |
| `moss_image_sft_kohaku_fp8_row.safetensors` | SFT, Kohaku stored FP8 compute, row recipe | 5,174,879,840 | 4.8195 |
| `moss_image_base_kohaku_fp8_weight_only.safetensors` | Base, FP8 weight-only | 5,346,415,987 | 4.9792 |

GiB means bytes divided by 2³⁰. File size is not total pipeline size, peak host RAM,
or required GPU memory. The two Base files use different serialization contracts
and retained dense-tensor dtypes; their different sizes do not indicate missing data.

The exact output hashes are in [SHA256SUMS](SHA256SUMS). Source and output identities,
format details and explicit unconfirmed fields are in
[conversion_manifest.json](conversion_manifest.json). These values come from local
byte-level measurements, not from filename assumptions.

## Row FP8 compute versus weight-only FP8

Both formats store E4M3 (`float8_e4m3fn`) weight codes with FP32 per-output-row
scales. **They require different loader behavior.**

| Property | Kohaku row FP8 compute | FP8 weight-only |
|---|---|---|
| File-level record | Safetensors `kohakumoss_fp8` JSON, `recipe="row"` and explicit layer list | No `kohakumoss_fp8` record |
| Per-layer record | `.weight` codes and `.weight_scale` | `.weight`, `.weight_scale`, and `comfy_quant` declaring `float8_e4m3fn`, no ConvRot |
| Activations | Dynamically quantized on FP8-compute paths | Remain in the selected 16-bit compute dtype |
| Arithmetic selection | Vendor chooses a path by device and shape; may select `w16` | Weight-only operation; not FP8 W8A8 compute |
| Training weights | Frozen | Frozen |

The row-compute exports write other floating tensors as BF16, while `rope.bands`
remains FP32. The weight-only export retains the converter's live dense dtypes:
BF16 plus FP32 for its `KEEP_FP32` sensitive parameters and rope. Its larger file
size must not be misreported as different measured quality or missing row weights.
Neither output is MXFP8. INT8 ConvRot and the existing W5/W6 local options are
separate formats and are **not included in this three-file publication draft**.

An FP8 storage dtype or an `fp8_linear_infer` binding does not prove every GEMM
executes with FP8 operands. The upstream selector can choose a **dequantized
16-bit matmul (`w16`)** for particular shapes. Actual dispatch must be inspected
alongside profiler evidence. No blanket throughput advantage is claimed.
On that w16 path, activations remain 16-bit; a row-compute checkpoint therefore
does not imply activation quantization on every actual dispatch.

## Provenance

### Base source

The measured input is the BF16 single-file repack from
[Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image),
`diffusion_models/moss_image_base_bf16.safetensors`:

- Exact bytes: `10164655808`
- SHA256: `cacd158728a1fb9e57c408a922b54752ec71fceb566b3821269927964e62b41f`

This repack originates from
[OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base).
The exact local download snapshot revision has not been independently confirmed;
the measured source file hash fixes its content identity.

### SFT source

The input is the local transformer shard set obtained from
[OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT).
Its snapshot revision is **not confirmed** and is deliberately not invented.

| Transformer source shard | Exact bytes | SHA256 |
|---|---:|---|
| `diffusion_pytorch_model-00001-of-00003.safetensors` | 4,991,978,016 | `e40d3444940a0b0fd7cd45ba7959a5bd3d5efd7ddf87aae5ce4d87ce8aa03d01` |
| `diffusion_pytorch_model-00002-of-00003.safetensors` | 4,983,137,016 | `d106da790ec2924ffd8af04b4ffa8f8de281a974b58e55649c3ec596dc8ef033` |
| `diffusion_pytorch_model-00003-of-00003.safetensors` | 189,549,184 | `11f637ab17c7093344a6110a1e87ffda8e56558dc55183d8e790222cfcdcf52d` |

Base and SFT have the same architecture but different weights. A Base filename
must not be used to relabel SFT parameters. The conversion reads each model's own
checkpoint tensors and does not substitute an SFT retained-parameter sidecar.

## Loading

Use a loader that explicitly understands the relevant record and scale layout,
such as the matching **MOSS LowVRAM Optimized** integration using KohakuMoss. A generic FP8 dtype
cast or arbitrary Diffusers `from_pretrained` call is not a documented loader for
these files. This draft provides no unassigned plugin or model download URL.

For local ComfyUI use, place the selected DiT file under `models/diffusion_models/`
and explicitly select its variant and FP8 file in the compatible integration. Use
its separate text-encoder, tokenizer and VAE assets. Verify the file hash with
`SHA256SUMS` separately, then confirm that the loader reports the expected variant,
format and 263 stored quantized layers.

The row and weight-only paths can use bounded joint-q payload placement where the
integration supports it. Scale/aux buffers and nonjoint tensors still require
memory. The original text adapter has its own placement and rebinding lifecycle.
Successful checkpoint loading alone does not prove that a chosen VRAM profile or
every inference shape is supported.

## Conversion reproducibility

Conversion uses the packaged Kohaku IO/quantization APIs through
`tools/export_fp8.py` and `runtime/fp8_support.py`. The vendor origin is
KohakuMossTrainer commit `5223637706bc3b847b27589ad18e9221226ffa93`, with the
integration's documented local changes. That origin commit is not a claim that
every distributed converter byte is unchanged upstream code.

For example, from the plugin checkout, with a compatible installed runtime and
the **measured local BF16 source**:

```bash
python tools/export_fp8.py --source /local/moss_image_base_bf16.safetensors --output /local/moss_image_base_kohaku_fp8_row.safetensors --variant base --precision fp8_compute --recipe row --device cpu
```

For weight-only output, use `--precision fp8_weight_only`. For SFT row output,
select the SFT transformer folder and `--variant sft`. This is a reproduction
example, not a reconstructed log of the historical conversion device or command.
CPU conversion needs sufficient host RAM. Existing output files are refused; the
source files remain unchanged. Check the regenerated output and receipts rather
than assuming that a different software stack produces identical bytes.

## Validation and limitations

The validation environment is **Windows, NVIDIA SM89, PyTorch 2.10 with a CUDA
13.0 build**. The integration has reported **limited smoke passes in 4g and 8g
profiles**. These profile labels are operating budgets, not proof of testing every
physical GPU sold with 4 GB or 8 GB of VRAM.

Smoke coverage is finite and varies by checkpoint, profile, dtype and dispatch.
It must not be read as a completed all-artifact/all-profile matrix. In particular,
local export and checksum verification alone are not image-generation tests.
Consult the actual per-case runtime receipts and release validation when they are
attached; missing cases are not assumed to pass.

In a separate real-weight row-compute probe reported by the operator, q,
ffn-up/gate and ffn-down at M=64 and M=4096 formed six isolated cases. All six
selected Triton configs; the profiler recorded the row-quantization and scaled
kernel three times each per measured case, and no w16 selection was observed in
those cases. This observation covers those tested layers and shapes only. It is
not a full-model, all-shape hardware-instruction certification or a throughput
comparison.

No claim is made of lossless image quality, universal BF16 equivalence, faster
inference at every shape, or new small-card certification. Linux and other GPU
architectures have not been qualified by this validation boundary. Earlier R4
INT8/W5/W6 performance results do not transfer automatically to these FP8 files.
Use matched prompts, seeds, samplers, text embeddings and VAE settings when
evaluating quality or speed.

## Licenses and attribution

**Model weights:** the source MOSS-Image weights are declared **Apache-2.0** by
their upstream repositories. The quantized derivative weights retain that weight
license and applicable upstream notices. Preserve the actual upstream license
text and notices when publishing. See the
[Comfy-Org license](https://huggingface.co/Comfy-Org/MOSS-Image/blob/24147db865fcdc207270f844777587223fed8d6e/LICENSE)
and [OpenMOSS-Team SFT license](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/main/LICENSE).
No new copyright owner is asserted here.

**Converter and integration code:** the Kohaku source has the separate
[KohakuMoss License 1.0](https://github.com/KohakuBlueleaf/KohakuMossTrainer/blob/5223637706bc3b847b27589ad18e9221226ffa93/LICENSE),
which adds a Section 10 naming requirement to Apache terms. Do not relabel that
code as plain Apache-2.0. Section 10 expressly excludes models, adapters, images
and other outputs produced by running the work from its derivative-work naming
requirement. Using the converter does not make its code license replace the
model-weight license.

This is an independent conversion draft, not an official OpenMOSS or Comfy-Org
release. Upstream names identify provenance and do not imply endorsement.
