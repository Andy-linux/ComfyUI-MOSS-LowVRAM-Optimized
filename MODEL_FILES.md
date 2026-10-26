# Model files and reproducible downloads

Code ships without weights. Opening/importing a workflow never downloads a model.
`model_manifest.json` pins each public file to a real HF API revision, exact byte
size, SHA256 and relative destination. Do not substitute Git blob IDs for SHA256.

## Default: prepare the selected files in the unified node

Open `example_workflows/00_moss_lowvram.json`, choose the model family, DiT
storage/compute, TE format/compute and VAE, then click **Prepare / check selected
configuration**. Review sizes, paths and sources before confirming download. The
new 00 default is SFT W5 + Sumomo v2 INT8 TE (W8A16 compute) + original MOSS VAE.
Only the current selection is prepared; no import-time model download occurs.
Existing registered model paths can be reused. A conflict is reported rather than
overwriting a mismatched file. Original TE preparation includes all shards,
index/config and tokenizer in the correct nested directories.

The independent maintenance CLI keeps its older default selection of Base INT8
(Sumomo separate Q/K/V), Sumomo v2 INT8 TE and original MOSS VAE. This older CLI
default is distinct from the new 00 UI default. Commands below are optional
maintenance tools, not required for the default user flow.

```sh
python tools/download_models.py --models-dir /path/to/ComfyUI/models --dry-run
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets base_w8 te_int8 vae
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets base_w8 te_int8 vae --verify-only
```

`--dry-run` validates metadata and prints exact total selected bytes without network
or filesystem writes. `--verify-only` reads local bytes, returns nonzero for a
missing/mismatched file and never downloads. Downloads refuse incomplete hashes or
floating revisions. Existing mismatched files are not overwritten. New bytes are
streamed to a same-directory temporary file, size/hash checked, then atomically
published without overwrite. This requires hard-link support (e.g. NTFS/ext4);
unsupported filesystems fail safely. Temporary files are removed after failures.
No Python package installation is performed. Credentials are never stored in JSON.

## Public and local assets

| Asset ID | File bytes (exact) | Status / scope |
|---|---:|---|
| `base_w8` | 5,413,723,800 | public; Default public Base INT8, separate Q/K/V; original validated community checkpoint. |
| `base_w8_comfy` | 5,413,698,958 | public; Alternative fused-QKV packing; distinct file/hash, requires normalization; do not silently substitute. |
| `sft_w8` | 5,413,698,958 | public; Public SFT INT8 fused variant; distinct from local benchmark weight; verify checkpoint compatibility before claiming R4 equality. |
| `te_int8` | 8,452,959,908 | public; Pinned Sumomo v2 core-compatible keys; 8,452,959,908 bytes. Matches existing community asset name. |
| `te_int8_comfy` | 9,748,121,532 | public; Alternative 9,748,121,532 bytes; NOT byte-identical to 8.45GB Sumomo TE. Do not relabel local measured TE as this file. |
| `te_w4` | 7,343,591,088 | public; Optional public W4 TE; a different precision path, not default R4 INT8. |
| `vae` | 249,521,372 | public; Default original MOSS VAE, identical LFS bytes to original HF VAE. |
| `vae_flux2_small` | 249,521,340 | public; Alternative repack, different file/hash; not silently interchangeable based on similar file size. |
| `te_original` | 16,805,544,650 | public; Original BF16 TE: all four shards plus index/config and tokenizer. Exact small-file SHA256 required; downloader refuses incomplete metadata. |
| `sft_w5` | 3,630,052,176 | public; existing Andylinux SFT W5 packed checkpoint, pinned revision/hash; matches the measured local artifact. |
| `sft_w6` | 4,224,594,400 | public; existing Andylinux SFT W6 packed checkpoint, pinned revision/hash; matches the measured local artifact. |
| `base_fp8` | 5,174,879,864 | Base Kohaku FP8 row; public at the fixed R1 below. One file supports FP8 or 16-bit compute. |
| `sft_fp8` | 5,174,879,840 | SFT Kohaku FP8 row; public at the same fixed R1 below. Distinct SFT bytes, not a renamed Base model. |
| `base_fp8_weight_only` | 5,346,415,987 | Legacy local-only separate weight-only export; not included in the new two-row public distribution. Existing explicit local loading remains compatibility scope. |

The 9,748,121,532-byte Comfy-Org INT8 TE differs from the 8,452,959,908-byte
Sumomo v2 file. It is an alternative, not proof of the same measured runtime
weights or layout. Likewise fused Comfy-Org DiT and separate Sumomo DiT have
different hashes. Loader normalization and real qualification must precede any
claim that their outputs match. Similar VAE names/sizes do not imply equality.

The default `te_int8` asset uses a versioned **local destination**:
`text_encoders/moss-image-qwen35-int8-convrot-v2.safetensors`.
Its upstream filename remains `text_encoders/moss-image-qwen35-int8-convrot.safetensors`,
with the same pinned revision, 8,452,959,908 bytes and SHA256 `511379...` recorded
in the manifest. The local alias prevents an installed legacy v1 file with the
same upstream basename from silently satisfying name-only frontend detection.
Desktop/downloader placement uses the destination alias. A self-hosted browser
download may retain the upstream basename and require an explicit local rename;
prefer the unified GUI preparation/verification to place and check the correct alias,
or use maintenance `--verify-only` to verify actual bytes. No weights are rewritten.

Original `te_original` downloads all four BF16 shards, index/config and tokenizer
to `text_encoders/MOSS-Image-Base/`. The model asset resolver finds its
`text_encoder` directory there. This is much larger than quantized TE; choose it
explicitly. The SFT original repository has the same four TE shard hashes in the
captured API snapshot; it is recorded as an alternate source, not auto-substituted.

## Existing public W5/W6 — index, do not upload again

Both files already exist in
[Andylinux/MOSS-Image-SFT-W5A8-W6A8](https://huggingface.co/Andylinux/MOSS-Image-SFT-W5A8-W6A8)
at immutable revision `bfb77cb73ae7acb4f788d50d4043c2d71989b0a7`.
The public file hashes match the measured local bytes:

| Asset | Filename | SHA256 |
|---|---|---|
| `sft_w5` | `q5_ridge_gptq.safetensors` | `3f8a775e179a63a7511d469f08430b89f5807ebd181f37f4bbaea7d039096f82` |
| `sft_w6` | `q6_direct.safetensors` | `dc5af59f22eb89320e457536ae90b170c6548c74a9fca584684139cf88018f87` |

```sh
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets sft_w5 sft_w6 --dry-run
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets sft_w5
```

Review the pinned repository's
[README](https://huggingface.co/Andylinux/MOSS-Image-SFT-W5A8-W6A8/blob/bfb77cb73ae7acb4f788d50d4043c2d71989b0a7/README.md),
[LOWBIT_MANIFEST.json](https://huggingface.co/Andylinux/MOSS-Image-SFT-W5A8-W6A8/blob/bfb77cb73ae7acb4f788d50d4043c2d71989b0a7/LOWBIT_MANIFEST.json)
and [Apache license](https://huggingface.co/Andylinux/MOSS-Image-SFT-W5A8-W6A8/blob/bfb77cb73ae7acb4f788d50d4043c2d71989b0a7/LICENSE).
The model repository's weight license is distinct from the packaged inference-code
licenses. Matching file hashes establishes artifact identity, not automatic
qualification of every runtime/GPU combination. These existing weights will not
be re-uploaded to a new HF repository.

## Self-produced FP8: two row files, shared compute choices

The destination is
[Andylinux/MOSS-Image-Kohaku-FP8](https://huggingface.co/Andylinux/MOSS-Image-Kohaku-FP8).
Base and SFT row checkpoints are the two files selected for public distribution.
The two Base/SFT FP8 row checkpoints are public at [R1 `d8a0b2f46a79770fe711bf5322cf6cb3219c8d41`](https://huggingface.co/Andylinux/MOSS-Image-Kohaku-FP8/tree/d8a0b2f46a79770fe711bf5322cf6cb3219c8d41), totaling 10,349,759,704 bytes. Exact sizes and remote LFS SHA256 match local full-file SHA256; small companion files were read back. The approximately 10GB remote weights were not downloaded again for a full-stream SHA256 check.
Weight upload commit R0 is `082aebf29d592d9a9f0ba61b64f780ad160ec72b`; R1 adds the finalized documentation with unchanged weight identities. Registry publication and actual Manager installation/download/generation remain pending.

| Asset | Exact bytes | Local full-file / remote LFS SHA256 | Compute choice |
| --- | --- | --- | --- |
| `base_fp8` | 5,174,879,864 | `4f18284539c84c804919e7c343ee33b1550dfcd1c4cd5dc5dd7f4569905d0f14` | FP8 / 16-bit, using the same row file |
| `sft_fp8` | 5,174,879,840 | `06c58d186852abd0fd073fba6486a82543a6006a8c17a3eac7d62828b9c03862` | FP8 / 16-bit, using the same row file |

Storage and matrix arithmetic remain separate. Switching to 16-bit compute does
not require a second weight-only download. The old 5,346,415,987-byte independent
Base weight-only export is retained only for legacy local loading; it is not a
third public upload. Original conversion/source hashes remain in
`docs/huggingface/`; historical export receipts are preserved unchanged.
Private `models.local.json` mappings remain excluded from release packaging.

## Workflow download boundary

Official frontend model metadata must match a concrete selected widget filename.
A node's `properties.models` list is not an instruction to install every listed
file: an `auto` widget or an HF directory does not select all associated model
records. Use explicit recommended filenames for downloadable single-file examples.
The current SFT example uses the verified public filename explicitly; see
`docs/SHARING_WORKFLOWS.md` for the metadata and placement rules.

The Original TE is a complete directory with shards, config/index and tokenizer.
Use the new unified node preparation GUI for that complete asset. The optional
maintenance equivalent is `--assets te_original`. Do not promise that ordinary
frontend Download all can fetch every JSON/Jinja file or reconstruct the directory.
Desktop and self-hosted browser downloading have different placement behavior.

## Dependency and license boundaries

Existing ComfyUI supplies Torch/CUDA. Manager requirements processing and its
install.py hook prepare missing Triton only under the narrow policy documented in
[INSTALLATION_GUIDE](docs/INSTALLATION_GUIDE.md). The exact Windows automatic
preparation stack is x64 Python3.12 / Torch2.10.0+cu130 / SM89; SM75 has a historical
nonstandard pairing, not a new workflow qualification. Unknown hosts are diagnosed,
not upgraded to latest or given replacement Torch/CUDA/Python. Windows Manager
may perform queued preparation at the next restart. Runtime routes/checks never
install packages. Sage is optional and not installed by the default workflow.
RTX50 qualification remains pending; RTX30 full-workflow support is not promised.
The user authorized a separate GitHub Release for the self-built SM75 Turing
bundle; the source package still excludes its binaries. The 43,707,834-byte ZIP and 17-member manifest are now public in the fixed GitHub
Release. Anonymous complete HTTP200 download and remote asset digest matched
the pinned SHA256, so strict-ABI Manager preparation is enabled. The pinned upstream repository has no
confirmed license declaration; actual source/build/component-license/NOTICE
records will be preserved without inventing blanket clearance. Manager prepares
only the fixed strict-ABI artifact once published; default auto does not need it.
The package requires Python >=3.12. Normal declared requirements include
`accelerate`; use the same Python that launches ComfyUI. Current testing uses
Python 3.12, not every later version admitted by the lower-bound declaration.

Upstream model-card license declarations are attribution data, not a blanket
license covering vendored kernels or every derivative. Preserve
`LICENSE_PROVENANCE.json`, `THIRD_PARTY_NOTICES.md`, `runtime/vendor/SOURCES.json`
and component license files. The current component source/license inventory has been checked. Preserve these notices and publish only the clean release history.
