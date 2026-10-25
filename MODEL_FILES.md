# Model files and reproducible downloads

Code ships without weights. Opening/importing a workflow never downloads a model.
`model_manifest.json` pins each public file to a real HF API revision, exact byte
size, SHA256 and relative destination. Do not substitute Git blob IDs for SHA256.

The default explicit selection is Base INT8 (Sumomo separate Q/K/V), Sumomo v2
INT8 TE and original MOSS VAE. It does not fetch W5/W6, FP8, SFT or every variant.

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
| `sft_w5` | — | local_only/publish_pending; No user HF namespace/repo supplied. No public URL/revision/hash invented; not downloadable. |
| `sft_w6` | — | local_only/publish_pending; No user HF namespace/repo supplied. No public URL/revision/hash invented; not downloadable. |
| `base_fp8` | — | local_only/publish_pending; No user HF namespace/repo supplied. No public URL/revision/hash invented; not downloadable. |
| `sft_fp8` | — | local_only/publish_pending; No user HF namespace/repo supplied. No public URL/revision/hash invented; not downloadable. |

The 9,748,121,532-byte Comfy-Org INT8 TE differs from the 8,452,959,908-byte
Sumomo v2 file. It is an alternative, not proof of the same measured runtime
weights or layout. Likewise fused Comfy-Org DiT and separate Sumomo DiT have
different hashes. Loader normalization and real qualification must precede any
claim that their outputs match. Similar VAE names/sizes do not imply equality.

Original `te_original` downloads all four BF16 shards, index/config and tokenizer
to `text_encoders/MOSS-Image-Base/`. The model asset resolver finds its
`text_encoder` directory there. This is much larger than quantized TE; choose it
explicitly. The SFT original repository has the same four TE shard hashes in the
captured API snapshot; it is recorded as an alternate source, not auto-substituted.

W5/W6 local names are `q5_ridge_gptq.safetensors` and
`q6_direct.safetensors`. FP8 Base/SFT row and Base weight-only exports have now been produced locally by
the operator; their public HF identity remains pending. Local export completion
is distinct from GPU/shape/UI qualification; the final evidence document records
those results.
These entries remain `local_only/publish_pending`: supply local files via the
normal loader or `models.local.json`. That machine-specific mapping is ignored by
git/Registry packaging. No fictional HF owner, URL or hash has been generated.

## Dependency and license boundaries

Existing ComfyUI supplies Torch/CUDA. Install a tested Triton implementation for
your platform (Windows triton-windows versus Linux Triton), compatible with its
Torch/CUDA/Python ABI. This is an explicit prerequisite; requirements do not
replace Torch or automatically choose a GPU toolchain. Sage is optional and should
be installed only through the documented explicit platform-specific procedure.

Upstream model-card license declarations are attribution data, not a blanket
license covering vendored kernels or every derivative. Preserve
`LICENSE_PROVENANCE.json`, `THIRD_PARTY_NOTICES.md`, `runtime/vendor/SOURCES.json`
and component license files. The current component source/license inventory has been checked. Preserve these notices and publish only the clean release history.
