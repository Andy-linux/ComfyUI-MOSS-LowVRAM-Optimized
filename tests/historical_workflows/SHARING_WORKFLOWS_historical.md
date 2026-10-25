# Shareable workflows

Open the flat JSON files in `example_workflows/` in ComfyUI. Their same-name API
graphs live only in `tests/integration_workflows/`, so the template browser does
not display duplicate UI/API examples. The API files are prompt dictionaries;
submit them as the `prompt` field of a normal ComfyUI queue request.

| Example | DiT / TE | Steps | Process allocation profile |
| --- | --- | ---: | --- |
| 01_sft_w8_legacy_1024.json | Explicit public SFT W8 fused file / historical INT8 ConvRot TE | 16 | 6 GB |
| 02_base_w8_standard_1024.json | Base W8 / historical INT8 ConvRot TE | 50 | 8 GB |
| 03_base_fp8_row_standard_1024.json | Base Kohaku FP8 row / historical INT8 ConvRot TE | 50 | 8 GB |
| 04_base_w8_original_bf16_te_1024.json | Base W8 / original BF16 TE | 50 | 8 GB |
| 05_base_w8_real_w8a8_te_1024.json | Base W8 / real activation-INT8 TE | 50 | 8 GB |
| 06_sft_w5_lowvram_1024.json | Public SFT W5 / explicit INT8 ConvRot W8A16 TE / FP32 VAE | 16 | 4 GB |
| 07_sft_w6_lowvram_1024.json | Public SFT W6 / explicit INT8 ConvRot W8A16 TE / FP32 VAE | 16 | 6 GB |

Every graph uses 1024×1024, batch 1, fixed seed 314159265, CFG 3, Euler,
sgm_uniform, shift 5, SLG layer 22, an empty negative prompt and the same short
English/Chinese riverside-village scene. Base examples use the requested standard
50-step schedule; the SFT example retains the 16-step compatibility schedule.
The seed widget's control-after-generate is fixed, so comparable runs keep the
same seed. Changing model/precision can still change the generated image.

Attention is `auto`, avoiding optional private Flash/Sage binaries. TE memory is
`auto`; Examples 01–05 explicitly select BF16 VAE, a memory/precision choice with possible
pixel differences from FP32; the new low-VRAM examples 06/07 explicitly select FP32 VAE. These profiles limit process allocations and are
not promises of total physical VRAM consumption. Original TE uses ring under
the integrated auto policy. All variants still emit FP16 conditioning features.

Each UI graph contains Session, Paired Text Encode, EmptyFlux2LatentImage,
Sampler, VAE Decode, ordinary PreviewImage and SaveImage, an idle Controls node
and a frontend Note. Controls do not prewarm/unload during ordinary queueing;
use their buttons explicitly. Examples 01–05 retain the historical
`KohakuMOSS/<example stem>` save prefix; new 06/07 use `MOSS/<example stem>`.

## Models and precise provenance

The Session node's `properties.models` entries are derived from
`model_manifest.json`, using only `name`, `url`, `directory`, `hash`, `hash_type`.
URLs pin immutable revisions; hashes are the manifest SHA256 values. File sizes
remain in the manifest and the Note lists the aggregate available download size.
No unregistered `cnr_id` is included. Opening a graph does not download models.

The intended Registry package is `kohaku-moss-lowvram-optimized`; the display name
is **MOSS LowVRAM Optimized**, using KohakuMoss inference components. This is not
an assertion that the package is already registered. After publishing and
installing the real version, re-export templates with its actual `cnr_id` and
`ver`. Retain existing node class IDs so earlier JSON/PNG files remain loadable.

Base W8 examples select explicit manifest filenames. Example 01 now also selects
`moss_image_sft_int8_convrot.safetensors` explicitly. This public fused checkpoint
is distinct from the local R4 checkpoint previously selected by `auto`. Earlier
normal-step images and execution records belong to that old graph/resolved file;
they do not qualify the current exact public-file revision. That boundary remains
true; the current exact public-file graph now has its own successful normal-step
HTTP execution in `PUBLICATION_VALIDATION.md`, rather than inheriting the old PASS.

Every currently declared single-file `properties.models[].name` matches a concrete
selected widget value. This is necessary because official frontend selection
filters embedded metadata by widget strings; `auto` is not a substitute for a
filename in that mechanism. Preserve actual resolved hashes in runtime receipts.

The six quantized-TE examples 01/02/03/05/06/07 now explicitly select
`moss-image-qwen35-int8-convrot-v2.safetensors`. The same local alias appears in
their metadata names. The upstream file URL still uses the unversioned upstream
basename and its unchanged revision/SHA. This avoids a same-name installed v1
silently satisfying frontend missing-file checks. Browser downloads may require
renaming to the local alias and explicit hash verification. The separate prior
exact-v2 API check is retained as such; it does not certify every newly aliased
normal-step graph. The new seven-graph ledger supplies that separate current evidence.

The Original TE asset requires all four OpenMOSS BF16 shards, index/config and
tokenizer assets. Example 04 now selects the directory `MOSS-Image-Base/text_encoder`.
Its `properties.models` intentionally lists only the downloadable DiT and VAE
single files. Prepare the complete TE asset with the explicit downloader and keep
the prescribed layout:

```text
models/text_encoders/MOSS-Image-Base/text_encoder/
models/text_encoders/MOSS-Image-Base/tokenizer/
```

The current directory discovery adds installed folders containing
`model.safetensors.index.json` to the text-encoder choices.
Missing any required local asset must be resolved before queueing. Original BF16
is dense original-weight compute, not renamed INT8 dequantization. Changing from
the historical auto selector produces a revised graph. Its own 50-step HTTP run
now passed in the publishing-round ledger.

The full TE manifest is provenance, not a promise that the standard
Missing Models dialog installs an entire HF directory. The frontend's download
allowlist does not include JSON/Jinja/tokenizer configuration files, and the
directory selector does not match individual shard filenames. Prepare all assets
explicitly before queueing:

```sh
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets te_original --dry-run
python tools/download_models.py --models-dir /path/to/ComfyUI/models --assets te_original
```

For ordinary single-file models, Desktop can download through its host bridge to
model locations. A self-hosted browser generally opens a browser download; users
must place the files on the actual ComfyUI server or run the explicit downloader
there. A shared JSON does not transfer Python packages, drivers or the model files.

The FP8 row model is **local-only / publish pending** in the supplied manifest.
The template names `moss_image_base_kohaku_fp8_row.safetensors`, which must be
exported locally and selected after it exists. Its Note explicitly states this
prerequisite; no FP8 download URL, revision or hash has been invented. Download
metadata in that graph covers only its public TE and VAE. It is a prepared
integration example, not a claim of out-of-the-box downloadable FP8 weights.

The existing W5/W6 files now have a real public source in
`Andylinux/MOSS-Image-SFT-W5A8-W6A8`, pinned in the model manifest. They can be
downloaded with `--assets sft_w5` or `sft_w6`; no duplicate upload is needed.
Examples 06/07 now provide UI/API templates for these public files. They retain
the 01 layout, use 16 steps with fixed seed 314159265, auto attention, FP16-fixed
DiT, explicitly selected W8A16 TE, FP32 VAE and idle Controls. Both new
artifacts now passed their own 16-step HTTP executions; 06 was repeated after
formatting. Historical runs remain separate and were not used to fabricate PASS.
The three new FP8 outputs still have no user-selected HF destination.

## Precision labels

The historical `INT8 ConvRot` TE widget retains its existing W8A16 numerical
path. The new `INT8 ConvRot · W8A8` example requests true dynamic activation-INT8
GEMM after rotation. This is an experimental numerical option with possible
quality differences, not an accuracy or losslessness guarantee. It requires the
supported CUDA hardware/build and real runtime qualification. Do not rename an
old workflow to imply that its previous execution already used A8.

## Artifact validation and historical evidence

The seven UI/API pairs have matching settings and link references. Session has
20 persistent widgets (currently 7 required plus 13 optional, preserving order);
Sampler has 8 including seed control. Each graph has 9 UI nodes, 13 reciprocal
links and 8 API nodes. Embedded public single-file model names match selected
widget filenames and their URL/hash/destination metadata matches the manifest.
No graph invents a Registry `cnr_id` before actual publication.

The editing-round static check passed for **7/7 UI/API pairs**: 20 Session and
8 Sampler widgets per graph, 9 UI nodes, 8 API nodes and 13 reciprocal links.
Public model metadata entry counts for 01–07 were **3, 3, 2, 2, 3, 3, 3**;
every entry matched a selected filename and the manifest URL/hash/destination.
All seven samplers retained seed 314159265 with fixed control. Fixed enum values
were checked against current node definitions; the Original TE directory value
is supported by index-based directory discovery, but requires the complete
Original TE files to exist on the actual server. Static validation is not proof
that an installed server already has those files or has executed the revised graph.

## Current normal-step execution results

All **seven exact current UI/API pairs passed** the publishing-round normal
ComfyUI HTTP prompt API tests. Every request carried the UI workflow for PNG
metadata; schemas and widgets were checked separately. There were eight success
rows for seven distinct graphs because 06 W5 was rerun after formatting. Each
execution recorded two 1024x1024 PNG outputs with hashes. A separate actual Windows/
Chrome **v4 final check passed** final-Note 06 import and real button/dialog interaction:
9 nodes, 20 original-input persistent Session widgets plus two temporary widgets,
settled status, and a callback-refreshed report showing RTX 4070 Ti SUPER and
cudaMallocAsync. All six dialog checks passed; close removed the dialog and the
serialized 20 input values were unchanged. No console errors or POST /prompt were
observed. The status label/displayName both read “环境状态：依赖检查通过” and were
confirmed visible in a 2x screenshot. All v3 checks plus the new label check passed.
Only frontend label rendering changed; readonly/name/value and 20 inputs remain,
with unchanged UI SHA and no GPU/runtime changes or new execution counts. See
[the final receipt](validation/frontend-final-20261025.json). The unchanged
[v3 receipt](validation/frontend-dialog-20261025.json) preserves earlier successful
button/dialog results and the disabled-value visibility issue corrected by v4.
Only 06 has this browser result; the other six are not individually browser-certified.
The browser check did not generate an image and is separate from seven-graph GPU
execution. No new-image visual inspection is claimed.

The tests used the existing Windows environment with isolated standard model/node
directories, no `models.local.json` and no private attention packages. This was
not a fresh OS/Python installation. Four Base graphs ran 50 steps; the three SFT
graphs ran 16 steps. Current API SHA256 values match the execution ledger. The
release UI files have a subsequent Note-text-only update, recorded separately.

Windows tests passed 62/62. Official CLI 1.23.0 validation passed, with zero
selected E702/S102/S307 findings. The 32-file formatting proof has exact AST
identity; 01 and the repeated 06 ran after formatting, while earlier completed
runs remain connected by that proof. Recorded elapsed time is not a benchmark.

Full sanitized evidence: [`PUBLICATION_VALIDATION.md`](PUBLICATION_VALIDATION.md),
[`publication-20261025.json`](validation/publication-20261025.json) and
[`e702-formatting-proof.json`](validation/e702-formatting-proof.json).

The previous release's five-graph evidence in `../RELEASE_VALIDATION.md` and
`validation/results.json` remains historical. The old 01 used an auto-resolved
local SFT file and old 04 used auto directory resolution. Distributed old Save
PNGs still embed those historical graphs; they are not repackaged as this round's
new images. Final user-facing instructions replaced only the Note node's text
after execution. The original tested UI SHA256 is preserved; current release UI
SHA256 and unchanged API SHA256 are linked in
[`workflow-note-lineage.json`](validation/workflow-note-lineage.json). Node IDs,
links, execution settings and model metadata are unchanged. This is a documented
UI-text lineage, not a claim that GPU runs were repeated after changing a Note.

## Official contracts used by the sharing flow

- [Built-in template model links and Desktop/browser behavior](https://docs.comfy.org/interface/features/template)
- [Official template repository and model metadata](https://github.com/Comfy-Org/workflow_templates)
- [Registry publishing, immutable package identity and real publisher setup](https://docs.comfy.org/registry/publishing)
- [Frontend selected-model matching](https://github.com/Comfy-Org/ComfyUI_frontend/blob/d7251420b82f4ff52c60460350e563fa262186c5/src/platform/workflow/core/utils/modelRequirements.ts)
- [Frontend downloadable-file restrictions](https://github.com/Comfy-Org/ComfyUI_frontend/blob/d7251420b82f4ff52c60460350e563fa262186c5/src/platform/workflowErrors/missingModel/missingModelDownload.ts)

These contracts enable guided setup, not an unconditional one-click guarantee.
Missing-node installation also depends on Registry extraction/indexing and a
working Manager. Installation-time dependencies, optional binaries and model
access permissions remain separate prerequisites.
