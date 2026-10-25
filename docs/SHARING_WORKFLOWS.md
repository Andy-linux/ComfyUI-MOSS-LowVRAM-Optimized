# Shareable workflows

Open the flat JSON files in `example_workflows/` in ComfyUI. Their same-name API
graphs live only in `tests/integration_workflows/`, so the template browser does
not display duplicate UI/API examples. The API files are prompt dictionaries;
submit them as the `prompt` field of a normal ComfyUI queue request.

| Example | DiT / TE | Steps | Process allocation profile |
| --- | --- | ---: | --- |
| 01_sft_w8_legacy_1024.json | SFT W8 / historical INT8 ConvRot TE | 16 | 6 GB |
| 02_base_w8_standard_1024.json | Base W8 / historical INT8 ConvRot TE | 50 | 8 GB |
| 03_base_fp8_row_standard_1024.json | Base Kohaku FP8 row / historical INT8 ConvRot TE | 50 | 8 GB |
| 04_base_w8_original_bf16_te_1024.json | Base W8 / original BF16 TE | 50 | 8 GB |
| 05_base_w8_real_w8a8_te_1024.json | Base W8 / real activation-INT8 TE | 50 | 8 GB |

Every graph uses 1024×1024, batch 1, fixed seed 314159265, CFG 3, Euler,
sgm_uniform, shift 5, SLG layer 22, an empty negative prompt and the same short
English/Chinese riverside-village scene. Base examples use the requested standard
50-step schedule; the SFT example retains the 16-step compatibility schedule.
The seed widget's control-after-generate is fixed, so comparable runs keep the
same seed. Changing model/precision can still change the generated image.

Attention is `auto`, avoiding optional private Flash/Sage binaries. TE memory is
`auto`; VAE precision is explicitly BF16, a memory/speed choice with possible
pixel differences from FP32. These profiles limit process allocations and are
not promises of total physical VRAM consumption. Original TE uses ring under
the integrated auto policy. All variants still emit FP16 conditioning features.

Each UI graph contains Session, Paired Text Encode, EmptyFlux2LatentImage,
Sampler, VAE Decode, ordinary PreviewImage and SaveImage, an idle Controls node
and a frontend Note. Controls do not prewarm/unload during ordinary queueing;
use their buttons explicitly. Images save under `KohakuMOSS/<example stem>`.

## Models and precise provenance

The Session node's `properties.models` entries are derived from
`model_manifest.json`, using only `name`, `url`, `directory`, `hash`, `hash_type`.
URLs pin immutable revisions; hashes are the manifest SHA256 values. File sizes
remain in the manifest and the Note lists the aggregate available download size.
No unregistered `cnr_id` is included. Opening a graph does not download models.

Base W8 examples select explicit manifest filenames. The SFT compatibility
example uses `dit_checkpoint=auto` so the existing resolver accepts its documented
filename aliases or explicit local mapping; its public download metadata still
names the actual recommended manifest file. The public SFT fused checkpoint is not
claimed byte-identical to the private R4 benchmark checkpoint. Preserve actual
source hashes in any new numerical/performance claims.

The original-TE example lists all four OpenMOSS BF16 shards, index/config and
tokenizer assets. Keep the prescribed layout:

```text
models/text_encoders/MOSS-Image-Base/text_encoder/
models/text_encoders/MOSS-Image-Base/tokenizer/
```

Its TE checkpoint widget remains `auto` because the current filename dropdown
does not list directories; the asset resolver recognizes that exact local
`text_encoder/` path. Missing any required local asset must be resolved before
queueing. Original BF16 is dense original-weight compute; it is not a renamed
INT8 dequantization.

The FP8 row model is **local-only / publish pending** in the supplied manifest.
The template names `moss_image_base_kohaku_fp8_row.safetensors`, which must be
exported locally and selected after it exists. Its Note explicitly states this
prerequisite; no FP8 download URL, revision or hash has been invented. Download
metadata in that graph covers only its public TE and VAE. It is a prepared
integration example, not a claim of out-of-the-box downloadable FP8 weights.

## Precision labels

The historical `INT8 ConvRot` TE widget retains its existing W8A16 numerical
path. The new `INT8 ConvRot · W8A8` example requests true dynamic activation-INT8
GEMM after rotation. This is an experimental numerical option with possible
quality differences, not an accuracy or losslessness guarantee. It requires the
supported CUDA hardware/build and real runtime qualification. Do not rename an
old workflow to imply that its previous execution already used A8.

## Validation performed for these artifacts

All five UI/API pairs were parsed and checked against the actual current custom
node `INPUT_TYPES` definitions: required inputs, enum values, ordering and
persistent widget counts, including the sampler seed-control widget. Session
has 20 persistent widgets; Sampler has 8. All 13 UI links per graph match source
and destination slots/types and reciprocal link lists. API references resolve,
UI/API scalar settings match, node rectangles do not overlap, and every model
metadata entry matches the manifest URL/hash. All graphs contain 9 UI nodes and
8 executable API nodes (the frontend Note is omitted from API prompts).

The static checks were followed by actual ComfyUI HTTP execution on the validated
Windows SM89 environment: all five templates completed their normal step counts
(SFT 16, Base 50), including original BF16 TE, real W8A8 TE and Base FP8 row.
PreviewImage and SaveImage produced ten valid PNGs. The five distributed SaveImage
PNGs retain workflow and prompt metadata and were visually inspected for blank
output and obvious tile seams. A separate public TE v2 request also completed.

See `../RELEASE_VALIDATION.md` and `validation/results.json` for exact scope and
receipts. These are execution and limited visual checks, not browser interaction
automation, a throughput benchmark, or a general image-quality certification.
