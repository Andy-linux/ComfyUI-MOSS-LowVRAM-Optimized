# TE precision modes and integration contract

## Existing behavior established from the bundled source

`paired_session.py` attaches `layer_base` as `QuantWeight` / `W4A8Weight`.
`kohakumoss/quant/weights.py::QuantWeight.bind` resolves `quant_linear_rotated`.
`kohakumoss_kernels/backends/triton/quant_rotated.py` rotates the floating input,
and its separate path uses `backends/triton/quant.py::_fwd_kernel`. That kernel
does `tl.dot(a, w.to(a.dtype), acc)` with FP32 accumulation and applies row scale
after the dot. It does not quantize a to INT8. The fused rotated kernel likewise
casts the rotated input to its floating dtype. TileLang and eager entries are
weight-only too. Thus the legacy INT8 ConvRot TE is **W8A16**, currently FP16.
Legacy W4A8 checkpoint codes decode onto the INT8 grid and enter that same A16
operator. Its name describes checkpoint format, not the actual GEMM activation.

Old UI labels retain their original dispatch. The paired text session defaults
to `te_mode="auto", compute_dtype="fp16"`, preserving existing numerical behavior.

## Session and node integration

The node maps each TE selection to an asset, mode and compute dtype. The outer
session configuration includes `te_mode` and `te_compute_dtype` in its identity,
and passes them as `te_mode` and `compute_dtype` to the paired text session.
Changing modes therefore selects a distinct TE session and feature-cache identity.
Values are case-sensitive.

| mode | input | numerical behavior |
| --- | --- | --- |
| auto | existing Comfy file | old behavior: stored codes, A16; output FP16 |
| w4a8 | W4A8 file (mixed INT8 allowed) | decode original W4 codes to original INT8 grid, dynamic per-token A8 after rotation, integer dot |
| w8a8 | INT8 file | original INT8 codes/scales/rotation; dynamic per-token A8 after rotation, integer dot |
| w8a16 | INT8 file | original codes/scales/rotation through existing upstream weight-only operator |
| original | original floating single file, HF TE directory, or OpenMOSS pipeline root | dense original BF16/FP16 values, cast to requested compute dtype, no quantization |

`compute_dtype` is `fp16` or `bf16`; default remains `fp16`. Output conditioning
remains **CPU FP16** for compatibility, independently of weight/compute dtype.
Original BF16 source + FP16 compute is explicitly a cast, not original BF16
compute. INT8 dequantized to floating is refused as an original source.
All modes retain ring/resident placement (ring default, host=file, pinned=False).
HF loading uses existing `hf_checkpoint`/`load_placed_text`, only text_config,
skips visual/MTP/LM head through `hf_text_files`, and requires local tokenizer
assets at pipeline `tokenizer/` or the TE directory. No downloads occur.

## Explicit A8 definition and boundaries

The new A8 path is session-local; no registry edits or DiT behavior changes.
QuantWeight/W4A8Weight subtypes preserve upstream ring rebind and decode-once.
The codes/scale buffers are never requantized. Embeddings keep their original
stored format and inverse-rotation lookup; A8 applies only to quantized linears.
Dense nonquantized linears remain dense, as in mixed-precision checkpoints.

For each token row, rotate x by the upstream regular Hadamard when required;
sx=max(max(abs(x_rot)),1e-12)/127; qa=round(x_rot/sx), clamp [-127,127].
Then use torch._int_mm(qa,q.T), obtaining INT32; convert to FP32, multiply by
sx and original per-output-row weight scales, add bias, cast to incoming dtype.
Zero-pad M/N/K to multiples of 32 and remove padding. No floating fallback.
W4 correction terms are refused rather than discarded. CUDA A8 currently requires
SM80+ explicitly; SM75 is refused. torch._int_mm is an internal torch API and its
availability/shape support is a GPU validation gate. This is a new quality/cost
option, not a quality-preserving guarantee. The first implementation favors
correct, inspectable dispatch over fused-kernel performance.

## Receipts and validation

`load_receipt.te_precision` and each request receipt distinguish stored format,
GEMM, activation bits/input dtype, output dtype, requested mode and attached A8
base count. `a8_linear_calls` counts successfully executed INT8 dots for this
request; legacy/A16 reports zero.

The final Windows SM89 validation ran all 41 component tests without skips.
The TE numerical CPU checks include an independent integer reference, nonaligned
padding, rotation and zero-input behavior. Real W8A8 and W4A8 requests each
reported 200 attached A8 bases and 200 successful INT8 linear calls. Original
BF16 and Original FP16 both completed real encoding and image execution, with
CPU FP16 conditioning at the public output boundary. The normal 50-step Base
workflow also passed with original BF16 TE and with explicit W8A8 TE. A separate
request verified the exact public INT8 ConvRot v2 file used by the model manifest.

These results establish the recorded execution paths on the tested build. They
do not establish equality with a separate HF reference implementation, unchanged
image quality, complete cache/error-path coverage, or support for every GPU.
Broader quality evaluation should compare fixed seeds/prompts, including Chinese,
layout and long text, against original BF16 using identical template/tokens.
Full case scope is in `../RELEASE_VALIDATION.md` and `validation/results.json`.

## Dependencies and licensing

No additional packages, models, weight conversions, retraining, upstream source
modifications or global registry changes. Uses existing torch and bundled
Kohaku APIs under their existing distribution notices. Loading original weights
adds their existing source/license provenance to the user's weight selection;
it does not grant redistribution rights beyond the model's own license. No
license metadata changes or new external source copying are introduced.
