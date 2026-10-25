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

Do not change old UI labels' dispatch. New SessionConfig fields default to
`te_mode="auto", compute_dtype="fp16"`, retaining existing numerical behavior.

## Session/node integration (owned by caller)

Pass `SessionConfig(..., te_mode=mode, compute_dtype=dtype)` at session creation.
Add BOTH fields to the outer session/config identity so switching modes replaces
the TE and cannot reuse features from another mode. Values are case-sensitive.

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

## Receipts and required validation

load_receipt.te_precision and each request receipt distinguish stored format,
GEMM, activation bits/input dtype, output dtype, requested mode and attached A8
base count. `a8_linear_calls` counts successfully executed INT8 dots for this
request, proving the new path was exercised (legacy/A16 should report zero).

Before enabling in ordinary UI:

1. Run `python -m unittest discover -s tests -p test_te_precision.py -v` with the
   plugin's bundled Kohaku dependency environment. CPU checks include independent
   integer reference, nonaligned M/N padding, rotation and zero-input behavior.
2. On SM89, compare a8_linear with independent INT32 reference for FP16/BF16,
   short M including 1/2 and representative TE K/N. Test no bias and zero rows.
3. Load W8A16 and legacy auto from same INT8 file: verify q/scale bytes and
   rotation equal; same dtype/layout should retain old numerical path.
4. Run real W8A8/W4A8 ring requests twice, packed and single, cache miss/hit,
   record positive a8_linear_calls, output shape/dtype, elapsed seconds, VRAM and
   errors. Confirm original memory/ring release works after errors too.
5. Load local HF original BF16 shards with ring, verify no visual tensors held,
   and compare features to the same-source HF baseline. Separately test BF16
   versus FP16 compute. Do not compare outputs without matching template/tokens.
6. Compare final images with fixed seeds/prompts including Chinese, layout and
   long text against original BF16. A8 changes quantization error and is optional.

Local review environment has no torch: syntax/import compatibility checks pass;
three numerical CPU tests are skipped and MUST be run by the GPU/runtime owner.
No GPU timing or quality claim has been made here.

## Dependencies and licensing

No additional packages, models, weight conversions, retraining, upstream source
modifications or global registry changes. Uses existing torch and bundled
Kohaku APIs under their existing distribution notices. Loading original weights
adds their existing source/license provenance to the user's weight selection;
it does not grant redistribution rights beyond the model's own license. No
license metadata changes or new external source copying are introduced.
