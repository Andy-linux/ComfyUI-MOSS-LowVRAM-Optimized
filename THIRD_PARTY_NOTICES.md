# Kohaku MOSS Optimized — component notices

The root `LICENSE` is byte-identical to `licenses/KohakuMoss-LICENSE.txt`:
**KohakuMoss License 1.0, October 2026**, including Section 10 naming terms.
The original integration is prepared for release under that agreement. This does
not replace the distinct licenses of other components or create a new copyright
owner. The primary product name is **Kohaku MOSS Optimized**.

| Current component | Scope | Preserved terms |
|---|---|---|
| KohakuMossTrainer `5223637706bc3b847b27589ad18e9221226ffa93` | `runtime/vendor/kohaku_src`, covered upstream interfaces and integration | KohakuMoss License 1.0; `licenses/KohakuMoss-LICENSE.txt`. Upstream attribution: Copyright 2026 Shih-Ying Yeh (KohakuBlueLeaf) and contributors. |
| OpenMOSS/MOSS-Image `0ee86ac9576a6b3f1ada73e0e02aab5c710a58da` | `runtime/vendor/official_moss` loader source | Original Apache-2.0 text in `licenses/MOSS-Image-LICENSE.txt`. No copyright owner inferred from the generic appendix. |
| Qwen3.5/MOSS tokenizer and configuration | Packaged `encoders/text/assets/qwen3.5-9b` | Original local `LICENSE` and `NOTICE` retained. Tokenizer is gzip compressed; configuration and template attribution remain with these assets. |
| comfy-kitchen 0.2.37 | External runtime dependency; no binary bundled | `licenses/ComfyKitchen-LICENSE.txt` and unchanged `ComfyKitchen-NOTICE.txt`, including upstream NVIDIA/PyTorch/nanobind/FlashAttention/torchao/SageAttention notices. |
| triton-windows 3.6.0.post26 | External Windows compiler/runtime; no toolchain bundled | MIT text and attribution in `licenses/Triton-Windows-LICENSE.txt`. Linux Triton is a separate platform prerequisite. |
| xFormers 0.0.34 Flash wheel | Optional external dependency; not source-package binaries | BSD-3-Clause text in `licenses/xFormers-LICENSE.txt`; preserve wheel source headers and dist-info notices when obtaining it separately. |
| SageAttention Windows post6 | Optional external isolated wheel, not bundled | Apache-2.0 text in `licenses/SageAttention-LICENSE.txt`; Copyright 2024 Jintao Zhang, Haofeng Huang. Pinned wheel SHA256 `1635283f5c01ec3cda58a784d0d7eabbcaffaf9511d1b263db4750e1ed7958bb`. |

Exact preserved-file byte counts and hashes are in `LICENSE_PROVENANCE.json`.
Existing upstream copyright/license headers remain authoritative for their files.
The component inventory has been checked; it is not a formal legal opinion or a
claim about uninspected transitive dependencies.

## Independent packed decoder and historical exclusion

`runtime/packed_decode.py` was independently implemented from the mathematical
little-endian B-bit two's-complement layout
`UINT8[ceil(K*B/8),N] -> INT8[N,K]`. The operator reports not reading the removed
GPL decoder implementation. SM89 black-box parity against the existing runner
checked 10 W5/W6 cases and **169,939,328 original integer codes**, all equal; the
operator retains `packed-decoder-parity-v5.log` as private benchmark evidence.
No scaling, quantization or training is performed by this decoder.

The former `runtime/vendor/unpack_sm75.py` is absent from this candidate. Its GPL
text is no longer included as a current component license. Historical private
baseline evidence remains private; public release must use clean release history
that does not include that baseline. Removed code has not been relicensed, and
this engineering process is not described as formal clean-room legal certification.

## Models and modifications

Model weights are external assets with their own upstream terms, pinned in
`model_manifest.json`. A model-card license does not license this entire source
package. W5/W6 and exported FP8 publication remains pending the owner's actual HF
repository and immutable hashes. Checkpoint conversion is inference preparation,
not training. Integration modifications are summarized in the root `NOTICE`.
