# MOSS LowVRAM Optimized — component notices

使用KohakuMoss推理组件 · Formal package: `kohaku-moss-lowvram-optimized`

The root `LICENSE` is byte-identical to `licenses/KohakuMoss-LICENSE.txt`:
**KohakuMoss License 1.0, October 2026**, including Section 10 naming terms.
The original integration is prepared for release under that agreement. This does
not replace the distinct licenses of other components or create a new copyright
owner. The formal distributed package name **kohaku-moss-lowvram-optimized**
includes Kohaku. Main display is **MOSS LowVRAM Optimized**; the subtitle
provides component attribution, not an asserted exception to Section 10. The
existing repository URL is <https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized>;
internal node IDs remain unchanged.

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
package. W5/W6 already exist at `Andylinux/MOSS-Image-SFT-W5A8-W6A8`,
revision `bfb77cb73ae7acb4f788d50d4043c2d71989b0a7`; the operator verified
their hashes against the local files. Index these files rather than re-uploading.
Two Base/SFT FP8 row checkpoints target the public repository
`Andylinux/MOSS-Image-Kohaku-FP8`; file upload/verification is still in progress.
Each row file supports FP8 or 16-bit compute. The older separate Base weight-only
export remains legacy local and is not a third public upload. Checkpoint conversion
is inference preparation, not training. Integration modifications are summarized
in the root `NOTICE`.

## Optional Turing bundle — separately authorized distribution

The user explicitly authorized distributing the self-built SM75 bundle as a
separate GitHub Release. The source/Registry code archive still excludes all
Turing binaries. Manager's installation phase may prepare only the exact fixed
public asset described by `optional_binary_manifest.json`; actual ZIP publication
and URL/hash verification remain pending, not implied by this notice.

The pinned `ssiu/flash-attention-turing` source commit is
`9ef98fcb506bb1e2fe3cece50935e2935bf6b124`, with CUTLASS submodule
`df18f5e4f5de76bed8be1de8e4c245f2f5ec3020`. Available build records report
no source patch. No LICENSE or repository license declaration was confirmed at
the pinned Turing HEAD. Actual CUTLASS/NVIDIA terms cover their respective
components; they are not asserted to license the entire Turing implementation
or combined binary. User authorization does not change that upstream fact or
create a blanket open-source license.

The separate release is to preserve actual source archives, build records,
portable binary/source hashes, original component license texts and notices.
Private machine paths, GPU identifiers and benchmark caches are not publication
payloads. Installation is restricted to Windows x64 / CPython3.12 /
Torch2.10.0+cu130 / SM75 and preserves default auto if optional preparation fails.
The adapter's historical dense FP16 evidence is not new full-workflow validation.
See `docs/DISTRIBUTION_INVENTORY.md` for artifact decisions.
