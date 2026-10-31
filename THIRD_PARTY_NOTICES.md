# Component notices

**MOSS LowVRAM Optimized** uses KohakuMoss inference components. Formal package: `kohaku-moss-lowvram-optimized`.
The root [LICENSE](LICENSE) is KohakuMoss License 1.0, including Section 10 naming terms. Preserve Kohaku branding and the distinct licenses below.

| Current component | Scope | Preserved terms |
|---|---|---|
| KohakuMossTrainer `5223637706bc3b847b27589ad18e9221226ffa93` | `runtime/vendor/kohaku_src`, covered upstream interfaces and integration | KohakuMoss License 1.0; `licenses/KohakuMoss-LICENSE.txt`. Upstream attribution: Copyright 2026 Shih-Ying Yeh (KohakuBlueLeaf) and contributors. |
| OpenMOSS/MOSS-Image `0ee86ac9576a6b3f1ada73e0e02aab5c710a58da` | `runtime/vendor/official_moss` loader source | Original Apache-2.0 text in `licenses/MOSS-Image-LICENSE.txt`. No copyright owner inferred from the generic appendix. |
| Qwen3.5/MOSS tokenizer and configuration | Packaged `encoders/text/assets/qwen3.5-9b` | Original local `LICENSE` and `NOTICE` retained. Tokenizer is gzip compressed; configuration and template attribution remain with these assets. |
| comfy-kitchen 0.2.37 | External runtime dependency; no binary bundled | `licenses/ComfyKitchen-LICENSE.txt` and unchanged `ComfyKitchen-NOTICE.txt`, including upstream NVIDIA/PyTorch/nanobind/FlashAttention/torchao/SageAttention notices. |
| triton-windows 3.6.0.post26 | External Windows compiler/runtime; no toolchain bundled | MIT text and attribution in `licenses/Triton-Windows-LICENSE.txt`. Linux Triton is a separate platform prerequisite. |
| xFormers 0.0.34 Flash wheel | Optional external dependency; not source-package binaries | BSD-3-Clause text in `licenses/xFormers-LICENSE.txt`; preserve wheel source headers and dist-info notices when obtaining it separately. |
| SageAttention Windows post6 | Optional external isolated wheel, not bundled | Apache-2.0 text in `licenses/SageAttention-LICENSE.txt`; Copyright 2024 Jintao Zhang, Haofeng Huang. Pinned wheel SHA256 `1635283f5c01ec3cda58a784d0d7eabbcaffaf9511d1b263db4750e1ed7958bb`. |


Exact preserved-file identities and source facts are recorded in [LICENSE_PROVENANCE.json](LICENSE_PROVENANCE.json).
Original component headers, LICENSE and NOTICE texts remain with their files. Integration modifications are recorded in the root [NOTICE](NOTICE).

## Packed decoder

`runtime/packed_decode.py` is independently implemented from the little-endian B-bit two's-complement layout
`UINT8[ceil(K*B/8),N] -> INT8[N,K]`. The operator reports implementation without reading the removed GPL decoder.
Removed historical code has not been relicensed.

## Model weights

Weights have their own upstream terms, recorded in [model_manifest.json](model_manifest.json).
W5/W6 are indexed from their existing public repository. The Base/SFT FP8 row derivatives use the sources and conversion records in
[conversion_manifest.json](docs/huggingface/conversion_manifest.json), with original model terms in[LICENSE](docs/huggingface/LICENSE).

## Optional Turing asset

The separately authorized [Turing Release](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/tag/turing-sm75-cp312-cu130-v1)
preserves its source archives, build records, portable manifest, component licenses and NOTICE.
Pinned `ssiu/flash-attention-turing` source: `9ef98fcb506bb1e2fe3cece50935e2935bf6b124`;
CUTLASS: `df18f5e4f5de76bed8be1de8e4c245f2f5ec3020`. Available build records report no source patch.
The pinned Turing repository has no confirmed LICENSE or repository license declaration. CUTLASS/NVIDIA terms apply to their own components;
the combined Turing implementation's upstream license status remains unconfirmed.
The fixed asset and Windows x64/Python3.12/Torch2.10+cu130/SM75 limits are recorded in [optional_binary_manifest.json](optional_binary_manifest.json).

## Windows development-file preparation

Python 3.12/3.13 development archives are obtained from the triton-windows maintainer's documented release.
Python 3.14 development files are extracted from the Python Software Foundation's official `python` NuGet package;
the package keeps its upstream Python license. Only include/libs are extracted into the selected embedded interpreter.
No Python runtime binary is redistributed in this node package. Artifact URLs and SHA256 identities are in `runtime/install_dependencies.py`.

Sage CUDA12 and CUDA13 builds are selected separately. Both upstream artifact identities and sizes are recorded in `runtime/sage_artifacts.py`;
their original Apache-2.0 terms are retained in the downloaded wheels.
