# Environment support and validation boundaries

**MOSS LowVRAM Optimized** uses KohakuMoss inference components. The intended
Registry package is `kohaku-moss-lowvram-optimized`. Hardware generations are
reported separately; support for a dtype or import does not certify every kernel,
checkpoint, profile, operating system or optional wheel.

## NVIDIA generations

| Family | Evidence | Current publication wording |
|---|---|---|
| RTX 20 / Turing, SM75 | Historical RTX 2080 Ti with 22 GB physical memory (as reported), Windows, Python 3.12, Torch 2.10/cu130, triton-windows 3.2 post21; W5 4g/6g/unlimited auto end-to-end runs passed | Historical compatibility evidence only. The current release has not been rerun on a physical Turing card. |
| RTX 40 / Ada, SM89 | Current full release validation on RTX 4070 Ti SUPER, 16 GB physical VRAM, Windows/Python 3.12.13, Torch 2.10.0+cu130, comfy-kitchen 0.2.37 | Primary currently validated environment; see exact cases in `../RELEASE_VALIDATION.md`. |
| RTX 50 / Blackwell | No completed hardware qualification supplied | Pending. Do not extrapolate from SM89 or from an `SM>=89` condition in code. |
| Linux | No completed release validation supplied | Untested. A compatible Python/Torch/Triton stack is a prerequisite, not a test result. |
| AMD, Apple Metal, Intel accelerators | No supported end-to-end GPU path established by this release evidence | Not claimed supported by this release. |

The historical 22 GB Turing card is not a test of a stock 11 GB RTX 2080 Ti.
Likewise, 4g/6g/8g profiles in current Ada testing are process allocation budgets
on a 16 GB physical card, not physical small-card certifications. Async reserved
pool memory, driver/context memory and allocations outside the tracked budget
remain separate from the profile allocation cap.

## Sanitized historical SM75 evidence

The maintainer's earlier receipts record an RTX 2080 Ti SM75 device with **22 GB
physical memory**. The tested stack was Windows, Python 3.12, Torch 2.10 with a
CUDA 13.0 build and triton-windows 3.2 post21. W5 with auto attention completed
end-to-end under the software profiles labelled **4g, 6g and unlimited**.

The 4g/6g labels describe allocation policies applied on that 22 GB device. They
do not establish successful execution on physical 4 GB/6 GB cards, nor on an
unmodified 11 GB 2080 Ti. No historical timing, exact peak or quality score is
invented in this summary. The private raw receipt locations and machine-specific
paths are intentionally omitted; these facts are a sanitized maintainer summary.

Those receipts predate the current publishing workflow edits, directory selector,
exact public model choices and new W8A8/FP8 integrations. They are not current
release requalification. The new 06 W5 and 07 W6 workflow files now have their own successful current
SM89 HTTP executions, recorded in `PUBLICATION_VALIDATION.md`; that does not
constitute a new SM75 run. In particular, a historical W5 PASS is not a W6,
FP8, A8-TE or all-profile PASS. Current full release evidence remains the separate
SM89/RTX 4070 Ti SUPER ledger in `../RELEASE_VALIDATION.md`.

## Mode boundaries

- Turing's historical evidence applies to the stated W5 auto path. Native FP8
  compute and the newly introduced activation-INT8 TE path must not be advertised
  as established Turing features.
- Ada covers the combinations actually listed in the release evidence. FP8 storage
  does not prove every GEMM uses FP8 operands: upstream may choose a w16 fallback.
- Original BF16/FP16 TE, W8A16 TE, W8A8 TE and W4A8 TE are different numerical
  contracts. Preserve their separate receipts and FP16 conditioning boundary.
- Optional attention backends are build-specific. The ordinary `auto` path uses
  PyTorch SDPA and does not require a private Flash or Sage binary.
- The historical locally built Turing bundle has source without a LICENSE. It
  is **not distributed**, linked as a public download, or installed automatically.
  Do not infer distribution permission from local successful execution.

## Install-time dependencies

The current package requires **Python >=3.12**. Use the Python environment that
starts ComfyUI. Normal Python requirements belong
in the package's declared dependencies and `requirements.txt`; keep them in sync.
They currently include `accelerate`, transformers, safetensors, einops, NumPy,
Pillow and comfy-kitchen. The actual release stack uses Python 3.12; a lower-bound
declaration does not qualify every newer Python or dependency version.
Use a platform-appropriate Triton stack compatible with the existing Torch/CUDA
and Python ABI. Do not replace users' Torch/CUDA merely to install this node pack.

```sh
python -m pip install -r requirements.txt
python tools/check_environment.py
```

Manager can install requirements and run a root `install.py` at installation
time. A manual directory copy does not necessarily run those steps. The absence
of an optional dependency should produce a clear feature-specific diagnostic.
Node imports and queued execution must not invoke pip, install compilers or
download optional binaries. This follows the official
[Registry security standards](https://docs.comfy.org/registry/standards) and
[Manager installation lifecycle](https://docs.comfy.org/custom-nodes/backend/manager).

Any future precompiled binary must identify OS, architecture, Python ABI, Torch,
CUDA build, source revision, build procedure, SHA256 and license. Prefer existing
upstream artifacts when available. A wheel is not a model file and does not belong
in workflow `properties.models`.

## Declared compatibility versus observed compatibility

Registry metadata can describe Python, ComfyUI, frontend, OS and accelerator
requirements. Those fields are install guidance, not proof that a specific GPU
was tested. Only mark platforms as supported to the extent established by the
release's runtime checks; avoid an unconditional OS Independent declaration.
Official field reference:
[Registry pyproject specification](https://docs.comfy.org/registry/specifications).

The current evidence establishes execution and limited visual checks, not
universal speed improvements, lossless image quality or unattended installation
on every host. Additional Turing/current-release, Blackwell and Linux runs must
be recorded before expanding those claims.
