# Changelog

## 0.1.0 — initial release candidate

- Provide MOSS SFT W5/W6/W8, Base INT8 ConvRot W8A8, and Kohaku FP8 inference
  through stable ComfyUI node IDs under `MOSS/Optimized`.
- Add explicit W8A16, genuine W8A8/W4A8, and original BF16/FP16 text encoder
  modes. The historical `INT8 ConvRot` selection remains compatible.
- Prefer ComfyUI's `cudaMallocAsync` mechanism with a startup probe and a fresh
  native-allocator process when that mechanism is unsupported. Integrate model
  admission, VRAM budgets, transfer scheduling, and unload behavior.
- Provide only `00_moss_lowvram.json` as the user configuration/preparation
  workflow. Keep the seven earlier precision examples solely as compatibility
  fixtures in `tests/historical_workflows/`, excluded from the Registry package.
  Separate stored weight formats, actual arithmetic, and floating boundary
  precision; preserve all six earlier internal node IDs.
- Prepare selected models in the ComfyUI server's registered model directories
  through an explicit GUI confirmation, with pinned versions, size/SHA256 checks,
  progress, retries, verified-file reuse, and complete nine-file Original TE sets.
- Permit reversible isolation of a confirmed corrupt same-name model only after
  the GUI has shown that repair plan and the user confirms it.
- Share each Base/SFT row FP8 checkpoint between FP8 and 16-bit activation modes,
  without requantizing its stored weights; separate session/cache identities.
- Prepare supported Triton dependencies in the Manager installation lifecycle,
  including the Windows restart stage, without runtime pip or replacing Torch.
- Add environment diagnostics at installation, in the model node, and before
  prompt execution. Preserve the host Torch/CUDA installation.
- Preserve upstream licenses, component provenance, and modification records.

Publication status and measured hardware scope are recorded in
`RELEASE_CHECKLIST.md` and `docs/ENVIRONMENT_SUPPORT.md`. A release candidate is
not a Registry publication. Public download links are enabled only after the
actual model upload and immutable file metadata have been verified.
