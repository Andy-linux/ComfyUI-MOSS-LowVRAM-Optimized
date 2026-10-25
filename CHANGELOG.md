# Changelog

## 0.1.0 — initial release candidate

- Provide MOSS SFT W5/W6/W8, Base INT8 ConvRot W8A8, and Kohaku FP8 inference
  through stable ComfyUI node IDs under `MOSS/Optimized`.
- Add explicit W8A16, genuine W8A8/W4A8, and original BF16/FP16 text encoder
  modes. The historical `INT8 ConvRot` selection remains compatible.
- Prefer ComfyUI's `cudaMallocAsync` mechanism with a startup probe and a fresh
  native-allocator process when that mechanism is unsupported. Integrate model
  admission, VRAM budgets, transfer scheduling, and unload behavior.
- Include seven portable UI workflows, pinned model references, and an explicit
  size/SHA256-verified downloader for complete model assets.
- Add environment diagnostics at installation, in the model node, and before
  prompt execution. Preserve the host Torch/CUDA installation.
- Preserve upstream licenses, component provenance, and modification records.

Publication status and measured hardware scope are recorded in
`RELEASE_CHECKLIST.md` and `docs/ENVIRONMENT_SUPPORT.md`. A release candidate is
not a Registry publication; optional FP8 downloads remain unavailable until an
actual model repository and immutable file metadata have been recorded.
