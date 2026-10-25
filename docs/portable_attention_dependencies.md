# Portable optional attention dependencies

Run the read-only inventory using ComfyUI's Python:

```text
python tools/check_environment.py
```

It prints JSON only, installs nothing, downloads nothing and launches no GPU
workload. It reports Python/torch/CUDA/GPU, Triton, comfy-kitchen and optional
attention package versions/origins, duplicate import candidates and private/public
shadow risks. Optional binary extensions are not imported. The Flash availability
probe establishes compiled support only, not a successful tensor-layout test.

## Flash

The existing private `runtime/vendor/xformers034cu128` Windows binary is preferred
when present. Its existing explicit xformers Flash FwOp and DLL handle ownership
remain intact. Loading a conflicting xformers beforehand remains a clear error.
Version receipts read the imported module; the old directory-name CUDA hint is
identified as a hint, not a runtime measurement.

Without that private binary, explicit Flash uses public CUDA PyTorch through
`sdpa_kernel(SDPBackend.FLASH_ATTENTION)` exclusively. This disables ordinary
math/efficient/cuDNN attention substitution within that call. No extra xformers
installation is required for this route. Missing Flash build support or failure
for the requested dtype/layout raises an actionable error: use a compatible CUDA
PyTorch build, restore the optional dependency, or select auto.

The torch public route is a new candidate, not covered by the private binary's
previous validation. Both routes retain the SM89 hardware restriction, THD D128,
equal FP16/BF16 Q/K/V, one complete dense segment and caller-owned CFG scheduling.
Other GPUs are not claimed qualified. Test the actual public build with real
image and long-sequence cases before making quality/performance claims.

## SageAttention

The private `runtime/vendor/sageattn220/sageattention` installation has priority
when present. Otherwise the normally installed `sageattention` package is used.
The selected package must expose the requested explicit FP16-PV or FP8-PV CUDA
function. Missing package, incompatible extension or missing function gives an
actionable error; auto is not silently replaced by Sage, and an explicit Sage
request is not silently replaced by another attention backend.

Receipts report the actual module path, dependency source, callable interface,
torch/CUDA runtime and discovered version. Private module/distribution metadata
is used for a private package; a separately installed public package's metadata
is never presented as the private binary's version. If a build contains no
version metadata, its version is reported unknown, not hardcoded to post6.

The existing Q/K and optional P/V quantization choices, rotation and head-group
controls are unchanged. Numerics and speed can differ between Sage builds;
only the actual SM89 build tested on the host can receive qualification.

## Distribution and licensing

No private DLLs or wheels have been added, copied, downloaded or redistributed.
The plugin can use the user's compatible public torch/Sage installations. No
new package dependency is mandatory for auto, whose existing SDPA behavior is
unchanged. Binary/package licenses remain with those installations; this change
does not alter the bundled dependency notices or grant binary redistribution.
