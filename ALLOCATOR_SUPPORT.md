# Allocator selection and evidence

Start ComfyUI with its existing `--cuda-malloc` mechanism. The supplied launcher
runs a disposable probe before starting ComfyUI. It sets both
`PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_ALLOC_CONF` to the same backend, before
importing torch. A probe `UNSUPPORTED` result permits a native probe and a new
ComfyUI process with `--disable-cuda-malloc`. A failed correctness test does not
silently fall back, and an inference failure never automatically reruns a workflow.
No package installation, downloads, system Torch changes or runtime allocator swap.

```
python tools/allocator_probe.py --backend cudaMallocAsync
python tools/allocator_probe.py --backend native
python tools/launch_comfy_allocator.py --comfy-main /path/to/ComfyUI/main.py -- --listen 127.0.0.1
```

The probe returns one JSON result on stdout. It allocates only small bounded
buffers: a 4 MiB GPU stats case, an intentionally rejected 64 MiB allocation under
a 32 MiB fraction limit, a 1 KiB recovery, and a 4 MiB pinned H2D transfer. It holds
the pinned source until the copy-stream event completes and checks copied contents.
GPU empty-cache and host idle-cache are observed separately. It restores its
fraction even after a failed cap test. Run it in a separate process, never against
a live ComfyUI CUDA context.

## Observed Torch 2.10 compatibility issue

The operator's first actual Torch 2.10+cu130 async probe selected cudaMallocAsync:
4 MiB live allocation / 32 MiB pool, then 0 used / 32 MiB reserved after free, and
0 reserved after GPU empty-cache. Its initial fraction getter returned
1074299169.1500106 (the unset UINT64_MAX limit divided by device bytes). Restoring
that raw value via the setter failed because the setter accepts only [0,1]. Keep
that failed receipt. Version 2 records the raw getter and restores values >1 as
1.0, the legal full-device/unbounded fraction. Runtime resources uses the same
normalization; negative or non-finite values are errors. This is an operator
observation, not a claim that every Torch version behaves identically.

## Budget and measurement contract

Native retains its existing allocator-reservation cap interpretation. Async
fraction limits PyTorch requested bytes, while aggregate used/reserved stats come
from the CUDA pool. Pool reserved may exceed that fraction: never reject async
solely because reserved >= cap. Pool used can include other pool users; do not
claim the pool stats are an exact PyTorch requested-byte counter. The admission
helper subtracts pool used conservatively, labels its estimate and includes
physical-free plus estimated reusable pool backing. External mapped VAE weights
must be counted once. Neither fraction nor the estimate guarantees total process
or device physical peak. Keep a separate GPU observer.

Old native R4 qualification remains unchanged. Async results need a separate
backend-tagged series; they do not retroactively replace the native 60-cell matrix.
Do not invoke memory_snapshot under async. Native block/fragmentation/cumulative
statistics do not acquire meaning merely because the async API returns zeros.

GPU empty_cache does not release cached pinned host RAM. Preserve the existing
host-cache lifetime and bounded pinned-chunks fixes, including asynchronous-copy
ownership. Close-session host cleanup remains separate from GPU pool trimming.

## Session integration (owner edits session.py separately)

Import `admission` from `.allocator`. Replace the VAE fraction/raw arithmetic with:

```
plan = admission(torch, self.config.device,
                 include_physical=self.config.qualified_vae_budget or self.config.vae_admission)
allocator_available = plan['available_allocator_bytes']
available = plan['effective_available_bytes']
physical_free = plan['physical_free_bytes']
reusable_cache = plan['reusable_allocator_cache_bytes']
```

Once dynamic VAE external mapped bytes are known, recompute the same helper with
`external_bytes=external_mapped_bytes`; use its effective available value and add
the entire backend-tagged plan to `last['VAE_memory_plan']['allocator']`. Preserve
native legacy arithmetic if strict bit-for-bit admission-decision compatibility
is required; apply the helper only to async in that case. Do not subtract external
mapped bytes twice. Preserve the existing model-size/target-size qualifications.

The existing `_dispose_dit`/unload call to `torch.cuda.empty_cache()` can stay: it
trims the selected GPU allocator and can synchronize, so retain current lifecycle
boundaries. It does not select native. Session close must still call
`release_idle_host_cache()` after owned references and in-flight copies are safe.
Any `vae_stage_budget.py` temporary fraction restoration also needs
`restorable_fraction(raw)` before passing its saved value to the setter; this
file is intentionally left for the session owner to integrate.

CPU checks:
`python -m unittest discover -s tests -p test_allocator_support.py -v`.
These verify policy and arithmetic, not GPU compatibility or performance.

A real standalone grad-enabled FP8 dispatch probe (outside allocator_budget) also
exposed the unset async fraction sentinel in vendor hardware.memory.working().
That helper now normalizes finite raw fractions above 1 to 1.0, rejects negative
or non-finite values, bases async bounds on allocated rather than reserved bytes,
never raises an enclosing cap, and closes its measurement window even on setter/
restore errors. Without a fraction getter it measures only, preserving an unknown
outer cap instead of restoring it incorrectly to 1.0. Native/XPU retain a reserved
entry basis. Actual FP8 revalidation follows in the operator's GPU evidence.
