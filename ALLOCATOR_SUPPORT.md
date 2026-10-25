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

## Integrated session behavior

`runtime/session.py` uses `allocator.admission()` for the VAE memory plan and
records the backend, logical budget, physical free memory and reusable pool
backing in `last['VAE_memory_plan']['allocator']`. The helper applies the selected
backend's accounting rules. When DynamicVRAM exposes externally mapped VAE
weights, the session recomputes admission with `external_bytes` so those weights
count once toward the logical budget.

`runtime/resources.py` and `runtime/vae_stage_budget.py` normalize the saved
fraction before restoring it. Existing model-size, shape and hardware guards,
offload decisions and bounded VAE-stage admission remain in place. The stage
budget is an admission policy, not a measured physical-peak guarantee.

`_dispose_dit` and unload keep their lifecycle-boundary `torch.cuda.empty_cache()`
calls. Those trim the selected GPU allocator; they do not select native. Session
close releases owned references and in-flight work before the separate
`release_idle_host_cache()` path. The installed ComfyUI service completed an
explicit unload under cudaMallocAsync, with the host allocator's current
allocation count and allocated bytes both reported as zero. This does not claim
that every process allocation or every cached statistic became zero.

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
entry basis. The repeated real-weight FP8 probe passed all six observed cases;
its scope and the normal workflow evidence are recorded in RELEASE_VALIDATION.md.
