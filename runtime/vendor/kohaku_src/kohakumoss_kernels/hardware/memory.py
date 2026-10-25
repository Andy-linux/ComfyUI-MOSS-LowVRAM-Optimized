"""Device-memory bookkeeping (CUDA / XPU; no-ops on the host): returning free blocks to
the driver, peak windows (`window`, `open_window` / `close_window`) that nest without
losing an enclosing window's peak, a block's working memory under an optional bound
(`working`), and the device's free / used bytes (`mem_info`, `device_used`). See
docs/internals/kernel-dev.md#peak-windows.
"""

import gc
import math
from contextlib import contextmanager

import torch

from kohakumoss_kernels.hardware.current import device_module

# device module per device type, overriding torch's (tests)
MODULES: dict[str, object] = {}
_OPEN: dict[str, list["Window"]] = {}


def _accel(device: torch.device | str):
    mod = device_module(device)
    if mod is None:
        return None
    return MODULES.get(torch.device(device).type) or mod


def release(device: torch.device | str) -> None:
    """Synchronise `device` and return the allocator's free blocks to the driver."""
    mod = _accel(device)
    if mod is None:
        return
    gc.collect()
    mod.synchronize(device)
    mod.empty_cache()


def trim(device: torch.device | str) -> None:
    """Synchronise `device` and return the allocator's free blocks to the driver (no
    garbage collection)."""
    mod = _accel(device)
    if mod is None:
        return
    mod.synchronize(device)
    mod.empty_cache()


class Window:
    """Peak allocated / reserved bytes on `device` while open (`alloc`, `reserved`)."""

    def __init__(self, device: torch.device, alloc: int = 0, reserved: int = 0):
        self.device = device
        self.alloc = alloc
        self.reserved = reserved


STAT_KEYS = (
    "allocated_bytes.all.current",
    "reserved_bytes.all.current",
    "allocated_bytes.all.peak",
    "reserved_bytes.all.peak",
)


def _counters(mod, device: torch.device) -> tuple[int, int, int, int]:
    """(allocated, reserved, peak allocated, peak reserved) from one allocator read."""
    stats = getattr(mod, "memory_stats", None)
    if stats is None:
        return (
            mod.memory_allocated(device),
            mod.memory_reserved(device),
            mod.max_memory_allocated(device),
            mod.max_memory_reserved(device),
        )
    s = stats(device)
    return tuple(int(s.get(k, 0)) for k in STAT_KEYS)


def _fold(mod, device: torch.device) -> tuple[int, int]:
    """Fold the peaks into every open window of `device`, reset them; the bytes
    allocated / reserved now."""
    alloc, held, peak, peak_held = _counters(mod, device)
    for w in _OPEN.get(str(device), []):
        w.alloc = max(w.alloc, peak)
        w.reserved = max(w.reserved, peak_held)
    mod.reset_peak_memory_stats(device)
    return alloc, held


def fold(device: torch.device | str) -> None:
    """Fold the device's peak counters into every open window, then reset them."""
    mod = _accel(device)
    if mod is None:
        return
    _fold(mod, torch.device(device))


def open_window(device: torch.device | str) -> Window:
    """A new window on `device`, starting at the bytes allocated / reserved now."""
    device = torch.device(device)
    mod = _accel(device)
    if mod is None:
        return Window(device)
    w = Window(device, *_fold(mod, device))
    _OPEN.setdefault(str(device), []).append(w)
    return w


def restart_window(w: Window) -> tuple[int, int]:
    """`close_window(w)` then reopen it in place, with one allocator read: its
    (peak allocated, peak reserved) so far; it then starts at the bytes held now."""
    mod = _accel(w.device)
    if mod is None:
        return w.alloc, w.reserved
    now = _fold(mod, torch.device(w.device))
    out = (w.alloc, w.reserved)
    w.alloc, w.reserved = now
    return out


def close_window(w: Window) -> tuple[int, int]:
    """Close `w`; its (peak allocated, peak reserved) bytes."""
    fold(w.device)
    windows = _OPEN.get(str(w.device), [])
    if w in windows:
        windows.remove(w)
    return w.alloc, w.reserved


@contextmanager
def window(device: torch.device | str):
    """`with window(device) as w:` peak bytes in `w.alloc` / `w.reserved` after it."""
    w = open_window(device)
    try:
        yield w
    finally:
        close_window(w)


BUDGET_FRACTION = 0.5


def tuning_budget(device: torch.device | str, peak: int = 0) -> int | None:
    """Bytes a tuning call may add on `device` above the bytes allocated now: up to
    `peak` (the device peak of the run's own work at its planned sizes, e.g. the kernel
    prewarm's; 0: unknown) and at most BUDGET_FRACTION of the memory the device has
    free; None on the host. See docs/internals/tuning.md#working-memory-bound."""
    info = mem_info(device)
    if info is None:
        return None
    device = torch.device(device)
    room = int(BUDGET_FRACTION * info[0])
    allocated = _accel(device).memory_allocated(device)
    if peak > allocated:
        room = min(room, peak - allocated)
    return room


class Working:
    """Peak bytes allocated above the bytes allocated at the start (`peak`), and
    whether the allocator enforced the bound (`enforced`)."""

    def __init__(self):
        self.peak = 0
        self.enforced = False


@contextmanager
def working(device: torch.device | str, bound: int | None = None):
    """Measure additional live bytes, optionally tightening a restorable limit.

    Native/XPU use reserved bytes as the entry basis; CUDA async uses allocated
    bytes (pool reserved is not its requested-byte limit). An outer fraction is
    never increased. Without a getter the block only measures: restoring an
    unknown outer cap to 1.0 would incorrectly relax it.
    """
    w = Working()
    mod = _accel(device)
    if mod is None:
        yield w
        return
    device = torch.device(device)
    if device.index is None and hasattr(mod, "current_device"):
        device = torch.device(device.type, mod.current_device())
    win = open_window(device)
    start = 0
    setter = getattr(mod, "set_per_process_memory_fraction", None)
    getter = getattr(mod, "get_per_process_memory_fraction", None)
    before = None
    attempted = False
    try:
        start = mod.memory_allocated(device)
        if bound is not None:
            if not math.isfinite(float(bound)) or bound < 0:
                raise ValueError("Working memory bound must be finite and nonnegative")
            if setter is not None and getter is not None and hasattr(mod, "mem_get_info"):
                free, total = mod.mem_get_info(device)
                raw = float(getter(device))
                if not math.isfinite(raw) or raw < 0:
                    raise ValueError("Invalid allocator memory fraction")
                # Async's initially unset UINT64_MAX limit can report >1.
                before = min(1.0, raw)
                if bound < free:
                    if total <= 0:
                        raise ValueError("Invalid device memory size")
                    provider = getattr(mod, "memory", mod)
                    backend = getattr(provider, "get_allocator_backend", None)
                    if backend is None:
                        backend = getattr(mod, "get_allocator_backend", None)
                    async_backend = device.type == "cuda" and backend is not None and backend() == "cudaMallocAsync"
                    basis = start if async_backend else mod.memory_reserved(device)
                    target = min(before, 1.0, (basis + bound) / total)
                    attempted = True  # restore even if a setter partially changes then raises
                    setter(target, device)
                    w.enforced = True
        yield w
    finally:
        try:
            if attempted:
                setter(before, device)
        finally:
            try:
                w.peak = max(close_window(win)[0] - start, 0)
            finally:
                # close_window folds counters first; retain no stale window even
                # if that read/reset itself fails while another error propagates.
                windows = _OPEN.get(str(device), [])
                if win in windows:
                    windows.remove(win)


def allocated(device: torch.device | str) -> int:
    """Bytes of live tensors on `device` now."""
    mod = _accel(device)
    return 0 if mod is None else mod.memory_allocated(device)


def reserved(device: torch.device | str) -> int:
    """Bytes the allocator holds on `device` now."""
    mod = _accel(device)
    return 0 if mod is None else mod.memory_reserved(device)


def mem_info(device: torch.device | str) -> tuple[int, int] | None:
    """(free, total) bytes of the whole `device` from the driver; None on the host or
    without `mem_get_info`."""
    mod = _accel(device)
    if mod is None or not hasattr(mod, "mem_get_info"):
        return None
    free, total = mod.mem_get_info(device)
    return int(free), int(total)


def device_used(device: torch.device | str) -> int:
    """Bytes in use on the whole device (every process; what nvidia-smi shows)."""
    info = mem_info(device)
    return 0 if info is None else info[1] - info[0]
