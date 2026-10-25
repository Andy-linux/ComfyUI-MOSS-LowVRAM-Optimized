"""Session-owned packed decoder scratch; no global module/class monkeypatch.

Pass the lossless packed-format Triton kernel explicitly at construction.
The runtime canonical FrozenPackedBase holds this owner and calls owner.unpack;
FFN/QKV helpers must receive the same owner or an explicitly reserved tensor.
Module import and constructor do not import torch or initialize CUDA.
"""

from contextlib import contextmanager
from threading import Lock, get_ident


def unpack_into(kernel, packed, bits, k, out, *, bn=64, bk=128, num_warps=4):
    """Tensor-only launch body for an outer validated owner/custom_op.

    out is [N,K] INT8, nonaliasing packed. No allocation, global patch, locks,
    counters, stream query or hardware query. Caller owns its lifetime/serialization.
    """
    n = packed.shape[1]
    kernel[((n + bn - 1) // bn, (k + bk - 1) // bk)](
        packed, out, n, k, bits, bn, bk, num_warps=num_warps, num_stages=1
    )
    return out


class SessionScratch:
    """One model/session, one active host thread and CUDA stream at a time.

    execution() is required around sampler/whole backbone calls. Its lifetime
    covers every launch using the shared buffer; resizing is forbidden inside
    execution. reserve() the maximum N*K before binding/compiling any block.
    close() drops this owner's buffer references. Compiled graphs/external tensor
    references are separately owned and must be destroyed by their caller.
    """

    def __init__(self, *, decoder_kernel, device, maximum_elements=None):
        self.kernel = decoder_kernel
        self.device = str(device)
        if not self.device.startswith("cuda:"):
            raise ValueError("Explicit cuda:N device required")
        int(self.device.split(":")[1])
        self.maximum_elements = maximum_elements
        self._buffer = None
        self._lock = Lock()
        self.active = False
        self.closed = False
        self._thread = None
        self._stream = None
        self._bound_stream = None
        self.stats = dict(
            executions=0,
            unpack_calls=0,
            decoded_bytes_written=0,
            allocations=0,
            reserved_bytes=0,
        )
        self.receipt = dict(
            ownership="session-local private INT8 decoder storage",
            global_module_patch=False,
            global_class_patch=False,
            original_packing_and_scale_semantics=True,
            required_scope="explicit execution around whole sampler/backbone",
            code_qualification="independent packed-format decoder; parity recorded in release validation",
        )

    @property
    def buffer(self):
        if self.closed or self._buffer is None:
            raise RuntimeError("Scratch is closed or has not been reserved")
        return self._buffer

    def reserve(self, elements):
        import torch

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot reserve/resize active scratch")
        try:
            if self.closed:
                raise RuntimeError("Scratch owner closed")
            if not isinstance(elements, int) or elements <= 0:
                raise ValueError("Positive integer scratch size required")
            if self.maximum_elements is not None and elements > self.maximum_elements:
                raise MemoryError("Scratch exceeds explicit session byte budget")
            if self._buffer is None or self._buffer.numel() < elements:
                self._buffer = torch.empty(
                    elements, device=self.device, dtype=torch.int8
                )
                self.stats["allocations"] += 1
                self.stats["reserved_bytes"] = self._buffer.numel()
            return self._buffer
        finally:
            self._lock.release()

    def _validate_scope(self):
        import torch

        if self.closed or not self.active:
            raise RuntimeError("Packed decoder requires active session execution")
        if self._thread != get_ident():
            raise RuntimeError("Packed decoder used by another host thread")
        if self._stream != torch.cuda.current_stream(self.device).cuda_stream:
            raise RuntimeError("Packed decoder CUDA stream changed")

    @contextmanager
    def execution(self):
        import torch

        if self.closed:
            raise RuntimeError("Scratch owner closed")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent/nested session scratch execution refused")
        try:
            if self._buffer is None:
                raise RuntimeError("Reserve maximum scratch before execution")
            stream = torch.cuda.current_stream(self.device).cuda_stream
            if self._bound_stream is not None and stream != self._bound_stream:
                raise RuntimeError(
                    "Scratch lifetime is bound to one CUDA stream; create a separate owner"
                )
            self._bound_stream = stream
            self._thread = get_ident()
            self._stream = stream
            self.active = True
            self.stats["executions"] += 1
            yield self
        finally:
            self.active = False
            self._thread = self._stream = None
            self._lock.release()

    def unpack(self, packed, bits, k, *, bn=64, bk=128, num_warps=4):
        """Drop-in original decoder signature; output borrowed until next unpack.

        Every consumer must enqueue its GEMM before another projection calls
        unpack. Same stream order protects prior reads; no per-call GPU sync.
        """
        import torch

        self._validate_scope()
        if (
            packed.device != torch.device(self.device)
            or packed.dtype != torch.uint8
            or not packed.is_contiguous()
        ):
            raise ValueError(
                "Same-device original contiguous UINT8 packed tensor required"
            )
        if (
            bits not in (5, 6)
            or packed.ndim != 2
            or packed.shape[0] != (k * bits + 7) // 8
        ):
            raise ValueError("Original W5/W6 packed dimensions required")
        n = packed.shape[1]
        size = n * k
        if size > self.buffer.numel():
            raise MemoryError(
                "Reserve maximum decoder size before binding; no resize inside execution"
            )
        out = self.buffer[:size].view(n, k)
        unpack_into(
            self.kernel, packed, bits, k, out, bn=bn, bk=bk, num_warps=num_warps
        )
        self.stats["unpack_calls"] += 1
        self.stats["decoded_bytes_written"] += size
        return out

    def tensor_view(self, n, k):
        """Bind tensor-only custom_op before compile; caller does not resize later."""
        if self.active:
            raise RuntimeError("Bind tensor scratch before entering execution")
        if n * k > self.buffer.numel():
            raise MemoryError("Reserved scratch too small")
        return self.buffer[: n * k].view(n, k)

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close active session scratch")
        try:
            self._buffer = None
            self.stats["reserved_bytes"] = 0
            self.closed = True
        finally:
            self._lock.release()
