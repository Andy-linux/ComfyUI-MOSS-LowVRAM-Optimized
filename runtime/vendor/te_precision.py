"""Session-local TE precision policy. Never modifies the global kernel registry.

The legacy Comfy W4A8 filename describes its storage, not this loader's GEMM:
upstream QuantWeight/W4A8Weight use weight-only A16. Explicit A8 modes below
quantize each rotated token row symmetrically and use integer matmul. They are
new numerical modes and require quality validation against original weights.
"""
import torch
from kohakumoss.quant.weights import QuantWeight
from kohakumoss.quant.w4a8 import W4A8Weight
from kohakumoss_kernels.hadamard import group_hadamard_eager

MODES = ("auto", "w4a8", "w8a8", "w8a16", "original")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def validate_mode(mode, dtype):
    if mode not in MODES:
        raise ValueError(f"TE mode {mode!r} not in {MODES}")
    if dtype not in DTYPES:
        raise ValueError("TE compute_dtype must be fp16 or bf16")


def a8_linear(x, q, scale, bias, rotation=0):
    """Dynamic symmetric per-token INT8 activations; INT32 dot, FP32 epilogue.

    Rotation is the upstream regular Hadamard, applied BEFORE quantization.
    Existing weight codes/scales are untouched. Padding is zero and is removed
    before returning. No floating fallback silently masquerades as INT8 GEMM.
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("A8 TE expects FP16/BF16 incoming activations")
    if q.dtype != torch.int8 or x.shape[-1] != q.shape[1]:
        raise ValueError("A8 TE requires compatible INT8 [N,K] weights")
    shape = x.shape[:-1]
    xr = group_hadamard_eager(x, rotation) if rotation else x
    xf = xr.reshape(-1, xr.shape[-1]).float()
    sx = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    a = (xf / sx).round().clamp(-127, 127).to(torch.int8)
    m, k = a.shape
    n = q.shape[0]
    # CUDA _int_mm alignment constraints vary by torch/cuBLAS version. Use
    # multiples of 32 for every axis, including short one-token requests.
    pm, pk, pn = (-m) % 32, (-k) % 32, (-n) % 32
    a = torch.nn.functional.pad(a, (0, pk, 0, pm)).contiguous()
    b = torch.nn.functional.pad(q, (0, pk, 0, pn)).t().contiguous()
    dot = torch._int_mm(a, b)[:m, :n]
    out = dot.float() * sx * scale.float().reshape(1, -1)
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype).reshape(*shape, n)


class TEA8Weight(QuantWeight):
    """QuantWeight subtype survives upstream ring rebind_quant isinstance checks."""
    def bind(self, dtype, device=None):
        if self.fmt != "int8":
            raise ValueError("A8 TE requires INT8 codes")
        self._linear = _a8_weight


def _a8_weight(w, x, bias):
    result = a8_linear(x, w.q, w.scale, bias, w.rotation)
    w._te_a8_calls = getattr(w, "_te_a8_calls", 0) + 1
    return result


class TEA8PackedWeight(W4A8Weight):
    def bind(self, dtype, device=None):
        if not self.decodes_to_int8:
            raise ValueError("A8 TE does not support W4 correction terms")
        # Keep the existing decoder on this instance (initially eager). Avoid
        # introducing a compiler dependency for this experimental A8 path.
        self._linear = _a8_packed

    def to_int8(self):
        if not self.decodes_to_int8:
            raise ValueError("A8 TE does not support W4 correction terms")
        return TEA8Weight(self.decode(), self.s_channel, "int8", self.rotation)


def _a8_packed(w, x, bias):
    result = a8_linear(x, w.codes(), w.s_channel, bias, w.rotation)
    w._te_a8_calls = getattr(w, "_te_a8_calls", 0) + 1
    return result


def configure_base(base, mode):
    """Adapt only newly-created bases owned by this TE session; preserve buffers."""
    if mode not in ("w4a8", "w8a8"):
        return base
    if isinstance(base, W4A8Weight):
        if not base.decodes_to_int8:
            raise ValueError("A8 TE does not support W4 correction terms")
        base.__class__ = TEA8PackedWeight
    elif isinstance(base, QuantWeight) and base.fmt == "int8":
        base.__class__ = TEA8Weight
    else:
        raise ValueError("A8 TE requires stored INT8 or W4A8 bases")
    return base
