"""Share unchanged fixed ConvRot256/A8 preparation inside ONE attention invocation.

Only original FrozenPackedBase q/k/v are supported; each projection still unpacks its
own exact codes, invokes original INT32 GEMM and native dequantization. No global
quantizer patch, no concatenation, no cross-block/request activation cache.
"""

import os
import types
import torch


def key_of(x):
    # data_ptr incorporates storage offset. Hold first input alive through the scope.
    version = None if x.is_inference() else x._version
    return (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype, x.device, version)


def linear_quantized(x, a, sa, codes, scales):
    """Exact post-quantization body of explicit_a8.linear (cublas_int32 only)."""
    from .fixed_a8 import int32_gemm
    from comfy_kitchen.backends import cuda as ck

    acc = int32_gemm(a, codes)
    out = torch.empty((a.shape[0], codes.shape[0]), device=x.device, dtype=x.dtype)
    empty = torch.empty(0, device=x.device, dtype=x.dtype)
    ck._C.dequantize_int8_linear(
        ck._wrap_for_dlpack(acc),
        ck._wrap_for_dlpack(sa),
        ck._wrap_for_dlpack(scales.contiguous()),
        ck._wrap_for_dlpack(empty),
        ck._wrap_for_dlpack(out),
        ck.DTYPE_TO_CODE[x.dtype],
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    return out.reshape(*x.shape[:-1], codes.shape[0])


def unpack(base):
    wide = base.qweight.shape[1] >= 16384
    return base.decoder.unpack(
        base.qweight,
        base.bits,
        base.k,
        bn=64 if wide else 32,
        bk=256 if wide else 64,
        num_warps=8 if wide else 4,
    )


def install(bb):
    """Install after final rebind. Returns object with stats and restore()."""
    from ..storage import FrozenPackedBase
    from .fixed_a8 import quantize

    stats = {"attention_scopes": 0, "quantize_calls": 0, "reuse_hits": 0}
    originals = []
    for block in bb.layers:
        bases = [block.proj[k].base for k in ("q", "k", "v")]
        if not all(isinstance(b, FrozenPackedBase) for b in bases):
            raise ValueError("Only original FrozenPackedBase W5/W6 q/k/v supported")
        state = {
            "active": False,
            "input": None,
            "key": None,
            "a": None,
            "sa": None,
            "seen": set(),
        }
        old_attn = block.attn_part

        def scoped(self, x, ctx, _old=old_attn, _state=state):
            if _state["active"]:
                raise RuntimeError(
                    "Attention share workspace is not recursive/concurrent"
                )
            _state["active"] = True
            stats["attention_scopes"] += 1
            try:
                from kohakumoss.types.context import SHIFT_ATTN, SCALE_ATTN, GATE_ATTN

                ops, seq, mod = self.ops, ctx.seq, ctx.mod
                seg = seq.seg_joint
                y = self._cast(
                    ops.rms_norm_mod(
                        x,
                        self.norm1.weight,
                        mod[:, SHIFT_ATTN],
                        mod[:, SCALE_ATTN],
                        seg,
                        self.norm1.eps,
                    )
                )
                q, k = self._qk(
                    y,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    seq.pos_joint,
                    ctx.bands,
                    self.q_norm.eps,
                )
                v = self.proj["v"](y).unflatten(-1, (self.heads, -1))
                del y
                o = ops.attention(q, k, v, seq.attn_joint).flatten(1)
                del q, k, v
                r = ctx.residual_scale[self.index]
                return self._post(self._attn_out(o, x, mod[:, GATE_ATTN], seg, r))
            finally:
                _state.update(
                    active=False, input=None, key=None, a=None, sa=None, seen=set()
                )

        block.attn_part = types.MethodType(scoped, block)
        patched = []
        for name, base in zip(("q", "k", "v"), bases):
            old = base.linear

            def shared(x, bias, _base=base, _old=old, _state=state, _name=name):
                if not _state["active"]:
                    return _old(x, bias)
                if bias is not None:
                    raise ValueError("No q/k/v bias allowed")
                if x.device != _base.qweight.device or x.dtype not in (
                    torch.float16,
                    torch.bfloat16,
                ):
                    raise ValueError("Original placement/activation contract required")
                if _name in _state["seen"]:
                    raise RuntimeError("Repeated q/k/v call in one scope")
                key = key_of(x)
                if _state["key"] is None:
                    a, sa = quantize(x)
                    _state.update(input=x, key=key, a=a, sa=sa)
                    stats["quantize_calls"] += 1
                elif _state["key"] != key:
                    raise RuntimeError("q/k/v inputs differ; refuse sharing")
                else:
                    stats["reuse_hits"] += 1
                _state["seen"].add(_name)
                codes = unpack(_base)
                out = linear_quantized(
                    x, _state["a"], _state["sa"], codes, _base.wscales
                )
                _base.calls += 1
                if len(_state["seen"]) == 3:
                    _state.update(input=None, a=None, sa=None)
                return out

            base.linear = shared
            patched.append((base, old))
        originals.append((block, old_attn, patched, state))

    class Handle:
        def __init__(self):
            self.stats = stats

        def restore(self):
            if any(s["active"] for _, _, _, s in originals):
                raise RuntimeError("Cannot restore active attention")
            for block, old, patched, _ in originals:
                block.attn_part = old
                for base, fn in patched:
                    base.linear = fn

    return Handle()
