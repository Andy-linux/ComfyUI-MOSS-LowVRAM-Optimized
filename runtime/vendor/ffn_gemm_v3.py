"""Experimental native INT8 Tensor Core up+gate GEMM with register INT32 SiTU epilogue.

No packed decoder is fused here: input weights are unchanged unpacked INT8 [2F,K].
The same fixed ConvRot64/256 A8 activation and order0 FP32 scales are used.
Not a default backend. Exact equality and actual measured speed decide acceptance.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _up_gate_situ(A, W, SA, SW, O, ACC_OUT, M: tl.constexpr, F: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, STORE_ACC: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    au = tl.zeros((BM, BN), tl.int32)
    ag = tl.zeros((BM, BN), tl.int32)
    for kb in range(tl.cdiv(K, BK)):
        kk = kb * BK + ki
        a = tl.load(A + mi[:, None] * K + kk[None, :],
                    (mi[:, None] < M) & (kk[None, :] < K), other=0)
        wu = tl.load(W + ni[None, :] * K + kk[:, None],
                     (ni[None, :] < F) & (kk[:, None] < K), other=0)
        wg = tl.load(W + (ni[None, :] + F) * K + kk[:, None],
                     (ni[None, :] < F) & (kk[:, None] < K), other=0)
        au = tl.dot(a, wu, au, out_dtype=tl.int32)
        ag = tl.dot(a, wg, ag, out_dtype=tl.int32)
    if STORE_ACC:
        mask = (mi[:, None] < M) & (ni[None, :] < F)
        tl.store(ACC_OUT + mi[:, None] * (2 * F) + ni[None, :], au, mask)
        tl.store(ACC_OUT + mi[:, None] * (2 * F) + F + ni[None, :], ag, mask)
    sa = tl.load(SA + mi, mi < M, other=0).to(tl.float32)
    su = tl.load(SW + ni, ni < F, other=0).to(tl.float32)
    sg = tl.load(SW + ni + F, ni < F, other=0).to(tl.float32)
    u = (au.to(tl.float32) * sa[:, None]) * su[None, :]
    g = (ag.to(tl.float32) * sa[:, None]) * sg[None, :]
    # Preserve the accepted native FP16 dequant rounding before nonlinear SiTU.
    u = u.to(O.dtype.element_ty).to(tl.float32)
    g = g.to(O.dtype.element_ty).to(tl.float32)
    sig = 1.0 / (1.0 + libdevice.exp(-g))
    y = (4.0 * libdevice.tanh(g * 0.25) * sig) * (25.0 * libdevice.tanh(u * 0.04))
    tl.store(O + mi[:, None] * F + ni[None, :], y.to(O.dtype.element_ty),
             (mi[:, None] < M) & (ni[None, :] < F))


CONFIGS = (
    dict(bm=32, bn=64, bk=64, warps=4, stages=2),
    dict(bm=64, bn=64, bk=64, warps=4, stages=2),
    dict(bm=32, bn=64, bk=128, warps=4, stages=2),
    dict(bm=64, bn=64, bk=128, warps=4, stages=2),
)


def quantized_up_situ(a, codes, sa, scales, *, dtype=torch.float16, config=CONFIGS[0], return_acc=False):
    if a.dtype != torch.int8 or codes.dtype != torch.int8:
        raise ValueError("Unchanged INT8 activation and decoded weight codes required")
    if not a.is_contiguous() or not codes.is_contiguous() or scales.dtype != torch.float32:
        raise ValueError("Expected original contiguous integer layout and FP32 row scales")
    m, k = a.shape
    n, kw = codes.shape
    if k != kw or n % 2:
        raise ValueError("Expected codes [2F,K]")
    f = n // 2
    out = torch.empty((m, f), device=a.device, dtype=dtype)
    acc_out = torch.empty((m, 2 * f), device=a.device, dtype=torch.int32) if return_acc else out
    c = config
    kernel = _up_gate_situ[(triton.cdiv(m, c["bm"]), triton.cdiv(f, c["bn"]))](
        a, codes, sa, scales, out, acc_out, m, f, k, c["bm"], c["bn"], c["bk"], return_acc,
        num_warps=c["warps"], num_stages=c["stages"], enable_fp_fusion=False)
    quantized_up_situ.last_kernel = kernel
    return (out, acc_out) if return_acc else out


def up_situ(x, codes, scales, *, config=CONFIGS[0]):
    from .fixed_a8 import quantize
    a, sa = quantize(x)
    return quantized_up_situ(a, codes, sa, scales, dtype=x.dtype, config=config)


