"""Minimal fixed A8/native INT32 primitives, copied with original call semantics.

No plugin-wide monkeypatch, alternative quantizer, FP32 GEMM or hidden fallback.
Canonical runtime storage classes may call these functions without historical
benchmark imports. CUDA/device validation belongs to the public request owner.
"""
import torch


def quantize(x):
    from comfy_kitchen.backends import cuda as ck
    return ck.quantize_int8_rowwise_convrot64(
        x.reshape(-1,x.shape[-1]).contiguous(),256)


def int32_gemm(a,codes):
    from comfy_kitchen.backends import cuda as ck
    if a.dtype!=torch.int8 or codes.dtype!=torch.int8:
        raise ValueError("Original INT8 A/W inputs required")
    m,k=a.shape;n,kw=codes.shape
    if k!=kw or k%16 or n%16:
        raise ValueError("Original MOSS aligned GEMM dimensions required")
    out=torch.empty((m,n),device=a.device,dtype=torch.int32)
    ck._C.cublas_gemm_int8(
        ck._wrap_for_dlpack(a.contiguous()),ck._wrap_for_dlpack(codes.contiguous()),
        ck._wrap_for_dlpack(out),ck._wrap_for_dlpack(ck.get_cublas_workspace()),
        torch.cuda.current_stream(a.device).cuda_stream)
    return out
