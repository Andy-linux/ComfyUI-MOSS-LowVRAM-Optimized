"""M-tiled native INT32 projection; no new kernel or quantization semantics.

Caller decodes codes once into its owned scratch before entering linear().
Quantization is once over complete x; every accumulator is released before the
next tile allocation. Dequantization writes final FP16/BF16 output slices directly.
Request owner must serialize its decoder/workspace and use one CUDA stream.
Importing this module does not import torch or initialize CUDA.
"""

def native_dequantize_into(acc,activation_scales,weight_scales,out):
    import torch
    from comfy_kitchen.backends import cuda as ck
    empty=torch.empty(0,device=out.device,dtype=out.dtype)
    ck._C.dequantize_int8_linear(
        ck._wrap_for_dlpack(acc),ck._wrap_for_dlpack(activation_scales),
        ck._wrap_for_dlpack(weight_scales),ck._wrap_for_dlpack(empty),
        ck._wrap_for_dlpack(out),ck.DTYPE_TO_CODE[out.dtype],
        torch.cuda.current_stream(out.device).cuda_stream)


def linear(x,codes,scales,*,tile=2048,quantize,int32_gemm,
           dequantize_into=None,accumulator_observer=None):
    """Return FP16/BF16 [M,N]; callbacks are canonical fixed-A8/native-INT32 ops.

    accumulator_observer(acc,start,end) is qualification-only; production leaves
    it None. It must not retain acc. No packed payload or scales are modified.
    This helper does not install/patch any model, class, writer or decoder.
    """
    import torch
    if not isinstance(tile,int) or isinstance(tile,bool) or tile<=0:
        raise ValueError("Positive integer M tile required")
    if x.ndim!=2 or codes.ndim!=2 or x.shape[1]!=codes.shape[1]:
        raise ValueError("Matching [M,K] input/[N,K] decoded codes required")
    if x.dtype not in (torch.float16,torch.bfloat16) or codes.dtype!=torch.int8 or scales.dtype!=torch.float32:
        raise ValueError("Original FP16/BF16 activation/INT8 codes/FP32 weight scales required")
    if tuple(scales.shape)!=(codes.shape[0],1):
        raise ValueError("Original [N,1] weight scales required")
    if not x.is_cuda or not all(t.device==x.device for t in (codes,scales)):
        raise ValueError("Same explicit CUDA device required")
    if not all(t.is_contiguous() for t in (x,codes,scales)):
        raise ValueError("Contiguous original operands required; no implicit layout copy")
    if not callable(quantize) or not callable(int32_gemm):
        raise ValueError("Explicit canonical quantize and native INT32 GEMM callbacks required")
    if x.requires_grad or codes.requires_grad or scales.requires_grad:
        raise ValueError("Frozen inference operands required")
    dequantize_into=native_dequantize_into if dequantize_into is None else dequantize_into
    if not callable(dequantize_into):raise ValueError("Direct-output dequant callback required")
    a,sa=quantize(x)
    if a.dtype!=torch.int8 or tuple(a.shape)!=tuple(x.shape) or not a.is_contiguous():
        raise ValueError("Canonical contiguous full-input A8 quantization required")
    if sa.dtype!=torch.float32 or sa.numel()!=len(x) or not sa.is_contiguous():
        raise ValueError("Canonical FP32 row activation scales required")
    if a.device!=x.device or sa.device!=x.device:
        raise ValueError("Quantized operands must remain on input device")
    out=torch.empty((len(x),codes.shape[0]),device=x.device,dtype=x.dtype)
    for start in range(0,len(x),tile):
        end=min(start+tile,len(x))
        acc=int32_gemm(a[start:end],codes)
        if acc.dtype!=torch.int32 or tuple(acc.shape)!=(end-start,codes.shape[0]) or not acc.is_contiguous():
            raise ValueError("Canonical contiguous tile INT32 accumulator required")
        if accumulator_observer is not None:accumulator_observer(acc,start,end)
        dequantize_into(acc,sa[start:end],scales,out[start:end])
        del acc
    return out
