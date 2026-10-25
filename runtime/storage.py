"""Canonical frozen SFT projection storage with an explicitly owned decoder.

Payload packing, FP32 row scales, fixed ConvRot64/256 A8, INT32 GEMM and
Selected FP16/BF16 dequantization order are unchanged from the measured frozen backend.
"""
import torch
from .vendor.fixed_a8 import quantize,int32_gemm

def linear(x,codes,scales):
    from comfy_kitchen.backends import cuda as ck
    a,sa=quantize(x)
    acc=int32_gemm(a,codes)
    out=torch.empty((a.shape[0],codes.shape[0]),device=x.device,dtype=x.dtype)
    empty=torch.empty(0,device=x.device,dtype=x.dtype)
    ck._C.dequantize_int8_linear(
        ck._wrap_for_dlpack(acc),ck._wrap_for_dlpack(sa),
        ck._wrap_for_dlpack(scales.contiguous()),ck._wrap_for_dlpack(empty),
        ck._wrap_for_dlpack(out),ck.DTYPE_TO_CODE[x.dtype],
        torch.cuda.current_stream(x.device).cuda_stream)
    return out.reshape(*x.shape[:-1],codes.shape[0])

class FrozenPackedBase(torch.nn.Module):
    def __init__(self,qweight,wscales,k,bits,decoder):
        super().__init__()
        if qweight.dtype!=torch.uint8 or wscales.dtype!=torch.float32:
            raise ValueError("Original UINT8 packed codes and FP32 row scales required")
        if bits not in (5,6) or tuple(qweight.shape)!=((k*bits+7)//8,wscales.shape[0]) or tuple(wscales.shape)!=(qweight.shape[1],1):
            raise ValueError("Original W5/W6 packed dimensions required")
        self.register_buffer("qweight",qweight)
        self.register_buffer("wscales",wscales)
        self.k,self.bits,self.decoder=k,bits,decoder
        self.calls=0
        self.row_tile=0

    def weight(self,dtype):
        raise RuntimeError("Frozen packed projection has no dense fallback")

    def linear(self,x,bias):
        if bias is not None:
            raise ValueError("Frozen joint projections have no bias")
        if x.device!=self.qweight.device or x.dtype not in (torch.float16,torch.bfloat16):
            raise ValueError("Explicit same-device FP16/BF16 activation boundary required")
        codes=unpack_base(self)
        if self.row_tile:
            from .vendor.native_projection_tiles import linear as tiled_linear
            result=tiled_linear(x,codes,self.wscales,tile=self.row_tile,quantize=quantize,int32_gemm=int32_gemm)
        else:
            result=linear(x,codes,self.wscales)
        self.calls+=1
        return result

class FrozenInt8Base(torch.nn.Module):
    def __init__(self,codes,scales):
        super().__init__()
        if codes.dtype!=torch.int8 or scales.dtype!=torch.float32:
            raise ValueError("Original INT8 codes and FP32 row scales required")
        self.register_buffer("codes",codes)
        self.register_buffer("scales",scales)
        self.row_tile=0

    def weight(self,dtype):
        raise RuntimeError("Frozen INT8 projection has no dense fallback")

    def linear(self,x,bias):
        if bias is not None:
            raise ValueError("Frozen joint projections have no bias")
        if self.row_tile:
            from .vendor.native_projection_tiles import linear as tiled_linear
            return tiled_linear(x,self.codes,self.scales,tile=self.row_tile,quantize=quantize,int32_gemm=int32_gemm)
        return linear(x,self.codes,self.scales)

def unpack_base(base):
    wide=base.qweight.shape[1]>=16384
    return base.decoder.unpack(base.qweight,base.bits,base.k,
                               bn=64 if wide else 32,bk=256 if wide else 64,
                               num_warps=8 if wide else 4)

def attach_frozen(model,state,bits,decoder):
    stems=sorted(k[:-8] for k in state if k.endswith(".qweight"))
    if len(stems)!=252:
        raise ValueError("Expected 42×6 frozen SFT joint projections")
    for stem in stems:
        projection=model.get_submodule(stem)
        dtype=projection.weight.dtype
        base=FrozenPackedBase(state[stem+".qweight"],state[stem+".wscales"],
                              projection.in_features,bits,decoder)
        projection.set_base(base)
        del projection.weight
        projection.__dict__["_held"]=torch.empty(0,device=base.qweight.device,dtype=dtype)
    return stems
