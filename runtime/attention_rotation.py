"""Matched signed orthogonal H128 for optional quantized attention.

(QR)(KR)^T = QK^T in real arithmetic. FP32 butterflies and one explicit
FP16/BF16 output rounding precede Sage's own quantizer; these are additional
numerical boundaries, and this path is always exposed as an optional backend.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(Q,K,OQ,OK,N:tl.constexpr,H:tl.constexpr,QS0:tl.constexpr,QS1:tl.constexpr,
            KS0:tl.constexpr,KS1:tl.constexpr,ROWS:tl.constexpr):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    col=tl.arange(0,128)
    is_q=tl.program_id(1)==0
    ptr=Q if is_q else K
    s0=QS0 if is_q else KS0;s1=QS1 if is_q else KS1
    offset=(row//H)[:,None]*s0+(row%H)[:,None]*s1+col[None,:]
    x=tl.load(ptr+offset,row[:,None]<N*H,other=0).to(tl.float32)
    # Deterministic Rademacher diagonal; no RNG state or per-request table.
    z=(col.to(tl.uint32)+1)*2654435769
    z=(z^(z>>16))*2246822507;z=z^(z>>13)
    x=x*tl.where((z&1)==0,1.0,-1.0)[None,:]
    for shift in tl.static_range(7):
        bit=1<<shift
        other=tl.gather(x,tl.broadcast_to((col^bit)[None,:],(ROWS,128)),1)
        x=tl.where((col&bit)[None,:]==0,x+other,other-x)
    out=OQ if is_q else OK
    tl.store(out+row[:,None]*128+col[None,:],x*0.08838834764831845,row[:,None]<N*H)


def rotate_qk(q,k):
    if q.shape!=k.shape or q.ndim!=3 or q.shape[-1]!=128 or q.dtype!=k.dtype or q.dtype not in (torch.float16,torch.bfloat16) or q.device!=k.device or q.stride(-1)!=1 or k.stride(-1)!=1:
        raise ValueError('Matching FP16/BF16 THD128 Q/K required')
    oq=torch.empty(q.shape,device=q.device,dtype=q.dtype);ok=torch.empty_like(oq)
    _rotate[(triton.cdiv(q.shape[0]*q.shape[1],4),2)](q,k,oq,ok,q.shape[0],q.shape[1],q.stride(0),q.stride(1),k.stride(0),k.stride(1),4,num_warps=4,enable_fp_fusion=False)
    return oq,ok
