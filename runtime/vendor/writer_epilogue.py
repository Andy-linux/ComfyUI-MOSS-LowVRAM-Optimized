"""Native INT32 -> FP16 boundary -> original gated residual epilogue candidate.

Keeps the complete native INT32 accumulator. Removes only the [M,D] FP16 linear
output write/read and one residual launch. No rounding reorder or register GEMM
claim. Reuse Kohaku scaled_writer for writer-input/gate scaling boundaries.
"""
import types
import torch
import triton
import triton.language as tl


@triton.jit
def _epilogue(ACC,SA,SW,X,GATE,SEG,OUT,M:tl.constexpr,D:tl.constexpr,
              GS0:tl.constexpr,GS1:tl.constexpr,R:tl.constexpr,
              BLOCK:tl.constexpr):
    ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row,col=ix//D,ix%D
    valid=row<M
    acc=tl.load(ACC+ix,valid,0).to(tl.float32)
    sa=tl.load(SA+row,valid,0).to(tl.float32)
    sw=tl.load(SW+col,valid,0).to(tl.float32)
    # Original native linear must round to FP16 BEFORE writer multiplication.
    o=((acc*sa)*sw).to(tl.float16).to(tl.float32)
    seg=tl.load(SEG+row,valid,0).to(tl.int64)
    gate=tl.load(GATE+seg*GS0+col*GS1,valid,0).to(tl.float32)
    o=o*gate
    x=tl.load(X+ix,valid,0).to(tl.float32)
    if R!=1.0:
        x=x*R
    tl.store(OUT+ix,(x+o).to(OUT.dtype.element_ty),valid)


def from_acc(acc,sa,scales,x,gate,seg,r,*,fp_fusion=True):
    """Gate is already scaled by original scaled_writer, not silently rescaled."""
    if acc.dtype!=torch.int32 or acc.shape!=x.shape or not acc.is_contiguous() or not x.is_contiguous():
        raise ValueError("Contiguous native INT32 accumulator and matching residual shape required")
    if x.dtype!=torch.float16 or gate.dtype not in (torch.float16,torch.float32):
        raise ValueError("Qualified candidate input boundary is FP16; gate is original FP16/FP32")
    if scales.dtype!=torch.float32 or sa.dtype!=torch.float32 or not scales.is_contiguous() or not sa.is_contiguous():
        raise ValueError("Original native contiguous FP32 activation/row scales required")
    if sa.numel()!=len(x) or scales.numel()!=x.shape[1] or seg.dtype not in (torch.int32,torch.int64):
        raise ValueError("Original activation/weight scale and segmentation dimensions required")
    if gate.ndim!=2 or gate.shape[1]!=x.shape[1] or seg.ndim!=1 or len(seg)!=len(x) or not seg.is_contiguous():
        raise ValueError("Original [B,D] gate/[M] segmentation required")
    tensors=(acc,sa,scales,x,gate,seg)
    if not all(t.device==x.device for t in tensors) or not x.is_cuda:
        raise ValueError("Same explicit CUDA device required")
    out=torch.empty_like(x);m,d=x.shape
    _epilogue[(triton.cdiv(m*d,256),)](acc,sa,scales,x,gate,seg,out,m,d,*gate.stride(),r,256,
                                    num_warps=4,enable_fp_fusion=fp_fusion)
    return out


def make_writer(projection,*,unpack,fp_fusion=True,s_in=1.,g=1.,projection_validator=None):
    """Create a standalone candidate writer; no global patch/default install.

    unpack(base) is a session-owned unchanged original decoder callback.
    Original scaled_writer applies a*s_in and gate*g with their old dtype
    boundaries, so this function only consumes its already-scaled arguments.
    """
    if projection_validator is None:
        from .inference_contracts_v3 import projection_errors
        projection_validator=projection_errors
    if not callable(projection_validator):raise ValueError("Explicit canonical projection validator required")
    errors=list(projection_validator(projection))
    if not hasattr(projection.base,"qweight") or not hasattr(projection.base,"wscales"):
        errors.append("Only packed W5/W6 writer candidate is implemented")
    if not callable(unpack):
        errors.append("Explicit unchanged decoder callback required")
    if errors:raise ValueError("; ".join(errors))
    from .fixed_a8 import quantize,int32_gemm
    from kohakumoss.model.stream import scaled_writer
    def base(a,x,gate,seg,r):
        codes=unpack(projection.base)
        aa,sa=quantize(a)
        acc=int32_gemm(aa,codes)
        return from_acc(acc,sa,projection.base.wscales,x,gate,seg,r,fp_fusion=fp_fusion)
    return scaled_writer(base,s_in,g)


def install(bb,*,unpack,fp_fusion=True,experimental=False,projection_validator=None):
    """Optional explicit candidate, o/down only; AFTER final bind/install.

    No base.linear override. Restore before rebinding kernels. Full-image numeric
    gates still required; this helper is not enabled in production runtime.
    """
    if projection_validator is None:
        from .inference_contracts_v3 import projection_errors
        projection_validator=projection_errors
    if not callable(projection_validator):raise ValueError("Explicit canonical projection validator required")
    if not experimental:raise ValueError("Explicit experimental qualification required")
    if tuple(torch.cuda.get_device_capability(next(bb.parameters()).device))!=(8,9):
        raise ValueError("Writer epilogue qualified only on SM89; SM75 remains pending")
    errors=[]
    for i,block in enumerate(bb.layers):
        for name in ("o","ffn_down"):
            errors += [f"layer {i} {name}: "+e for e in projection_validator(block.proj[name])]
    if errors:raise ValueError("; ".join(errors))
    plans=[]
    for i,b in enumerate(bb.layers):
        cfg=bb.stream.block(i)
        aw=make_writer(b.proj["o"],unpack=unpack,fp_fusion=fp_fusion,
                       s_in=cfg.in_attn,g=cfg.gate_attn,projection_validator=projection_validator)
        fw=make_writer(b.proj["ffn_down"],unpack=unpack,fp_fusion=fp_fusion,
                       s_in=cfg.in_ffn,g=cfg.gate_ffn,projection_validator=projection_validator)
        plans.append((b,b._attn_out,b._ffn_out,aw,fw))
    originals=[]
    for b,old_a,old_f,aw,fw in plans:
        originals.append((b,old_a,old_f))
        b._attn_out,b._ffn_out=aw,fw
    def restore():
        for b,a,f in originals:b._attn_out,b._ffn_out=a,f
    return dict(backend="native INT32 accumulator + FP16 boundary + gated residual epilogue",
                external_INT32_accumulator=True,FP16_linear_output_intermediate_removed=True,
                original_scaled_writer_reused=True,fp_fusion=fp_fusion,
                qualification="PENDING_MICRO_AND_FULLIMAGE"),restore
