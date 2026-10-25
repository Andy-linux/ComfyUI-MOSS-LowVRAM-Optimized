"""True QKV projection fusion candidates; original packed bytes/scales unchanged.

One fixed A8, one lossless multi-source unpack, one GEMM covering Q+K+V.
native uses original cublas INT8->INT32, register uses real tl.dot INT8->INT32.
Both dequantize order0 and produce three contiguous FP16 outputs in one launch.
Not qualified/default until per-architecture INT32, output, attention and image gates.
"""
import threading
import types
import torch
import triton
import triton.language as tl


@triton.jit
def _unpack3(PQ, PK, PV, OUT, D:tl.constexpr, K:tl.constexpr,
             BITS:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr):
    p = tl.program_id(0) // tl.cdiv(D,BN)
    ni = (tl.program_id(0) % tl.cdiv(D,BN))*BN+tl.arange(0,BN)
    ki = tl.program_id(1)*BK+tl.arange(0,BK)
    ptr = tl.where(p==0,PQ,tl.where(p==1,PK,PV))
    bit = ki*BITS
    byte,shift = bit//8,bit%8
    rows:tl.constexpr = (K*BITS+7)//8
    valid = (ki[:,None]<K)&(ni[None,:]<D)
    lo = tl.load(ptr+byte[:,None]*D+ni[None,:],valid,0).to(tl.int32)
    hi = tl.load(ptr+(byte[:,None]+1)*D+ni[None,:],
                 valid&((byte[:,None]+1)<rows)&((shift[:,None]+BITS)>8),0).to(tl.int32)
    code = ((lo|(hi<<8))>>shift[:,None])&((1<<BITS)-1)
    signed = ((code^(1<<(BITS-1)))-(1<<(BITS-1))).to(tl.int8)
    tl.store(OUT+(p*D+ni[:,None])*K+ki[None,:],tl.trans(signed),
             (ni[:,None]<D)&(ki[None,:]<K))


@triton.jit
def _dequant3(ACC,SA,SQ,SK,SV,OUT,M:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    ix = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    n = ix%(3*D)
    m = ix//(3*D)
    p = n//D
    sp = tl.where(p==0,SQ,tl.where(p==1,SK,SV))
    acc = tl.load(ACC+ix,m<M,0).to(tl.float32)
    sa = tl.load(SA+m,m<M,0).to(tl.float32)
    sw = tl.load(sp+n%D,m<M,0).to(tl.float32)
    y = (acc*sa)*sw
    tl.store(OUT+p*M*D+m*D+n%D,y,m<M)


@triton.jit
def _gemm3(A,W,SA,SQ,SK,SV,OUT,DEBUG,M:tl.constexpr,D:tl.constexpr,K:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,STORE_ACC:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    ni=tl.program_id(1)*BN+tl.arange(0,BN)
    ki=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.int32)
    for kb in range(tl.cdiv(K,BK)):
        kk=kb*BK+ki
        a=tl.load(A+mi[:,None]*K+kk[None,:],(mi[:,None]<M)&(kk[None,:]<K),0)
        w=tl.load(W+ni[None,:]*K+kk[:,None],(ni[None,:]<3*D)&(kk[:,None]<K),0)
        acc=tl.dot(a,w,acc,out_dtype=tl.int32)
    if STORE_ACC:
        tl.store(DEBUG+mi[:,None]*(3*D)+ni[None,:],acc,(mi[:,None]<M)&(ni[None,:]<3*D))
    p=ni//D
    sp=tl.where(p==0,SQ,tl.where(p==1,SK,SV))
    sa=tl.load(SA+mi,mi<M,0).to(tl.float32)
    sw=tl.load(sp+ni%D,ni<3*D,0).to(tl.float32)
    y=(acc.to(tl.float32)*sa[:,None])*sw[None,:]
    tl.store(OUT+p[None,:]*M*D+mi[:,None]*D+ni[None,:]%D,
             y,(mi[:,None]<M)&(ni[None,:]<3*D))


CONFIG = dict(bm=64,bn=64,bk=64,warps=4,stages=2)
SM89_REGISTER_CONFIG = dict(bm=64,bn=128,bk=64,warps=4,stages=2)


class Workspace:
    """Request-serialized workspace; may share old install_scratch() dictionary."""
    def __init__(self, scratch=None):
        self.scratch = {} if scratch is None else scratch
        self._lock = threading.Lock()
        self._stream = None
        self.stats = dict(calls=0,quantize_calls=0,unpack_launches=0,
                          gemm_launches=0,dequant_launches=0,weight_scratch_bytes=0)
        self.last_kernel = None

    def close(self):
        if self._lock.locked():
            raise RuntimeError("Cannot close active QKV workspace")
        self.scratch.clear()


def _validate_bases(bases):
    from ..storage import FrozenPackedBase
    if len(bases)!=3 or any(type(b) is not FrozenPackedBase for b in bases):
        raise ValueError("Exactly three original FrozenPackedBase Q/K/V required")
    d=bases[0].qweight.shape[1];k=bases[0].k;bits=bases[0].bits
    for b in bases:
        if b.k!=k or b.bits!=bits or tuple(b.qweight.shape)!=((k*bits+7)//8,d):
            raise ValueError("QKV original packed shapes/bits differ")
        if b.qweight.dtype!=torch.uint8 or not b.qweight.is_contiguous():
            raise ValueError("Original contiguous UINT8 codes required")
        if b.wscales.dtype!=torch.float32 or tuple(b.wscales.shape)!=(d,1) or not b.wscales.is_contiguous():
            raise ValueError("Original contiguous FP32 row scales required")
        if b.qweight.device!=bases[0].qweight.device or b.wscales.device!=b.qweight.device:
            raise ValueError("Explicit shared payload/scale device required")
    return d,k,bits


def project(x,bases,*,workspace,backend="native",config=CONFIG,return_acc=False):
    """Return three contiguous [M,D] tensors; source buffers never rewritten."""
    from .fixed_a8 import quantize,int32_gemm
    d,k,bits=_validate_bases(bases)
    if backend not in ("native","register"):
        raise ValueError("backend must be native or register")
    if x.device!=bases[0].qweight.device or not x.is_cuda or x.dtype not in (torch.float16,torch.bfloat16) or x.ndim!=2 or x.shape[1]!=k:
        raise ValueError("CUDA FP16/BF16 [M,K] same-device input required")
    capability=torch.cuda.get_device_capability(x.device)
    if capability==(7,5) and x.dtype==torch.bfloat16:
        raise ValueError("BF16 requires SM80 or newer")
    if capability not in ((7,5),(8,9)):
        raise ValueError("Only SM75/SM89 experimental qualification supported")
    if not workspace._lock.acquire(blocking=False):
        raise RuntimeError("Concurrent QKV workspace use refused")
    try:
        stream=(str(x.device),torch.cuda.current_stream(x.device).cuda_stream)
        if workspace._stream is None:workspace._stream=stream
        elif workspace._stream!=stream:
            raise RuntimeError("Workspace is bound to one CUDA stream; create another per stream")
        a,sa=quantize(x)
        size=3*d*k
        buf=workspace.scratch.get("buffer")
        if buf is None or buf.numel()<size:
            workspace.scratch.clear()
            buf=torch.empty(size,device=x.device,dtype=torch.int8)
            workspace.scratch["buffer"]=buf
        if buf.dtype!=torch.int8 or buf.device!=x.device or not buf.is_contiguous():
            raise ValueError("Shared unpack scratch incompatible")
        codes=buf[:size].view(3*d,k)
        _unpack3[(3*triton.cdiv(d,32),triton.cdiv(k,64))](
            *(b.qweight for b in bases),codes,d,k,bits,32,64,num_warps=4,num_stages=1)
        m=len(x)
        out=torch.empty((3,m,d),device=x.device,dtype=x.dtype)
        if backend=="native":
            acc=int32_gemm(a,codes)
            _dequant3[(triton.cdiv(m*3*d,256),)](
                acc,sa,*(b.wscales for b in bases),out,m,d,256,
                num_warps=4,enable_fp_fusion=False)
            debug=acc if return_acc else None
            workspace.stats["dequant_launches"]+=1
        else:
            debug=torch.empty((m,3*d),device=x.device,dtype=torch.int32) if return_acc else out
            c=config
            workspace.last_kernel=_gemm3[(triton.cdiv(m,c["bm"]),triton.cdiv(3*d,c["bn"]))](
                a,codes,sa,*(b.wscales for b in bases),out,debug,m,d,k,
                c["bm"],c["bn"],c["bk"],return_acc,
                num_warps=c["warps"],num_stages=c["stages"],enable_fp_fusion=False)
        for b in bases:b.calls+=1
        for key in ("calls","quantize_calls","unpack_launches","gemm_launches"):workspace.stats[key]+=1
        workspace.stats["weight_scratch_bytes"]=buf.numel()
        result=tuple(out[i] for i in range(3))
        return (result,debug) if return_acc else result
    finally:
        workspace._lock.release()


def install(bb,*,backend="native",scratch=None,experimental=False,config=CONFIG):
    """Install after final bind_ops; no LoRA/hooks/custom projections bypassed.

    Until architecture+image qualification, explicit experimental=True is mandatory.
    Return handle with restore(), receipt and cumulative exact projection counters.
    """
    from .inference_contracts_v3 import projection_errors
    if not experimental or backend not in ("native","register"):
        raise ValueError("Explicit experimental native/register backend required")
    if getattr(bb,"_qkv_fused_r2",False):
        raise RuntimeError("QKV fusion already installed")
    errors=[]
    if not len(bb.layers):
        raise ValueError("Cannot install QKV fusion on empty backbone")
    for i,b in enumerate(bb.layers):
        for name in ("q","k","v"):
            lin=b.proj[name]
            errors += [f"layer {i} {name}: "+e.replace("ffn_up_gate",name) for e in projection_errors(lin)]
            if lin.weight_like.dtype not in (torch.float16,torch.bfloat16) or lin.weight_like.device.type!="cuda":
                errors.append(f"layer {i}: only CUDA FP16/BF16 installation supported")
            if lin.in_features!=3072 or lin.out_features!=3072:
                errors.append(f"layer {i}: only D3072 QKV shape supported")
        try:_validate_bases([b.proj[n].base for n in ("q","k","v")])
        except ValueError as e:errors.append(f"layer {i}: {e}")
        if "attn_part" in vars(b):
            errors.append(f"layer {i}: existing attention override/share must be restored")
    if errors:raise ValueError("; ".join(errors))
    config=dict(config)
    workspace=Workspace(scratch)
    originals=[]
    from kohakumoss.types.context import SHIFT_ATTN,SCALE_ATTN,GATE_ATTN
    for block in bb.layers:
        originals.append((block,block.attn_part))
        def attn(self,x,ctx):
            ops,seq,mod=self.ops,ctx.seq,ctx.mod
            seg=seq.seg_joint
            y=self._cast(ops.rms_norm_mod(x,self.norm1.weight,mod[:,SHIFT_ATTN],mod[:,SCALE_ATTN],seg,self.norm1.eps))
            q,k,v=project(y,[self.proj[n].base for n in ("q","k","v")],
                          workspace=workspace,backend=backend,config=config)
            del y
            q,k=ops.qk_norm_rope(q.unflatten(-1,(self.heads,-1)),k.unflatten(-1,(self.heads,-1)),
                                self.q_norm.weight,self.k_norm.weight,seq.pos_joint,ctx.bands,self.q_norm.eps)
            v=v.unflatten(-1,(self.heads,-1))
            o=ops.attention(q,k,v,seq.attn_joint).flatten(1)
            del q,k,v
            r=ctx.residual_scale[self.index]
            return self._post(self._attn_out(o,x,mod[:,GATE_ATTN],seg,r))
        block.attn_part=types.MethodType(attn,block)
    bb._qkv_fused_r2=True
    class Handle:
        def __init__(self):
            self.workspace=workspace
            self.stats=workspace.stats
            self.receipt=dict(backend=backend,experimental=True,config=dict(config),
                              logical_projection_calls_per_scope=3,
                              quantize_calls_per_scope=1,unpack_launches_per_scope=1,
                              gemm_launches_per_scope=1,
                              dequant_launches_per_scope=int(backend=="native"),
                              register_dequant_fused=backend=="register",
                              packed_bytes_and_scales_unchanged=True)
        def restore(self):
            if workspace._lock.locked():raise RuntimeError("Active QKV call")
            for b,old in originals:b.attn_part=old
            bb._qkv_fused_r2=False
    return Handle()
