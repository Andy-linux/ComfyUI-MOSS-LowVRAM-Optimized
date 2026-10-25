"""Inference-only aliasing wrapper around the unchanged original RoPE kernel.

The original kernel owns disjoint (token,head) rows and loads each complete row
before either store. Fixed launch, never repeated autotuning of mutable inputs.
Only fresh independent Q/K output storages owned by register3 are admitted.
"""
import torch

CONFIG={"ROWS":16,"warps":4}

def qk_norm_rope_inplace(q,k,q_weight,k_weight,pos,bands,eps):
    from kohakumoss_kernels.backends.triton.blocks.rope import _fwd_launch
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad:
        raise ValueError("In-place RoPE requires inference/no-grad execution")
    if q.shape!=k.shape or q.ndim!=3 or tuple(q.shape[1:])!=(24,128):
        raise ValueError("Qualified THD H24/D128 pair required")
    if q.dtype not in (torch.float16,torch.bfloat16) or k.dtype!=q.dtype or not q.is_cuda or q.device!=k.device:
        raise ValueError("Same-device CUDA FP16/BF16 pair required")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("Contiguous THD outputs required")
    if q.storage_offset()!=0 or k.storage_offset()!=0 or q.untyped_storage().nbytes()!=q.nbytes or k.untyped_storage().nbytes()!=k.nbytes:
        raise ValueError("Q/K must own independent exact-size projection outputs")
    if q.untyped_storage().data_ptr()==k.untyped_storage().data_ptr():
        raise ValueError("Disjoint Q/K storages required")
    if any(t.device!=q.device for t in (q_weight,k_weight,pos,bands)):
        raise ValueError("Original weights/positions/bands must share the explicit device")
    capability=tuple(torch.cuda.get_device_capability(q.device))
    if capability==(7,5) and q.dtype==torch.bfloat16:
        raise ValueError("BF16 requires SM80 or newer")
    if capability not in ((8,9),(7,5)):
        raise ValueError("In-place wrapper is qualified only for SM89 or measured SM75")
    if capability==(7,5):
        import triton
        if str(triton.__version__).split(".")[:2]!=["3","2"]:
            raise ValueError("Measured SM75 in-place wrapper requires uniform Triton3.2")
    n,h,_=q.shape
    qr=torch.empty(n*h,device=q.device,dtype=torch.float32)
    kr=torch.empty_like(qr)
    _fwd_launch(CONFIG,q,k,q_weight,k_weight,pos,bands,q,k,qr,kr,eps)
    return q,k
