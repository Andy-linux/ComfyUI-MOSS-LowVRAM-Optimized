"""Explicit destructive private-Q SDPA query tiling; full K/V per tile."""
import torch

def attention(q,k,v,scale,*,tile=2048,overwrite_private_q=False):
    if not overwrite_private_q or torch.is_grad_enabled():
        raise ValueError("Private Q overwrite consent and inference required")
    if tile!=2048 or q.shape!=k.shape or q.shape!=v.shape or q.ndim!=3 or tuple(q.shape[1:])!=(24,128):
        raise ValueError("Qualified tile2048 equal THD H24 D128 required")
    if not q.is_cuda or q.dtype not in (torch.float16,torch.bfloat16) or any(t.device!=q.device or t.dtype!=q.dtype or t.requires_grad or not t.is_contiguous() or t.storage_offset()!=0 or t.untyped_storage().nbytes()!=t.numel()*t.element_size() for t in (q,k,v)):
        raise ValueError("Independent exact-size contiguous CUDA FP16/BF16 storage required")
    ranges=[(t.data_ptr(),t.data_ptr()+t.numel()*t.element_size()) for t in (q,k,v)]
    if any(a[0]<b[1] and b[0]<a[1] for i,a in enumerate(ranges) for b in ranges[i+1:]):
        raise ValueError("Q/K/V byte ranges must be disjoint")
    # Caller owns Q exclusively: no later consumer needs pre-attention Q.
    # K/V are complete, read-only and cannot alias the overwritten Q.
    kk=k.transpose(0,1).unsqueeze(0)
    vv=v.transpose(0,1).unsqueeze(0)
    for start in range(0,len(q),tile):
        qq=q[start:start+tile].transpose(0,1).unsqueeze(0)
        out=torch.nn.functional.scaled_dot_product_attention(qq,kk,vv,dropout_p=0.0,is_causal=False,scale=scale)
        q[start:start+tile].copy_(out.squeeze(0).transpose(0,1))
        del out,qq
    return q
