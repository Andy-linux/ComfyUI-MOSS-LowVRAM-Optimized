"""Inference-only private embed destination; original copies become in-place."""
from types import MethodType
import torch
from kohakumoss.model.backbone import MossBackbone

def install(model):
    if type(model) is not MossBackbone or "embed" in vars(model):
        raise TypeError("Canonical unmodified MossBackbone embed required")
    old=model.embed
    state={"seq":None}
    def embed(self,img,text,seq):
        if torch.is_grad_enabled() or self.training:
            raise RuntimeError("Private embed is inference only")
        if state["seq"] is not seq:
            a=seq.img_in_joint.detach().cpu()
            b=seq.txt_in_joint.detach().cpu()
            if a.ndim!=1 or b.ndim!=1 or a.dtype!=torch.int64 or b.dtype!=torch.int64:
                raise ValueError("One-dimensional INT64 joint indices required")
            together=torch.cat((a,b)).sort().values
            if not torch.equal(together,torch.arange(seq.n_joint,dtype=torch.int64)):
                raise ValueError("Image/text indices must be unique, disjoint and cover joint storage")
            state["seq"]=seq
        video=self._video_in(self.video_embedder(img.to(self.video_embedder.weight.dtype)))
        words=self.text_embedder(text,seq,self.rope.bands)
        x=video.new_empty(seq.n_joint,video.shape[-1],dtype=self._stream_dtype)
        # x was allocated here, has no aliases/other readers, and both writes
        # use the exact original source dtype conversion/order.
        x.index_copy_(0,seq.img_in_joint,video.to(x.dtype))
        return x.index_copy_(0,seq.txt_in_joint,words.to(x.dtype))
    model.embed=MethodType(embed,model)
    def restore():
        del model.embed
        state.clear()
    return {"mode":"private_embed_index_copy_inplace","source_order_and_dtype_unchanged":True,"index_partition_validated":True},restore
