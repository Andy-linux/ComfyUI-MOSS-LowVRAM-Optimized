"""Promote a loaded ring encoder to resident decoder layers with CPU embedding."""
import time
from dataclasses import replace

def promote_resident(encoder):
    """One weight upload; preserve stored quantization and original host lookup.

    Start from a ring load so the embedding never makes a temporary GPU trip.
    No old ring pass may be active. The old empty streamer is garbage collected;
    release() would destroy the model's shared weight Tensor objects, so is not
    called on this transferred placement.
    """
    from kohakumoss.offload.encoders.text import TextPlacement
    from kohakumoss.offload.weights.layers import module_tensors
    class ResidentCPUEmbedding(TextPlacement):
        name="resident_cpu_embedding"

        def place(self):
            for tensor in module_tensors(self.model).values():
                if id(tensor) not in self.emb_ids:
                    self.to_device(tensor)

        def embed(self,ids):
            return self.lookup(ids,self.device)

        def resident_bytes(self):
            return sum(tensor.nbytes for tensor in module_tensors(self.model).values()
                       if tensor.device.type=="cuda")

    if encoder.plan.memory!="ring":
        raise ValueError("Resident promotion requires the original CPU-backed ring load")
    old=encoder.placement
    started=time.perf_counter()
    placement=ResidentCPUEmbedding(encoder.model,list(encoder.model.layers),
                  encoder.model.embed_tokens,encoder.device,pinned=False,host="file",
                  dtypes=old.dtypes,deferred=old.deferred)
    encoder.placement=placement
    encoder.forward.placement=placement
    encoder.plan=replace(encoder.plan,memory="resident")
    import torch
    torch.cuda.synchronize(encoder.device)
    assert all(t.device.type=="cpu" for t in module_tensors(encoder.model.embed_tokens).values())
    return {"promotion_s":time.perf_counter()-started,
            "decoder_resident_bytes":placement.resident_bytes(),
            "embedding_device":"cpu","CPU_embedding_lookup_is_not_full_TE":True,
            "placement":"resident_cpu_embedding","weight_H2D_per_subsequent_encode":0}


def promote_partial(encoder,*,max_bytes):
    """Keep a byte-bounded decoder prefix, stream the remaining original layers.

The kernel forward, full paired token segments, dtypes and all weight values are
unchanged. Embedding stays on CPU. The suffix uses the original LayerStreamer,
including its separate pools for GDN and attention signatures.
"""
    import torch
    from functools import partial
    from kohakumoss.offload.encoders.text import TextPlacement
    from kohakumoss.offload.weights.layers import LayerStreamer,module_tensors,device_dtype,tensor_ids
    from kohakumoss_common.calls import nothing
    old=encoder.placement
    if encoder.plan.memory!="ring" or old.streamer.is_open:
        raise ValueError("Partial TE placement requires an idle, original CPU-backed ring")
    if torch.cuda.get_device_capability(encoder.device)!=(8,9):
        raise ValueError("Partial TE residency is currently qualified only for SM89")
    layers=list(encoder.model.layers)
    sizes=[]
    for layer in layers:
        sizes.append(sum(t.numel()*torch.empty((),device='meta',dtype=device_dtype(t,old.dtypes)).element_size()
                         for t in module_tensors(layer).values()))
    count=0;used=0
    for size in sizes:
        if used+size>max_bytes:break
        count+=1;used+=size
    if not 0<count<len(layers):
        raise ValueError("Partial TE budget must fit at least one but fewer than all decoder layers")
    class PartialCPUEmbedding(TextPlacement):
        name="partial_cpu_embedding"
        def place(self):
            suffix=self.layers[count:]
            excluded=tensor_ids(suffix)|self.emb_ids
            self.kept_ids=frozenset(id(t) for t in module_tensors(self.model).values() if id(t) not in excluded)
            for tensor in module_tensors(self.model).values():
                if id(tensor) in self.kept_ids:self.to_device(tensor)
            self.streamer=LayerStreamer(suffix,self.device,slots=2,pinned=False,hooks=False,host="file",
                                        dtypes=self.dtypes,deferred=self.deferred)
            self.enter=[nothing]*count+[partial(self.streamer.enter,i) for i in range(len(suffix))]
            self.leave=[nothing]*count+[partial(self.streamer.leave,i) for i in range(len(suffix))]
        def embed(self,ids):return self.lookup(ids,self.device)
        def begin(self):self.streamer.open()
        def end(self):
            self.streamer.close();self.h2d_bytes=self.streamer.h2d_bytes
        def resident_bytes(self):
            return sum(t.nbytes for t in module_tensors(self.model).values() if id(t) in self.kept_ids and t.device.type=='cuda')
        def pass_bytes(self):return self.resident_bytes()+self.streamer.slot_bytes()
        def release(self):
            self.streamer.close();self.streamer.remove();super().release()
    started=time.perf_counter()
    # Removing old hooks/references preserves the shared module tensors. Calling
    # old.release() here would destroy those very tensors and is forbidden.
    old.streamer.remove()
    placement=PartialCPUEmbedding(encoder.model,layers,encoder.model.embed_tokens,encoder.device,
                                  pinned=False,host="file",dtypes=old.dtypes,deferred=old.deferred)
    encoder.placement=encoder.forward.placement=placement
    encoder.plan=replace(encoder.plan,memory="partial")
    torch.cuda.synchronize(encoder.device)
    return {"promotion_s":time.perf_counter()-started,"placement":placement.name,
            "prefix_layers":count,"total_layers":len(layers),"prefix_budget_bytes":max_bytes,
            "prefix_weight_bytes":used,"decoder_resident_bytes":placement.resident_bytes(),
            "suffix_ring_slot_bytes":placement.streamer.slot_bytes(),"embedding_device":"cpu",
            "weight_H2D_per_subsequent_encode":sum(placement.streamer.layer_bytes),
            "arithmetic_changed":False,"original_full_positive_negative":True}
