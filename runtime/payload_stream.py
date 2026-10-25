"""Original integer payloads: bounded CPU arena, resident prefix, two GPU slots.

Only storage and transfer timing change. All FP32 scales and arithmetic remain
owned by the original frozen bases. No projection hook or new quantizer exists.
"""
import time
import torch

PROJECTIONS=('q','k','v','o','ffn_up_gate','ffn_down')
HOST_PIN_BUDGET_BYTES=int(2.5*1024**3)
FP8_SIDE_BUFFER_BUDGET_BYTES=256*1024**2


def _fp8_description(base):
    """Return FP8 format identity and resident scale/aux bytes, or None.

    Deliberately duck-check only after the exact vendor class test. Training
    transpose buffers are not valid inference payloads and are never hidden.
    """
    from kohakumoss.quant.compute import ComputeWeight
    from kohakumoss.quant.weights import QuantWeight
    if isinstance(base,ComputeWeight):
        if any(getattr(base,n,None) is not None for n in ('q_t','scale_t','aux_t')):
            raise ValueError('Training transpose buffers cannot enter FP8 payload streaming')
        recipe=base.recipe
        parts={n:t for n,t in base.tensors().items() if n!='q'}
    elif isinstance(base,QuantWeight) and base.fmt=='fp8':
        recipe='weight_only';parts={'scale':base.scale}
    else:return None
    if base.q.dtype!=torch.float8_e4m3fn or base.q.ndim!=2:
        raise ValueError('Original E4M3 FP8 matrix payload required')
    signature=(type(base).__name__,recipe,getattr(base,'rotation',0),tuple(base.q.shape),
               tuple((n,tuple(t.shape),t.dtype) for n,t in sorted(parts.items())))
    return signature,sum(t.numel()*t.element_size() for t in parts.values())


def _bind_fp8_payload(bb,cache,device):
    """Called only after q aliases and model-owned scale buffers are placed."""
    if not cache.fp8_signature:return
    from kohakumoss.quant.compute import ComputeWeight
    dtype=bb.video_embedder.weight.dtype
    for i,name,attr,shape,stored_dtype,start,size in cache.layout:
        base=bb.layers[i].proj[name].base
        if isinstance(base,ComputeWeight):base.bind(dtype,device,autograd=False)
        else:base.bind(dtype,device)


def _fp8_stats(cache):
    if not cache.fp8_signature:return {}
    return {'payload_format':'FP8_E4M3','resident_scale_aux_GPU_bytes':cache.side_buffer_bytes,
            'scale_aux_budget_bytes':cache.side_buffer_budget_bytes,
            'scale_aux_policy':'normal model buffers; excluded from streamed byte arena',
            'fp8_recipes':sorted({entry[1][1] for entry in cache.fp8_signature}),
            'training_transpose_allocated':False}

class _PinnedChunks:
    """Power-of-two pieces avoid doubling a suffix just over a pool boundary.

    This owns CPU copies only. Consumers copy intersections directly into the
    existing GPU destination; no concatenation or extra GPU buffer is created.
    """
    def __init__(self,source):
        if source.device.type!='cpu' or source.dtype!=torch.uint8 or source.ndim!=1:
            raise ValueError('Flat original CPU byte payload required')
        self.size=source.numel();self.chunks=[];offset=0;chunk=None
        try:
            while offset<self.size:
                remaining=self.size-offset
                size=min(2*1024**3,1<<(remaining.bit_length()-1))
                chunk=torch.empty(size,dtype=torch.uint8,device='cpu',pin_memory=True)
                chunk.copy_(source[offset:offset+size])
                self.chunks.append((offset,chunk));offset+=size
        except BaseException:
            # Tracebacks may outlive a failed request. Do not let a partially
            # constructed owner keep its already allocated pinned pieces alive.
            self.chunks.clear();chunk=None
            raise

    def numel(self):return self.size


def _copy_pinned_slice(destination,source,source_offset=0):
    """Enqueue the same byte interval on the caller's current CUDA stream."""
    size=destination.numel()
    if destination.dtype!=torch.uint8 or destination.ndim!=1 or source_offset<0 or source_offset+size>source.numel():
        raise ValueError('Pinned payload copy interval is invalid')
    if isinstance(source,torch.Tensor):
        destination.copy_(source[source_offset:source_offset+size],non_blocking=True)
        return 1
    if not isinstance(source,_PinnedChunks):raise TypeError('Unknown pinned payload storage')
    copied=0;calls=0
    for start,chunk in source.chunks:
        lo=max(start,source_offset);hi=min(start+chunk.numel(),source_offset+size)
        if hi<=lo:continue
        destination[lo-source_offset:hi-source_offset].copy_(chunk[lo-start:hi-start],non_blocking=True)
        copied+=hi-lo;calls+=1
    if copied!=size:raise RuntimeError('Pinned payload pieces did not cover the destination')
    return calls


def _pinned_geometry(source):
    if isinstance(source,_PinnedChunks):
        return {'pinned_chunks':len(source.chunks),'pinned_allocation_rounding_estimate_bytes':source.numel()}
    size=source.numel()
    return {'pinned_chunks':1,'pinned_allocation_rounding_estimate_bytes':1<<(size-1).bit_length() if size else 0}


class PayloadCache:
    """One checkpoint, one CPU byte arena. Survives disposable 4G requests."""
    def __init__(self):
        self.identity=None;self.arena=None;self.layout=None;self.block_bytes=0
        self.hits=0;self.misses=0;self.capture_s=0.0
        self.pinned=None;self.pin_start=None;self.pin_s=0.0
        self.fp8_signature=();self.side_buffer_bytes=0
        self.side_buffer_budget_bytes=FP8_SIDE_BUFFER_BUDGET_BYTES

    def capture(self,bb,identity,*,fp8_side_buffer_budget_bytes=FP8_SIDE_BUFFER_BUDGET_BYTES):
        from .storage import FrozenPackedBase,FrozenInt8Base
        layout=[];offset=0;block_sizes=[];fp8=[];side_bytes=0
        for i,block in enumerate(bb.layers):
            start=offset
            for name in PROJECTIONS:
                base=block.proj[name].base
                attr='qweight' if type(base) is FrozenPackedBase else 'codes' if type(base) is FrozenInt8Base else None
                if attr is None:
                    description=_fp8_description(base)
                    if description is not None:
                        signature,side=description;attr='q'
                        fp8.append((name,signature));side_bytes+=side
                if attr is None:raise TypeError('Canonical frozen payload required')
                tensor=getattr(base,attr)
                if tensor.device.type!='cpu' or not tensor.is_contiguous() or tensor.element_size()!=1:
                    raise ValueError('Original contiguous CPU integer payload required')
                size=tensor.numel()
                layout.append((i,name,attr,tuple(tensor.shape),tensor.dtype,offset,size))
                offset+=size
            block_sizes.append(offset-start)
        if len(set(block_sizes))!=1 or len(bb.layers)!=42:raise ValueError('42 equal-size SFT blocks required')
        if fp8:
            if len(fp8)!=42*len(PROJECTIONS):
                raise ValueError('FP8 streaming requires all 42 x 6 joint projections; use full model placement')
            first=fp8[:len(PROJECTIONS)]
            if any(fp8[i:i+len(PROJECTIONS)]!=first for i in range(0,len(fp8),len(PROJECTIONS))):
                raise ValueError('FP8 joint blocks must have identical recipe/scale layouts')
            if side_bytes>fp8_side_buffer_budget_bytes:
                raise ValueError(f'FP8 resident scale/aux buffers need {side_bytes} bytes, exceeding '
                                 f'{fp8_side_buffer_budget_bytes}; use full model placement or explicitly budget them')
        signature=tuple(fp8)
        hit=self.identity==identity and self.layout==layout and self.arena is not None and self.fp8_signature==signature
        if hit:self.hits+=1
        else:
            self.close();started=time.perf_counter()
            arena=torch.empty(offset,dtype=torch.uint8,device='cpu')
            for i,name,attr,shape,dtype,start,size in layout:
                source=getattr(bb.layers[i].proj[name].base,attr)
                arena[start:start+size].copy_(source.view(torch.uint8).reshape(-1))
            self.arena,self.identity,self.layout,self.block_bytes=arena,identity,layout,block_sizes[0]
            self.misses+=1;self.capture_s=time.perf_counter()-started
        self.fp8_signature=signature;self.side_buffer_bytes=side_bytes
        self.side_buffer_budget_bytes=fp8_side_buffer_budget_bytes
        return hit

    def close(self):
        self.identity=self.arena=self.layout=None;self.block_bytes=0
        self.pinned=None;self.pin_start=None
        self.fp8_signature=();self.side_buffer_bytes=0

    def pinned_suffix(self,start):
        """Immutable host suffix only; never pin the whole checkpoint implicitly."""
        size=(42-start)*self.block_bytes
        if size>HOST_PIN_BUDGET_BYTES:
            raise ValueError('Pinned suffix exceeds its 2.5 GiB host budget; use staging or more resident blocks')
        hit=self.pinned is not None and self.pin_start==start
        if not hit:
            began=time.perf_counter()
            self.pinned=None
            source=self.arena[start*self.block_bytes:]
            if size and (1<<(size-1).bit_length())>HOST_PIN_BUDGET_BYTES:
                pinned=_PinnedChunks(source)
            else:
                pinned=torch.empty(size,dtype=torch.uint8,device='cpu',pin_memory=True)
                pinned.copy_(source)
            self.pinned,self.pin_start=pinned,start;self.pin_s=time.perf_counter()-began
        return self.pinned,hit


class FrozenPayloadResident:
    """One immutable host arena, one full GPU slab, original integer views.

    The W8/6G lifecycle repeatedly rebuilds its core. This owner avoids 252
    separate code uploads without adding streamed transfers or block hooks.
    Optional suffix pinning retains the existing 2.5 GiB host-pinning bound;
    the full code payload is never implicitly pinned.
    Restoring the existing CPU originals before MM offload avoids copying
    immutable codes back from the GPU. Scales remain normal model buffers.
    """
    def __init__(self,bb,cache,device,*,pinning='staging'):
        self.bb,self.cache,self.device=bb,cache,torch.device(device)
        if pinning not in ('staging','suffix'):raise ValueError(pinning)
        self.pinning=pinning
        if self.device.type!='cuda' or self.device.index is None:raise ValueError('Explicit CUDA device required')
        if cache.arena is None or cache.arena.numel()>5*1024**3:raise ValueError('Resident host arena exceeds its 5 GiB budget')
        if any(not (attr=='codes' and dtype==torch.int8 or attr=='q' and dtype==torch.float8_e4m3fn)
               for _,_,attr,_,dtype,_,_ in cache.layout):
            raise ValueError('This resident arena is qualified for original W8 INT8 codes only')
        self.closed=False;self.slab=None
        self.stats={'mode':'full_resident_arena','resident_blocks':42,'streamed_blocks':0,
                    'resident_GPU_bytes':cache.arena.numel(),'host_arena_bytes':cache.arena.numel(),
                    'slot_GPU_bytes':0,'slots':0,'host_pinning':'pageable_arena',
                    'staging_pinned_bytes':0,'payload_H2D_bytes':0,'stream_H2D_calls':0,
                    'resident_H2D_bytes':0,'resident_upload_calls':0,'forwards':0,'planned_block_calls':0,
                    'arithmetic_changed':False,'FP32_scales_moved_with_payload':False,'block_hooks':0}
        self.stats.update(_fp8_stats(cache))

    def place(self):
        started=time.perf_counter()
        try:
            for i,name,attr,shape,dtype,start,size in self.cache.layout:
                self.bb.layers[i].proj[name].base._buffers[attr]=torch.empty(0,dtype=dtype)
            self.bb.to(self.device)
            if self.pinning=='suffix':
                blocks=min(42,HOST_PIN_BUDGET_BYTES//self.cache.block_bytes)
                start=42-blocks;split=start*self.cache.block_bytes
                suffix,hit=self.cache.pinned_suffix(start)
                self.slab=torch.empty_like(self.cache.arena,device=self.device)
                if split:self.slab[:split].copy_(self.cache.arena[:split],non_blocking=False)
                calls=_copy_pinned_slice(self.slab[split:],suffix)
                self.stats.update(host_pinning='bounded_suffix',staging_pinned_bytes=suffix.numel(),
                                  pinned_suffix_cache_hit=hit,pinned_suffix_prepare_s=0.0 if hit else self.cache.pin_s,
                                  resident_upload_calls=calls+bool(split),**_pinned_geometry(suffix))
            else:
                self.slab=self.cache.arena.to(self.device,non_blocking=False)
                self.stats['resident_upload_calls']=1
            for i,name,attr,shape,dtype,start,size in self.cache.layout:
                self.bb.layers[i].proj[name].base._buffers[attr]=self.slab[start:start+size].view(dtype).view(shape)
            _bind_fp8_payload(self.bb,self.cache,self.device)
            self.bb._moss_payload_owner=self
            torch.cuda.current_stream(self.device).synchronize()
            self.stats.update(resident_H2D_bytes=self.slab.numel(),
                              placement_s=time.perf_counter()-started)
            return self
        except BaseException:
            self.close();raise

    def begin_forward(self,skip_layers):
        if self.closed:raise RuntimeError('Resident payload owner was unloaded; rebuild before use')
        self.stats['forwards']+=1
        self.stats['planned_block_calls']+=42-len(set(skip_layers))

    def receipt(self):return dict(self.stats,closed=self.closed,counter_scope='owner_lifetime')

    def close(self):
        if self.closed:return
        torch.cuda.current_stream(self.device).synchronize()
        for i,name,attr,shape,dtype,start,size in self.cache.layout:
            self.bb.layers[i].proj[name].base._buffers[attr]=self.cache.arena[start:start+size].view(dtype).view(shape)
        self.slab=None;self.closed=True


class FrozenPayloadStream:
    """Per-model owner. Fixed aliases; copies overlap a complete block's compute.

begin_forward receives the *actual* skipped-layer set from the Comfy bridge.
If a skipped layer makes consecutive blocks share a slot, that prefetch waits
until the current block post-hook. CPU staging is never reused before H2D ends.
"""
    def __init__(self,bb,cache,device,*,resident_blocks,pinning='staging'):
        if not 0<=resident_blocks<=40:raise ValueError('Streaming prefix must be 0..40 blocks')
        if pinning not in ('staging','suffix'):raise ValueError(pinning)
        self.bb,self.cache,self.device=bb,cache,torch.device(device)
        self.pinning=pinning;self.host_suffix=None
        self.resident_blocks=resident_blocks;self.block_bytes=cache.block_bytes
        self.closed=False;self.hooks=[];self.resident=None;self.slots=[];self.staging=[]
        self.loaded=[None,None];self.ready=[torch.cuda.Event(),torch.cuda.Event()]
        self.free=[torch.cuda.Event(),torch.cuda.Event()];self.ready_recorded=[False,False];self.free_recorded=[False,False]
        self.copy_stream=torch.cuda.Stream(device=self.device)
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.next={};self.current=None;self.delayed=None
        self.stats={'resident_blocks':resident_blocks,'streamed_blocks':42-resident_blocks,
                    'block_payload_bytes':self.block_bytes,'slots':2,'slot_GPU_bytes':2*self.block_bytes,
                    'resident_GPU_bytes':resident_blocks*self.block_bytes,'host_arena_bytes':cache.arena.numel(),
                    'staging_pinned_bytes':2*self.block_bytes,'payload_H2D_bytes':0,'stream_H2D_calls':0,
                    'resident_H2D_bytes':0,'host_staging_copy_s':0.0,'host_staging_wait_s':0.0,
                    'forwards':0,'block_calls':0,'skipped_prefetches_avoided':0,
                    'arithmetic_changed':False,'FP32_scales_moved_with_payload':False}
        self.stats.update(_fp8_stats(cache))

    def place(self):
        """Call before any full-model CUDA move or fused installer."""
        started=time.perf_counter()
        try:
            for i,name,attr,shape,dtype,start,size in self.cache.layout:
                base=self.bb.layers[i].proj[name].base
                base._buffers[attr]=torch.empty(0,dtype=dtype)
            self.bb.to(self.device)
            if self.resident_blocks:
                size=self.resident_blocks*self.block_bytes
                self.resident=torch.empty(size,device=self.device,dtype=torch.uint8)
                self.resident.copy_(self.cache.arena[:size])
                self.stats['resident_H2D_bytes']=size
            self.slots=[torch.empty(self.block_bytes,device=self.device,dtype=torch.uint8) for _ in range(2)]
            if self.pinning=='suffix':
                self.host_suffix,hit=self.cache.pinned_suffix(self.resident_blocks)
                self.stats.update(host_pinning='immutable_suffix',staging_pinned_bytes=self.host_suffix.numel(),
                                  pinned_suffix_cache_hit=hit,pinned_suffix_prepare_s=0.0 if hit else self.cache.pin_s,
                                  **_pinned_geometry(self.host_suffix))
            else:
                self.staging=[torch.empty(self.block_bytes,device='cpu',dtype=torch.uint8,pin_memory=True) for _ in range(2)]
                self.stats['host_pinning']='two_staging_buffers'
            # Newly allocated default-stream buffers cannot be consumed on the
            # copy stream before their allocation dependencies are complete.
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            for i,name,attr,shape,dtype,start,size in self.cache.layout:
                storage=self.resident if i<self.resident_blocks else self.slots[self._slot(i)]
                offset=start if i<self.resident_blocks else start-i*self.block_bytes
                self.bb.layers[i].proj[name].base._buffers[attr]=storage[offset:offset+size].view(dtype).view(shape)
            _bind_fp8_payload(self.bb,self.cache,self.device)
            self.bb._moss_payload_owner=self
            for i,block in enumerate(self.bb.layers):
                self.hooks.append(block.register_forward_pre_hook(lambda module,args,_i=i:self.before(_i)))
                self.hooks.append(block.register_forward_hook(lambda module,args,result,_i=i:self.after(_i),always_call=True))
            torch.cuda.current_stream(self.device).synchronize()
            self.stats['placement_s']=time.perf_counter()-started
            return self
        except BaseException:
            self.close();raise

    def _slot(self,i):return (i-self.resident_blocks)%2

    def _load(self,i):
        if i is None or i<self.resident_blocks:return
        slot=self._slot(i)
        if self.loaded[slot]==i:return
        if self.host_suffix is not None:
            # Immutable inputs need no CPU wait before another upload. GPU slot
            # reuse is still ordered by its last compute event on copy_stream.
            lo=(i-self.resident_blocks)*self.block_bytes
            source=self.host_suffix
        else:
            started=time.perf_counter()
            if self.ready_recorded[slot]:self.ready[slot].synchronize()
            self.stats['host_staging_wait_s']+=time.perf_counter()-started
            started=time.perf_counter();lo=i*self.block_bytes
            self.staging[slot].copy_(self.cache.arena[lo:lo+self.block_bytes])
            self.stats['host_staging_copy_s']+=time.perf_counter()-started
            source=self.staging[slot]
        with torch.cuda.stream(self.copy_stream):
            if self.free_recorded[slot]:self.copy_stream.wait_event(self.free[slot])
            if self.host_suffix is not None:
                copies=_copy_pinned_slice(self.slots[slot],source,lo)
            else:
                self.slots[slot].copy_(source,non_blocking=True);copies=1
            self.ready[slot].record(self.copy_stream)
        self.ready_recorded[slot]=True;self.loaded[slot]=i
        self.stats['payload_H2D_bytes']+=self.block_bytes;self.stats['stream_H2D_calls']+=copies

    def begin_forward(self,skip_layers):
        if self.closed:raise RuntimeError('Payload owner was unloaded; rebuild the session model')
        if self.current is not None:raise RuntimeError('Concurrent/nested streamed forward refused')
        plan=[i for i in range(42) if i not in skip_layers]
        self.next={i:(plan[j+1] if j+1<len(plan) else None) for j,i in enumerate(plan)}
        suffix=[i for i in plan if i>=self.resident_blocks]
        # The previous forward ends at layer41. Force each new forward to own
        # its current generation; do not mistake a slot's old label for a cache.
        self.loaded=[None,None];self.delayed=None
        for i in suffix[:2]:
            if i!=suffix[0] and self._slot(i)==self._slot(suffix[0]):break
            self._load(i)
        self.stats['forwards']+=1
        self.stats['skipped_prefetches_avoided']+=sum(i>=self.resident_blocks for i in skip_layers)

    def before(self,i):
        if self.closed or i not in self.next:raise RuntimeError('Missing original forward plan for payload stream')
        if self.current is not None:raise RuntimeError('Concurrent streamed blocks refused')
        self.current=i;self.stats['block_calls']+=1
        if i<self.resident_blocks:return
        self._load(i);slot=self._slot(i)
        torch.cuda.current_stream(self.device).wait_event(self.ready[slot])
        nxt=self.next[i]
        if nxt is not None:
            if self._slot(nxt)==slot:self.delayed=nxt
            else:self._load(nxt)

    def after(self,i):
        if self.closed:return
        if self.current!=i:return  # pre-hook failure: close() handles ownership
        if i>=self.resident_blocks:
            slot=self._slot(i);self.free[slot].record(torch.cuda.current_stream(self.device));self.free_recorded[slot]=True
        self.current=None
        if self.delayed is not None:
            nxt=self.delayed;self.delayed=None;self._load(nxt)

    def receipt(self):return dict(self.stats,closed=self.closed,counter_scope='owner_lifetime')

    def close(self):
        if self.closed:return
        self.copy_stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()
        for hook in self.hooks:hook.remove()
        self.hooks=[]
        # Restore correct CPU codes before standard .to(cpu/meta). Shared GPU
        # slots must never be copied as if each logical layer owned their data.
        for i,name,attr,shape,dtype,start,size in self.cache.layout:
            self.bb.layers[i].proj[name].base._buffers[attr]=self.cache.arena[start:start+size].view(dtype).view(shape)
        self.slots=[];self.staging=[];self.host_suffix=None;self.resident=None;self.closed=True
