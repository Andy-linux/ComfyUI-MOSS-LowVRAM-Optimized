"""Backend-aware observations; never selects or replaces an initialized allocator."""

def backend(torch):
    return torch.cuda.memory.get_allocator_backend()


def observation(torch, device):
    name=backend(torch)
    allocated=int(torch.cuda.memory_allocated(device))
    reserved=int(torch.cuda.memory_reserved(device))
    return {"backend":name,"allocated_current_bytes":allocated,
            "reserved_current_bytes":reserved,
            "allocated_peak_bytes":int(torch.cuda.max_memory_allocated(device)),
            "reserved_peak_bytes":int(torch.cuda.max_memory_reserved(device)),
            "stats_scope":"cuda_pool_used_reserved" if name=="cudaMallocAsync" else "pytorch_native_blocks",
            "snapshot_supported":name=="native",
            "fraction_scope":"pytorch_requested_bytes" if name=="cudaMallocAsync" else "native_allocator_reserved_bytes",
            "reserved_is_fraction_capped":name=="native"}


def admission(torch,device,*,external_bytes=0,include_physical=True):
    """Conservative admission estimate, not a physical peak guarantee.

    Async used bytes can include pool users outside PyTorch. Subtracting them
    from its requested-byte limit is conservative. Never enforce reserved<=cap.
    Idle pool backing is only an estimate of reusable/releasable capacity.
    """
    row=observation(torch,device)
    total=int(torch.cuda.get_device_properties(device).total_memory)
    raw_fraction=torch.cuda.get_per_process_memory_fraction(device)
    limit=int(restorable_fraction(raw_fraction)*total)
    free,_=torch.cuda.mem_get_info(device)
    reusable=max(0,row['reserved_current_bytes']-row['allocated_current_bytes'])
    allocator_available=max(0,limit-row['allocated_current_bytes']-int(external_bytes))
    available=min(allocator_available,int(free)+reusable) if include_physical else allocator_available
    return {**row,"fraction_raw":raw_fraction,"fraction_limit_bytes":limit,"external_mapped_bytes":int(external_bytes),
            "available_allocator_bytes":allocator_available,"physical_free_bytes":int(free),
            "reusable_allocator_cache_bytes":reusable,"effective_available_bytes":available,
            "estimate_is_not_measured_peak":True}


def restorable_fraction(raw):
    """Async unset limit may report UINT64_MAX/total (>1), meaning uncapped.

    The setter only accepts [0,1]; restore that uncapped state as 1.0.
    Do not silently accept NaN or negative values.
    """
    import math
    value=float(raw)
    if not math.isfinite(value) or value<0:
        raise ValueError('Invalid allocator fraction: '+repr(raw))
    return min(1.0,value)
