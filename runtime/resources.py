"""Process-wide admission metadata and reversible per-device allocator budgets."""
from contextlib import contextmanager
from threading import RLock
import weakref
import os
import time

_lock=RLock()
_active={}
BUDGET_GIB={"4g":3.5,"6g":5.5,"8g":7.5,"12g":11.5,"16g":15.5,"unlimited":None}

def host_cache_stats():
    """Process pinned-pool accounting; distinct from active tensor byte sizes."""
    import torch
    getter=getattr(torch.cuda.memory,"host_memory_stats",None)
    if getter is None:return {"supported":False}
    try:stats=getter()
    except Exception as error:return {"supported":False,"error":str(error)}
    return {"supported":True,**{key:value for key,value in stats.items()
            if key.endswith(".current") or key in ("num_host_alloc","num_host_free")}}

def release_idle_host_cache():
    """Release reusable pinned blocks after closing an owned MOSS session.

    PyTorch's host pool is process-wide. Active tensors and unfinished stream
    dependencies remain owned by the allocator; this never frees live buffers.
    Do not call between denoising requests: repinning would defeat streaming.
    """
    import logging
    import torch
    started=time.perf_counter()
    result={"scope":"process idle pinned pool; active allocations preserved"}
    try:
        result["before"]=host_cache_stats()
        public=getattr(getattr(torch,"accelerator",None),"empty_host_cache",None)
        release=public or getattr(torch._C,"_host_emptyCache",None)
        if release is None:
            result["status"]="unavailable"
        else:
            release()
            result["status"]="released_idle"
            result["api"]="torch.accelerator.empty_host_cache" if public else "torch._C._host_emptyCache"
        result["after"]=host_cache_stats()
    except Exception as error:
        # Older Torch builds may not expose host-pool management. Releasing the
        # MOSS objects must still succeed; make the limitation explicit.
        result.update(status="unavailable",error=str(error))
        logging.warning("[KohakuMOSS host cache] %s",error)
    result["wall_s"]=time.perf_counter()-started
    return result

def activate(session):
    """At most one owned model configuration per logical CUDA device."""
    with _lock:
        device=session.config.device
        previous=_active.get(device)
        previous=previous() if previous is not None else None
        if previous is not None and previous is not session:
            previous.close()
        _active[device]=weakref.ref(session)

@contextmanager
def allocator_budget(session,*,operation=None):
    """Normal nodes restore the caller's fraction; benchmarks cap at startup."""
    with _lock:
        activate(session)
        import torch
        device=session.config.device
        from .allocator import observation, restorable_fraction
        original_raw=torch.cuda.get_per_process_memory_fraction(device)
        original=restorable_fraction(original_raw)
        gib=BUDGET_GIB[session.config.profile]
        target=1.0 if gib is None else min(1.0,gib*1024**3/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.set_per_process_memory_fraction(target,device)
        metrics_path=os.environ.get("KOHAKU_MOSS_METRICS_FILE")
        started_epoch=time.time()
        host_start=None
        completed=False
        try:
            if metrics_path:host_start=host_cache_stats()
            if metrics_path:torch.cuda.reset_peak_memory_stats(device)
            yield
            completed=True
        finally:
            try:
                if metrics_path:
                    # Explicit local diagnostics only. No tensor copies, CUDA
                    # synchronization, prompt text or altered inference dispatch.
                    import json
                    import logging
                    from pathlib import Path
                    try:
                        row={"started_epoch":started_epoch,"ended_epoch":time.time(),
                             "operation":operation,"operation_completed":completed,
                             "profile":session.config.profile,"candidate":session.config.candidate,
                             "precision":session.config.precision,"attention":session.config.attention,
                             "last_phase":getattr(session,"current_phase",None),"allocator_budget_GiB":gib,
                             "allocated_peak_bytes":torch.cuda.max_memory_allocated(device),
                             "reserved_peak_bytes":torch.cuda.max_memory_reserved(device),
                             "allocated_end_bytes":torch.cuda.memory_allocated(device),
                             "reserved_end_bytes":torch.cuda.memory_reserved(device)}
                        row["allocator"]=observation(torch,device)
                        row["allocator"]["caller_fraction_raw"]=original_raw
                        row["allocator"]["caller_fraction_restored_as"]=original
                        row["allocator"]["fraction_limit_bytes"]=int(target*torch.cuda.get_device_properties(device).total_memory)
                        row["allocator"]["budget_is_physical_peak_guarantee"]=False
                        vae=getattr(session,"vae",None)
                        if vae is not None:
                            patcher=vae.patcher
                            external=max(0,patcher.loaded_size()-getattr(patcher.model,"model_loaded_weight_memory",0)) if patcher.is_dynamic() else 0
                            row["VAE_loader"]={"class":type(patcher).__name__,"dynamic":patcher.is_dynamic(),
                                "loaded_target_bytes_at_end":patcher.loaded_size(),"external_mapped_bytes_at_end":external}
                        row["VAE_memory_plan"]=getattr(session,"last",{}).get("VAE_memory_plan")
                        row["TE_eviction"]=getattr(session,"last",{}).get("TE_eviction")
                        row["TE_memory_plan"]=getattr(session,"last",{}).get("TE_memory_plan")
                        row["kernel_plan"]=getattr(session,"last",{}).get("kernel_plan")
                        for key in ("NFE","DiT_s","VAE_s","DiT_precision","VAE_precision","VAE_effective_mode",
                                    "DiT_load_s","adapter_s","payload_host_cache","disposal"):
                            row[key]=getattr(session,"last",{}).get(key)
                        te=getattr(session,"last",{}).get("TE") or {}
                        row["TE"]={key:te.get(key) for key in ("request","request_s","token_lengths_positive_negative","layer_weight_passes","weight_h2d_bytes")}
                        payload=getattr(session,"last",{}).get("payload_stream") or {}
                        row["payload_stream"]={key:payload.get(key) for key in ("mode","counter_scope","resident_blocks","payload_H2D_bytes","resident_H2D_bytes","resident_upload_calls","host_pinning","staging_pinned_bytes","placement_s")}
                        row["host_allocator_start"]=host_start
                        row["host_allocator_end"]=host_cache_stats()
                        row["counts"]=dict(getattr(session,"counts",{}))
                        row["torch_num_threads"]=torch.get_num_threads()
                        with Path(metrics_path).open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                    except Exception as error:
                        logging.warning("[KohakuMOSS metrics] %s",error)
            finally:
                torch.cuda.set_per_process_memory_fraction(original,device)

def deactivate(session):
    with _lock:
        reference=_active.get(session.config.device)
        if reference is not None and reference() is session:
            _active.pop(session.config.device,None)
