"""Explicit reversible workspace admission for resident-TE VAE decode only.

Unlike an unrestricted profile, this stage deliberately caps the *whole process*
at current allocation + at most 5.5 GiB. It restores the caller's fraction and
owned VAE estimator in finally. No VAE compute or global MM method is patched.
"""
from contextlib import contextmanager

KNOWN_FULL_SHAPES={(64,64),(48,86),(104,78),(50,80)}

@contextmanager
def resident_vae_budget(vae,shape,*,device,extra_gib=5.5,physical_margin_gib=1.0):
    import torch
    if tuple(shape[-2:]) not in KNOWN_FULL_SHAPES or len(shape)!=4 or shape[0]!=1 or shape[1]!=128:
        raise ValueError("Resident VAE workspace budget currently admits measured exact latent shapes only")
    if vae.vae_dtype!=torch.float32 or vae.spacial_compression_decode()!=16 or tuple(torch.cuda.get_device_capability(device)) not in ((8,9),(7,5)):
        raise ValueError("Resident workspace stage currently admits measured SM89/SM75 FP32 MOSS VAE only")
    from .allocator import restorable_fraction
    previous_fraction_raw=torch.cuda.get_per_process_memory_fraction(device)
    previous_fraction=restorable_fraction(previous_fraction_raw)
    total=torch.cuda.get_device_properties(device).total_memory
    base=torch.cuda.memory_allocated(device)
    reserved=torch.cuda.memory_reserved(device)
    physical_free,_=torch.cuda.mem_get_info(device)
    physical_available=physical_free+reserved-base
    extra=min(int(extra_gib*1024**3),max(0,int(physical_available-physical_margin_gib*1024**3)))
    model_bytes=vae.patcher.model_size()
    if extra<=model_bytes:
        raise RuntimeError("Insufficient physical headroom to retain TE for a bounded VAE stage")
    cap_bytes=min(int(previous_fraction*total),base+extra)
    effective_extra=cap_bytes-base
    if effective_extra<=model_bytes:
        raise RuntimeError("Original allocator limit has insufficient effective VAE headroom")
    estimate_cap=effective_extra-model_bytes
    original=vae.memory_used_decode
    def estimate(current_shape,dtype):
        value=original(current_shape,dtype)
        if tuple(current_shape)==tuple(shape) and dtype==torch.float32:
            return min(value,estimate_cap)
        return value
    receipt={"scope":"VAE stage only; whole-process allocator admission, not mathematical kernel",
             "resident_base_allocated_bytes":base,"physical_additional_admission_bytes":extra,
             "additional_budget_bytes":effective_extra,
             "physical_free_bytes":physical_free,"reusable_cached_bytes":reserved-base,
             "physical_margin_bytes":int(physical_margin_gib*1024**3),
             "stage_allocator_cap_bytes":cap_bytes,"previous_fraction":previous_fraction,
             "previous_fraction_raw":previous_fraction_raw,
             "MM_activation_estimate_cap_bytes":estimate_cap,"original_estimate_bytes":int(original(shape,vae.vae_dtype)),
             "estimate_is_not_measured_peak":True,"fraction_restored":False}
    try:
        torch.cuda.set_per_process_memory_fraction(cap_bytes/total,device)
        vae.memory_used_decode=estimate
        yield receipt
    finally:
        try:
            vae.memory_used_decode=original
        finally:
            torch.cuda.set_per_process_memory_fraction(previous_fraction,device)
            receipt["fraction_restored"]=torch.cuda.get_per_process_memory_fraction(device)==previous_fraction
