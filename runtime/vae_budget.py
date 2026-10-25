"""Opt-in capacity estimates from strict full-FP32 same-runtime measurements.

This changes the owning VAE's scheduling estimate only, never VAE compute.
Unknown shapes retain upstream estimates. Observed maxima are not predictions.
"""
import math

# Exact latent H/W. Peaks include the VAE itself, so using them as activation
# estimates with 20% margin plus MM's model-size term is deliberately conservative.
FULL_FP32_PEAKS={
    (64,64):(3121230336,"sm89-4g-liveness-v1/q17"),
    (48,86):(3144512000,"sm89-4g-liveness-v1/test_00200"),
    (104,78):(5672172032,"sm89-auto-new3-v1/test_00207"),
    (50,80):(3055798784,"sm89-4g-balcony-v1/balcony"),
}

def install(vae,*,device):
    import torch
    if tuple(torch.cuda.get_device_capability(device))!=(8,9) or not torch.__version__.startswith("2.10."):
        raise ValueError("Qualified VAE estimate table requires measured SM89/Torch2.10 runtime")
    if vae.vae_dtype!=torch.float32 or vae.spacial_compression_decode()!=16:
        raise ValueError("Qualified full FP32 MOSS VAE with spatial compression16 required")
    original=vae.memory_used_decode
    applied=[]
    def estimate(shape,dtype):
        key=tuple(shape[-2:])
        if dtype!=torch.float32 or len(shape)!=4 or shape[0]!=1 or shape[1]!=128 or key not in FULL_FP32_PEAKS:
            return original(shape,dtype)
        peak,source=FULL_FP32_PEAKS[key]
        value=math.ceil(peak*1.2)
        if not any(row["latent_HW"]==list(key) for row in applied):
            applied.append({"latent_HW":list(key),"observed_independent_full_peak_bytes":peak,
                            "source_run_case":source,"safety_factor":1.2,"estimated_bytes":value,
                            "scope":"schedule estimate, not a new measured peak"})
        return value
    vae.memory_used_decode=estimate
    return applied
