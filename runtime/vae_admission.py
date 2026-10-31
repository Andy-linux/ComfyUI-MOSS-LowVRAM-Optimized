"""Per-VAE scheduling estimates from independent, full-image SM89 measurements.

This only changes an owned model's memory request to Comfy's normal manager.
Unknown shapes/runtime retain the original estimate. Allocation limits never
increase. VAE compute code and explicit backend settings are not modified;
workspace availability and official OOM/tiled fallback can still change results.
"""
import math
from packaging.version import Version

PEAKS={
    "fp32":{(64,64):3106685440,(104,78):5658270208},
    "bf16":{(64,64):1856165888,(104,78):3551098368},
}
MODEL_BYTES={"fp32":249493396,"bf16":124747214}

def target_model_bytes(model):
    """Logical target storage; Dynamic keeps source dtype in its state dict.

    Mirror the dtype geometry selected by ModelPatcherDynamic.setup_param.
    This does not modify Comfy's source-size accounting or VBAR allocation.
    Alignment and transient casts remain covered by the workspace margin and
    must also pass actual Dynamic end-to-end tests.
    """
    result=0
    for name,tensor in model.state_dict().items():
        parent,_,key=name.rpartition('.')
        module=model.get_submodule(parent) if parent else model
        dtype=getattr(module,key+"_comfy_model_dtype",None) or tensor.dtype
        result+=tensor.numel()*dtype.itemsize
    return result

def install(vae,*,device,precision,artifact_name):
    import torch
    from comfy.ldm.models.autoencoder import AutoencoderKL
    original=vae.memory_used_decode
    model=vae.first_stage_model
    target_bytes=target_model_bytes(model)
    source_bytes=vae.patcher.model_size()
    qualified=(tuple(torch.cuda.get_device_capability(device))==(8,9)
               and Version("2.10") <= Version(torch.__version__) < Version("2.11")
               and 91200 <= torch.backends.cudnn.version() < 91300
               and artifact_name=="moss-image-vae.safetensors"
               and type(model) is AutoencoderKL and model.bn is not None and list(model.ps)==[2,2]
               and vae.latent_channels==128
               and target_bytes==MODEL_BYTES[precision]
               and (precision!="bf16" or getattr(vae,"_moss_precision",{}).get("decoder")=="torch.bfloat16")
               and vae.vae_dtype==torch.float32 and vae.spacial_compression_decode()==16)
    vae._moss_admission_qualification={"qualified":qualified,"source_model_bytes":source_bytes,
        "target_model_bytes":target_bytes,"patcher_class":type(vae.patcher).__name__,
        "target_size_uses_archived_dtypes":True,"dynamic":vae.patcher.is_dynamic()}
    if qualified:vae._moss_admission_target_bytes=target_bytes
    records=[]
    def estimate(shape,dtype):
        key=tuple(shape[-2:])
        if not qualified or dtype!=torch.float32 or len(shape)!=4 or shape[0]!=1 or shape[1]!=128 or key not in PEAKS[precision]:
            return original(shape,dtype)
        peak=PEAKS[precision][key]
        # Independent peak already includes the full model. MM adds its model
        # bytes separately; subtract exactly once and reserve workspace margin.
        model=target_bytes
        result=math.ceil(max(0,peak-model)*1.05)+32*1024**2
        if not any(x["latent_HW"]==list(key) for x in records):
            records.append({"latent_HW":list(key),"decoder":precision,
                            "independent_full_peak_bytes":peak,"model_bytes":model,
                            "source_model_bytes":source_bytes,"patcher_class":type(vae.patcher).__name__,
                            "workspace_factor":1.05,"fixed_margin_bytes":32*1024**2,
                            "activation_estimate_bytes":result,
                            "source":"results/vae-"+precision+"-16g-v2/receipt.json; "+("q17" if key==(64,64) else "test_00207")+" full decode repeats 0..3 maximum",
                            "estimate_is_not_a_measured_combined_peak":True})
        return result
    vae.memory_used_decode=estimate
    return records
