"""Optional BF16 decoder with original FP32 latent normalization statistics.

The Comfy VAE public input dtype remains FP32 so its full/tiled dispatcher does
not round latents before de-normalization. Convolution weights and decoder
activations are BF16; its memory estimator is passed that actual compute dtype.
This is an explicitly different pixel path, never a claim of RGB equivalence.
"""

from types import MethodType
import torch


def _decode(self, z, **decoder_kwargs):
    from einops import rearrange
    import comfy.model_management as mm

    var = mm.cast_to(
        self.bn.running_var.view(1, -1, 1, 1), dtype=torch.float32, device=z.device
    )
    mean = mm.cast_to(
        self.bn.running_mean.view(1, -1, 1, 1), dtype=torch.float32, device=z.device
    )
    z = z.float() * torch.sqrt(var + self.bn_eps) + mean
    z = rearrange(
        z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=self.ps[0], pj=self.ps[1]
    )
    z = z.to(torch.bfloat16)
    return self.decoder(self.post_quant_conv(z), **decoder_kwargs)


def load_fp32_vae_patcher(vae_path, metadata=None, device=None, disable_dynamic=False):
    """Comfy cached VAE factory with an explicit FP32 numerical contract.

    Match load_vae_patcher's device slot and disable_dynamic keyword. Comfy0.41
    accepts but does not forward disable_dynamic to VAE; its outer clone path
    owns that policy. This VAE constructor has no disable_dynamic parameter.
    """
    import comfy.sd
    import comfy.utils

    if metadata is None:
        state, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    else:
        state = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=state, metadata=metadata, device=device, dtype=torch.float32)
    vae.throw_exception_if_invalid()
    return vae.patcher


def load_bf16_vae(name):
    import folder_paths
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    from comfy.ldm.models.autoencoder import AutoencoderKL
    from pathlib import Path

    path = (
        str(Path(name).resolve())
        if Path(name).is_file()
        else folder_paths.get_full_path_or_raise("vae", name)
    )
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    if "bn.running_mean" not in state or "bn.running_var" not in state:
        raise ValueError(
            "BF16 decoder option requires the original MOSS/Flux2 batch-normalized VAE"
        )
    # Capture before the loader casts parameters/buffers to its compute dtype.
    mean = state["bn.running_mean"].float().clone()
    var = state["bn.running_var"].float().clone()
    vae = comfy.sd.VAE(sd=state, metadata=metadata, dtype=torch.bfloat16)
    vae.throw_exception_if_invalid()
    model = vae.first_stage_model
    if (
        type(model) is not AutoencoderKL
        or model.bn is None
        or list(model.ps) != [2, 2]
        or model.max_batch_size is not None
        or vae.latent_channels != 128
    ):
        raise ValueError(
            "BF16 decoder option requires the exact batch-one MOSS VAE architecture"
        )
    model.bn.running_mean = mean
    model.bn.running_var = var
    model.decode = MethodType(_decode, model)
    # Preserve the constructor's BF16 target archives, including dynamic
    # loaders whose assigned checkpoint tensors still have their source dtype.
    mm.archive_model_dtypes(model.bn)
    original_estimate = vae.memory_used_decode
    vae.memory_used_decode = lambda shape, dtype: original_estimate(
        shape, torch.bfloat16
    )
    vae.vae_dtype = torch.float32  # input boundary; model weights stay BF16
    vae.size = None
    vae.patcher.size = 0

    def decode_only(*args, **kwargs):
        raise RuntimeError(
            "This MOSS BF16 VAE instance is decode-only; use the normal VAE loader for encoding"
        )

    vae.encode = decode_only
    vae.encode_tiled = decode_only
    vae._moss_precision = {
        "decoder": "torch.bfloat16",
        "latent_denormalization": "torch.float32",
        "normalization_statistics": "original FP32 checkpoint buffers",
        "public_input_dtype": "torch.float32",
        "output_postprocess": "original Comfy FP32",
        "memory_estimate_dtype": "torch.bfloat16",
        "additional_pixel_rounding": True,
    }
    return vae
