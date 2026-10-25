"""Fail-closed MOSS DiT checkpoint compatibility; no ComfyUI/GPU imports."""

import re


def normalize_fused_qkv(state):
    out = dict(state)
    for key in list(out):
        match = re.fullmatch(r"(layers\.\d+\.proj\.)qkv\.weight", key)
        if not match:
            continue
        prefix = match.group(1)
        weight = out[key]
        if (
            not weight.is_floating_point()
            or weight.ndim != 2
            or weight.shape[0] != 3 * weight.shape[1]
        ):
            raise ValueError(
                f"Unsupported fused QKV tensor: {key} shape={tuple(weight.shape)} dtype={weight.dtype}"
            )
        if any(k.startswith(prefix + "qkv.") and k != key for k in out):
            raise ValueError(
                f"Fused QKV quantization/bias metadata is not supported: {key}"
            )
        targets = [prefix + name + ".weight" for name in ("q", "k", "v")]
        if any(name in out for name in targets):
            raise ValueError(f"Ambiguous fused/separate QKV: {key}")
        del out[key]
        for target, part in zip(targets, weight.chunk(3, dim=0)):
            out[target] = part
    return out


def strict_load_diffusion_weights(model, state, assign=False):
    """Custom Comfy quant ops consume their metadata; any leftover/missing key fails."""
    result = model.load_state_dict(state, strict=False, assign=assign)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "Incomplete/incompatible MOSS DiT checkpoint: missing="
            + repr(result.missing_keys)
            + "; unexpected="
            + repr(result.unexpected_keys)
        )
    return result
