"""Inference-only FP8 binding and explicit offline checkpoint conversion.

The FP8 compute binding quantizes activations per call. Weight-only FP8 keeps
activations in the compute dtype; row checkpoints can select either at CPU load. Neither name promises a faster kernel on a GPU.
"""
from pathlib import Path
import json
import os
import uuid

from .checkpoints import bootstrap_vendor, inspect_checkpoint


FP8_ACTIVATION_MODES = ("checkpoint", "fp8", "16bit")


def select_fp8_activation(model, *, mode="checkpoint", checkpoint, dtype):
    """Select forward activation policy once, on freshly loaded CPU storage.

    The shared contract is E4M3 [N,K] with FP32 per-row scales. Switching that
    storage to A16 only replaces its registered base wrapper; no quantizer,
    materialization, training transpose, or alternate model copy is involved.
    Validate every layer before changing any base, so invalid input is atomic.
    Other compute recipes retain their legacy behavior only in checkpoint mode.
    """
    if mode not in FP8_ACTIVATION_MODES:
        raise ValueError(f"fp8_activation_mode must be one of {FP8_ACTIVATION_MODES}")
    bootstrap_vendor()
    import torch
    from kohakumoss.quant.compute import ComputeWeight
    from kohakumoss.quant.weights import QuantWeight

    storage = checkpoint["format"]
    if storage not in ("fp8_compute", "mxfp8", "fp8_weight_only"):
        raise ValueError("Activation selection requires an explicitly described FP8 checkpoint")
    if mode == "fp8" and storage == "fp8_weight_only":
        raise ValueError("Explicit fp8 activations require a stored row compute checkpoint")
    if mode != "checkpoint" and storage != "fp8_weight_only" and (
        storage != "fp8_compute" or checkpoint.get("recipe") != "row"
    ):
        raise ValueError("Explicit activation selection requires the shared row FP8 recipe")
    layers = []
    replacements = []
    for name, lin in model.named_modules():
        base = getattr(lin, "base", None)
        if not isinstance(base, (ComputeWeight, QuantWeight)):
            continue
        if isinstance(base, QuantWeight) and base.fmt != "fp8":
            raise ValueError(f"Unexpected non-FP8 stored base: {name}")
        compute = isinstance(base, ComputeWeight)
        if compute != (storage != "fp8_weight_only"):
            raise ValueError(f"Stored FP8 class does not match checkpoint metadata: {name}")
        if any(getattr(base, key, None) is not None for key in ("q_t", "scale_t", "aux_t")):
            raise ValueError(f"Inference session rejects FP8 training transposes: {name}")
        q, scale = base.q, base.scale
        recipe = base.recipe if compute else "row"
        if compute and recipe != checkpoint.get("recipe"):
            raise ValueError(f"Stored FP8 recipe does not match checkpoint metadata: {name}")
        if (q.device.type != "cpu" or scale.device.type != "cpu"
                or q.dtype != torch.float8_e4m3fn or q.ndim != 2
                or any(dim <= 0 for dim in q.shape)):
            raise ValueError(f"Expected CPU E4M3 [N,K] FP8 storage: {name}")
        if recipe == "row":
            if scale.dtype != torch.float32 or tuple(scale.shape) != (q.shape[0],):
                raise ValueError(f"Expected FP32 [N] FP8 row scales: {name}")
            if compute and (getattr(base, "aux", None) is not None or any(dim % 128 for dim in q.shape)):
                raise ValueError(f"Invalid shared FP8 row layout: {name}")
        elif mode != "checkpoint":
            raise ValueError(f"Activation selection requires row FP8 storage: {name}")
        if mode == "16bit" and compute:
            if not callable(getattr(lin, "set_base", None)):
                raise ValueError(f"FP8 layer has no registered base setter: {name}")
            replacements.append((lin, QuantWeight(q, scale, "fp8", rotation=0)))
        layers.append(name)
    if not layers:
        raise ValueError("No stored FP8 layers found")
    for lin, replacement in replacements:
        lin.set_base(replacement)
    effective = "16bit" if storage == "fp8_weight_only" or mode == "16bit" else "fp8"
    return {"storage_format": storage, "storage_recipe": checkpoint.get("recipe"),
            "requested_mode": mode, "effective_mode": effective,
            "compute_dtype": str(dtype), "layers": layers,
            "replaced_base_count": len(replacements), "requantized": False,
            "persistent_dense_copy": False, "training_transpose_allocated": False,
            "kernel_policy": "vendor dispatch may use 16-bit arithmetic; no blanket FP8 Tensor Core claim"}


def bind_fp8_inference(model, *, dtype, device, backend=None):
    """Rebind stored bases after placement. CPU requires explicit eager reference.

    This does not place tensors or allocate a training transpose/dgrad copy. A
    device capability check is not performance qualification; vendor per-shape
    selection can still choose its documented 16-bit path.
    """
    bootstrap_vendor()
    import torch
    from kohakumoss.quant.compute import ComputeWeight
    from kohakumoss.quant.weights import QuantWeight
    dev = torch.device(device)
    if backend != "eager":
        if dev.type != "cuda" or torch.cuda.get_device_capability(dev) < (8, 9):
            raise ValueError("FP8 compute requires CUDA SM89+; explicit backend='eager' is a reference only")
    bound = []
    for name, lin in model.named_modules():
        base = getattr(lin, "base", None)
        if not isinstance(base, (ComputeWeight, QuantWeight)):
            continue
        if isinstance(base, QuantWeight) and base.fmt != "fp8":
            continue
        if any(getattr(base,key,None) is not None for key in ("q_t","scale_t","aux_t")):
            raise ValueError(f"Inference session rejects FP8 training transposes: {name}")
        for key in ("q","scale","aux"):
            tensor=getattr(base,key,None)
            if tensor is not None and (tensor.device.type!=dev.type or
                    (dev.index is not None and tensor.device.index!=dev.index)):
                raise ValueError(f"Place FP8 {key} before binding: {name}")
        if isinstance(base, ComputeWeight):
            base.bind(dtype, dev, backend=backend, autograd=False)
            kind = "mxfp8" if base.recipe == "mx" else "fp8_compute"
            grad_policy = "binding_autograd_false"
        else:
            if backend == "eager":
                from kohakumoss.quant.weights import _eager_for
                base._linear = _eager_for(base.rotation)
            elif backend is not None:
                raise ValueError("Pinned weight-only API supports auto or eager binding")
            else:
                base.bind(dtype, dev)
            kind = "fp8_weight_only"
            grad_policy = "inference_mode_owned_by_session"
        bound.append({"path": name, "format": kind, "grad_policy": grad_policy,
                      "effective_activation_mode": "fp8" if isinstance(base, ComputeWeight) else "16bit",
                      "compute_dtype": str(dtype)})
    return {"bound": bound, "backend_request": backend, "device": str(dev),
            "training_transpose_allocated": False,
            "forward_only_request":all(item["grad_policy"] == "binding_autograd_false" for item in bound),
            "gpu_qualified": False,
            "kernel_policy": "vendor per-shape selection; no blanket FP8 GEMM or speed claim"}


def export_fp8(source, output, *, variant, precision="fp8_compute", recipe="row", device="cpu", min_features=256):
    """Offline conversion from dense BF16 weights; no training or remote download.

    Compute exports use the packaged Kohaku metadata contract. Weight-only output
    uses explicit Comfy per-layer format records and has no kohakumoss_fp8 record.
    The dense checkpoint remains untouched. Existing outputs are never replaced.
    """
    if variant not in ("base", "sft"):
        raise ValueError("Explicit base/sft source declaration required")
    if precision not in ("fp8_compute", "mxfp8", "fp8_weight_only"):
        raise ValueError("Choose fp8_compute, mxfp8 or fp8_weight_only")
    if recipe not in ("tensor", "row", "block"):
        raise ValueError("FP8 recipe must be tensor, row or block")
    if precision == "fp8_weight_only" and recipe != "row":
        raise ValueError("Weight-only FP8 uses per-row scales")
    if min_features < 256:
        raise ValueError("min_features must be >=256")
    source_info = inspect_checkpoint(source)
    if source_info["format"] != "dense":
        raise ValueError("Export requires dense source values, not already quantized codes")
    output = Path(output)
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Output or receipt already exists")
    bootstrap_vendor()
    import torch
    from safetensors.torch import save_file
    from kohakumoss.model.build import DEFAULT_KEEP_FP32
    from kohakumoss.model.io.load import load_backbone, base_keys
    from kohakumoss.model.io.quant_source import backbone_source
    from kohakumoss.quant.convert import default_filter, quantize_model
    from kohakumoss.quant.stored import quantized_state
    from kohakumoss.quant.weights import QuantWeight
    with torch.inference_mode():
        model, load = load_backbone(source, device="cpu", dtype=torch.bfloat16, strict=True)
        if getattr(model, "fold_scales", None) is not None:
            raise ValueError("Export requires unfolded source weights")
        values, scales = backbone_source(model)
        fmt = "fp8" if precision == "fp8_weight_only" else precision
        quant = quantize_model(model, fmt, default_filter(DEFAULT_KEEP_FP32, min_features),
                               torch.bfloat16, recipe, dgrad=False,
                               work_device=torch.device(device), source=values, scales=scales)
        if not quant["layers"]:
            raise ValueError("No eligible FP8 layers")
        keys = base_keys(model.cfg)
        if precision != "fp8_weight_only":
            state, metadata = quantized_state(model, keys, torch.bfloat16)
        else:
            live = {**dict(model.named_buffers()), **dict(model.named_parameters())}
            state = {}
            for key in keys:
                name = key[:-7] if key.endswith(".weight") else None
                base = getattr(model.get_submodule(name), "base", None) if name else None
                if isinstance(base, QuantWeight):
                    state[key] = base.q.contiguous().cpu()
                    state[name + ".weight_scale"] = base.scale.contiguous().cpu()
                    record = json.dumps({"format": "float8_e4m3fn", "convrot": False}).encode()
                    state[name + ".comfy_quant"] = torch.tensor(list(record), dtype=torch.uint8)
                else:
                    state[key] = live[key].detach().contiguous().cpu()
            metadata = {}
        metadata.update(source=Path(source).name, moss_variant=variant,
                        moss_export_precision=precision)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            save_file(state, str(temporary), metadata)
            # Validate our actual serialized format before publishing the output.
            verified = inspect_checkpoint(temporary)
            os.link(temporary, output)  # atomic fail-if-exists on local NTFS/POSIX
        finally:
            temporary.unlink(missing_ok=True)
    receipt = {"source": str(Path(source).resolve()), "output": str(output.resolve()),
               "variant": variant, "precision": precision,
               "recipe": "mx" if precision == "mxfp8" else recipe,
               "quantization": quant, "source_tensors": load.n_source,
               "bytes": output.stat().st_size, "verified_format": verified["format"],
               "training_performed": False, "gpu_qualified": False}
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    return receipt
