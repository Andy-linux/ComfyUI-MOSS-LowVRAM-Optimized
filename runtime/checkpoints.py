"""Explicit DiT checkpoint boundaries; no ComfyUI, CUDA or model load on import.

Dense tensors always come from the selected checkpoint, never an SFT retained
sidecar. The packaged Kohaku loader owns prefix/fused-QKV normalization. Variant
is a caller declaration: Base and SFT have identical shapes, not identical weights.
"""
import json
import struct
import sys
from pathlib import Path


def bootstrap_vendor():
    root = (Path(__file__).resolve().parent / "vendor/kohaku_src").resolve()
    import importlib.util
    for name in ("kohakumoss", "kohakumoss_kernels", "kohakumoss_common"):
        module = sys.modules.get(name)
        spec = importlib.util.find_spec(name) if module is None else None
        origin = getattr(module, "__file__", None) if module else getattr(spec, "origin", None)
        if origin and root not in Path(origin).resolve().parents:
            raise RuntimeError(f"Conflicting {name}: {origin}; expected {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _headers(path):
    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        folder = path / "transformer" if (path / "transformer").is_dir() else path
        index = folder / "diffusion_pytorch_model.safetensors.index.json"
        if index.is_file():
            names = set(json.loads(index.read_text("utf-8"))["weight_map"].values())
            if any(Path(n).name != n for n in names):
                raise ValueError("Shard index must name local files")
            files = [folder / n for n in sorted(names)]
        else:
            files = sorted(folder.glob("*.safetensors"))
            if len(files) != 1:
                raise ValueError("Expected a single safetensors file or indexed shard set")
    tensors, metadata = {}, []
    for file in files:
        with file.open("rb") as stream:
            size = stream.read(8)
            if len(size) != 8:
                raise ValueError(f"Truncated safetensors header: {file}")
            length = struct.unpack("<Q", size)[0]
            if length > 64 * 1024 * 1024 or length + 8 > file.stat().st_size:
                raise ValueError(f"Invalid safetensors header length: {file}")
            header = json.loads(stream.read(length))
        metadata.append(header.pop("__metadata__", {}))
        for key, info in header.items():
            if key in tensors:
                raise ValueError(f"Duplicate checkpoint tensor: {key}")
            tensors[key] = (file, length + 8, info)
    return tensors, metadata


def _small_tensor(entry):
    file, start, info = entry
    a, b = info["data_offsets"]
    if info["dtype"] != "U8" or not 0 <= a <= b or b - a > 65536:
        raise ValueError("Invalid quantization JSON tensor")
    with file.open("rb") as stream:
        stream.seek(start + a)
        return json.loads(stream.read(b - a))


def inspect_checkpoint(path):
    """Read headers and small quant records only, without torch or weight loading."""
    tensors, metadata = _headers(path)
    records = [m.get("kohakumoss_fp8") for m in metadata]
    if len(set(records)) != 1:
        raise ValueError("Inconsistent FP8 metadata across shards")
    record = json.loads(records[0]) if records[0] is not None else None
    quant = {}
    for key, entry in tensors.items():
        if key.endswith(".comfy_quant"):
            quant[key[:-12]] = _small_tensor(entry)
    if record is not None:
        if len(metadata) != 1:
            raise ValueError("Packaged Kohaku stored-FP8 loader requires a single file")
        recipe = record.get("recipe")
        layers = record.get("layers")
        if recipe not in ("tensor", "row", "block", "mx") or not isinstance(layers, list) or not layers:
            raise ValueError("Invalid kohakumoss_fp8 recipe/layers")
        if any(not isinstance(n, str) or not n for n in layers) or len(set(layers)) != len(layers):
            raise ValueError("Invalid or duplicate FP8 layer")
        if quant:
            raise ValueError("Ambiguous Kohaku FP8 and Comfy quant records")
        actual = {k[:-7] for k, (_, _, info) in tensors.items()
                  if k.endswith(".weight") and info["dtype"] in ("I8", "F8_E4M3")}
        if actual != set(layers):
            raise ValueError("FP8 metadata must account for every quantized weight")
        for name in layers:
            try:
                q = tensors[name + ".weight"][2]
                scale = tensors[name + ".weight_scale"][2]
            except KeyError as exc:
                raise ValueError(f"Missing FP8 tensor: {name}") from exc
            if q["dtype"] != "F8_E4M3" or len(q["shape"]) != 2:
                raise ValueError(f"Expected E4M3 [N,K]: {name}")
            n, k = q["shape"]
            expected = {"tensor": [1], "row": [n], "block": [(n + 127)//128, k//128], "mx": [n, k//32]}[recipe]
            if n % 128 or k % 128 or scale["shape"] != expected or scale["dtype"] != ("U8" if recipe == "mx" else "F32"):
                raise ValueError(f"Invalid {recipe} FP8 scale/shape: {name}")
        fmt = "mxfp8" if recipe == "mx" else "fp8_compute"
    else:
        recipe = None
        kinds = {v.get("format") for v in quant.values()}
        if kinds - {"int8_tensorwise", "float8_e4m3fn", "asym_w4a8_int8"}:
            raise ValueError(f"Unsupported quant format: {kinds}")
        if any(key.endswith(".weight") and info["dtype"].startswith("F8_") and info["dtype"] != "F8_E4M3"
               for key, (_, _, info) in tensors.items()):
            raise ValueError("Only E4M3 FP8 checkpoints are supported")
        low = [key for key, (_, _, info) in tensors.items() if key.endswith(".weight") and info["dtype"] in ("I8", "F8_E4M3")]
        for key in low:
            name = key[:-7]
            if name not in quant:
                raise ValueError(f"Quantized weight needs explicit comfy_quant metadata: {name}")
        if kinds == {"int8_tensorwise"}:
            fmt = "int8_convrot" if all(v.get("convrot") and v.get("convrot_groupsize") == 256 for v in quant.values()) else "int8_weight_only"
        else:
            fmt = "fp8_weight_only" if kinds == {"float8_e4m3fn"} else "mixed_quant" if kinds else "dense"
    return {"path": str(Path(path).resolve()), "format": fmt, "recipe": recipe,
            "tensor_count": len(tensors), "quant_layers": quant,
            "fp8_record": record, "fused_qkv": any(".proj.qkv.weight" in k for k in tensors),
            "variant_detection": "not inferable from identical Base/SFT architecture"}


def load_checkpoint(path, *, variant, dtype=None, device="cpu", int8_backend="w8a8", kernels=None, cfg=None, keep_fp32=None):
    """Return (backbone, receipt). Caller handles admission, placement and fp16 fix.

    W8A8 is the existing fixed ConvRot256/A8/native INT32 implementation. Its
    forward requires supported CUDA/comfy-kitchen; CPU loading is allowed. FP8
    compute binding is redone by bind_fp8_inference after the caller's placement.
    No weight-only fallback is silently substituted for requested W8A8.
    """
    if variant not in ("base", "sft"):
        raise ValueError("Explicit variant='base' or 'sft' required")
    if int8_backend not in ("w8a8", "weight_only"):
        raise ValueError("int8_backend must be w8a8 or weight_only")
    info = inspect_checkpoint(path)
    bootstrap_vendor()
    import torch
    from kohakumoss.model.io.load import load_backbone
    from kohakumoss.model.build import DEFAULT_KEEP_FP32
    from kohakumoss.quant.weights import QuantWeight
    dtype = torch.bfloat16 if dtype is None else dtype
    keep_fp32 = DEFAULT_KEEP_FP32 if keep_fp32 is None else tuple(keep_fp32)
    model, report = load_backbone(path, device=device, dtype=dtype, kernels=kernels, cfg=cfg,
                                  keep_fp32=keep_fp32, strict=True)
    replacements = []
    if int8_backend == "w8a8":
        for name, lin in model.named_modules():
            base = getattr(lin, "base", None)
            if not isinstance(base, QuantWeight) or base.fmt != "int8":
                continue
            if (base.rotation != 256 or not name.startswith("layers.") or ".proj." not in name
                    or lin.bias is not None or base.q.ndim != 2
                    or any(v % 16 for v in base.q.shape) or base.scale.numel() != base.q.shape[0]):
                raise ValueError(f"Unsupported fixed ConvRot256 W8A8 layer: {name}")
            replacements.append((name, lin, base))
        from .storage import FrozenInt8Base
        for name, lin, base in replacements:
            lin.set_base(FrozenInt8Base(base.q, base.scale.reshape(-1, 1).contiguous()))
            # Frozen fusion contracts forbid a leftover dense Parameter, even
            # an empty placeholder used by the general-purpose Kohaku loader.
            if "weight" in lin._parameters:
                del lin.weight
            lin.__dict__["_held"]=torch.empty(0,dtype=dtype,device=base.q.device)
    model.eval().requires_grad_(False)
    receipt = {**info, "variant": variant, "dense_source": str(Path(path).resolve()),
               "load_missing": report.missing, "load_unexpected": report.unexpected,
               "w8a8_layers": [n for n, _, _ in replacements],
               "int8_backend": int8_backend, "compute_dtype": str(dtype),
               "keep_fp32": list(keep_fp32),
               "dense_tensor_policy": "same selected checkpoint; vendor compute dtype and KEEP_FP32 rules",
               "gpu_qualified": False}
    return model, receipt
