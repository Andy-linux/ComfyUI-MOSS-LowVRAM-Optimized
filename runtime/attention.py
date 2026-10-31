"""Explicit CUDA attention choices. Dense geometry and complete K/V are preserved.

Sage choices add Q/K (and optionally P/V) quantization and are never silently
selected by the numerically conservative auto mode. Backend failure propagates.
"""

import importlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
import torch


class AttentionBackend:
    def __init__(self, device, backend, *, head_group=0):
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        major, minor = torch.cuda.get_device_capability(self.device)
        sm = major * 10 + minor
        minimum = 75 if backend == "ck_int8" else 89 if backend.startswith("sage_fp8") else 80
        if sm < minimum:
            raise ValueError(f"{backend} requires SM{minimum} or newer; detected SM{sm}")
        if head_group not in (0, 8, 12, 24):
            raise ValueError("head_group must be 0, 8, 12 or 24")
        self.backend, self.head_group = backend, head_group
        self.receipt = {
            "backend": backend,
            "head_group": head_group,
            "full_KV": True,
            "extra_quantization": backend.startswith("sage_") or backend == "ck_int8",
            "numerically_identical_to_auto": False,
            "calls": 0,
        }
        if backend.startswith("sage_"):
            vendor = (Path(__file__).parent / "vendor/sageattn220").resolve()
            existing = sys.modules.get("sageattention")
            public = importlib.util.find_spec("sageattention") if existing is None else None
            private = existing is None and public is None and (vendor / "sageattention").is_dir()
            if private and str(vendor) not in sys.path:
                sys.path.insert(0, str(vendor))
            try:
                sage = importlib.import_module("sageattention")
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "Optional SageAttention is unavailable or incompatible with this torch/CUDA build. Install a compatible sageattention package, restore the private installation, or select auto."
                ) from exc
            if backend in ("sage_fp16", "sage_fp16_rot"):
                fn_name = "sageattn_qk_int8_pv_fp16_cuda"
                self.kw = {"qk_quant_gran": "per_thread", "pv_accum_dtype": "fp32"}
            elif backend in ("sage_fp8", "sage_fp8_fast", "sage_fp8_rot"):
                fn_name = "sageattn_qk_int8_pv_fp8_cuda"
                self.kw = {
                    "qk_quant_gran": "per_thread",
                    "pv_accum_dtype": "fp32+fp32"
                    if backend == "sage_fp8"
                    else "fp32+fp16",
                }
            else:
                raise ValueError(backend)
            if sm >= 100:
                self.kw["qk_quant_gran"] = "per_warp"
            self.fn = getattr(sage, fn_name, None)
            if not callable(self.fn):
                raise RuntimeError(
                    "Installed SageAttention lacks "
                    + fn_name
                    + "; install a compatible build or select auto"
                )
            version = getattr(sage, "__version__", None)
            version_source = "module __version__"
            if not version:
                distributions = (
                    list(importlib.metadata.distributions(path=[str(vendor)]))
                    if private
                    else []
                )
                version = next(
                    (
                        d.version
                        for d in distributions
                        if d.metadata.get("Name", "").lower().replace("_", "-")
                        == "sageattention"
                    ),
                    None,
                )
                version_source = (
                    "private distribution metadata"
                    if private
                    else "installed distribution metadata"
                )
                if not private:
                    try:
                        version = importlib.metadata.version("sageattention")
                    except importlib.metadata.PackageNotFoundError:
                        version = None
            self.receipt.update(
                version=version or "unknown (no version metadata)",
                version_source=version_source,
                vendor=str(vendor) if private else None,
                module_path=str(Path(sage.__file__).resolve()),
                dependency_source="private" if private else "installed package",
                interface=fn_name,
                host_torch=torch.__version__,
                host_cuda=torch.version.cuda,
                qualification=f"CUDA SM{sm}; selected public interface",
                **self.kw,
            )
            self.receipt["matched_QK_H128"] = backend.endswith("_rot")
        elif backend == "ck_int8":
            from comfy_kitchen.sage_attention import int8_attention

            self.fn = int8_attention
        elif backend != "cudnn":
            raise ValueError(backend)

    def _call(self, q, k, v, scale):
        if self.backend == "cudnn":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            # Insert batch before transposing: canonical nonzero B stride matters
            # to cuDNN even at B=1. These are views, not a layout copy.
            operands = [t.unsqueeze(0).transpose(1, 2) for t in (q, k, v)]
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = torch.nn.functional.scaled_dot_product_attention(
                    *operands, dropout_p=0.0, is_causal=False, scale=scale
                )
            return out.transpose(1, 2).squeeze(0)
        if self.backend == "ck_int8":
            return (
                self.fn(
                    *(t.unsqueeze(0).transpose(1, 2) for t in (q, k, v)), scale=scale
                )
                .transpose(1, 2)
                .squeeze(0)
            )
        if self.backend.endswith("_rot"):
            from .attention_rotation import rotate_qk

            q, k = rotate_qk(q, k)
        return self.fn(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            tensor_layout="NHD",
            is_causal=False,
            sm_scale=scale,
            **self.kw,
        ).squeeze(0)

    def __call__(self, q, k, v, cu_seqlens, max_seqlen, scale):
        if cu_seqlens.numel() != 2 or max_seqlen != q.shape[0]:
            raise ValueError("Single full dense segment required")
        if (
            q.ndim != 3
            or q.shape != k.shape
            or q.shape != v.shape
            or q.shape[-1] != 128
        ):
            raise ValueError("Equal THD Q/K/V with D128 required")
        if q.dtype not in (torch.float16, torch.bfloat16) or any(
            t.dtype != q.dtype or t.device != self.device or t.stride(-1) != 1
            for t in (q, k, v)
        ):
            raise ValueError(
                "Same-device FP16/BF16 operands with contiguous head dimension required"
            )
        self.receipt["calls"] += 1
        if not self.head_group or self.head_group >= q.shape[1]:
            return self._call(q, k, v, scale)
        out = torch.empty_like(q)
        for start in range(0, q.shape[1], self.head_group):
            out[:, start : start + self.head_group] = self._call(
                *(x[:, start : start + self.head_group] for x in (q, k, v)), scale
            )
        return out

    def close(self):
        pass


def install(bb, backend, head_group=0):
    selected = AttentionBackend(
        bb.video_embedder.weight.device, backend, head_group=head_group
    )
    ops = bb.layers[0].ops
    old = ops.attention_varlen
    ops.attention_varlen = selected

    def restore():
        ops.attention_varlen = old
        selected.close()

    return selected.receipt, restore
