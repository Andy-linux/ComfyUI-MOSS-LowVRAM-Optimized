"""Explicit dense FP16/BF16 Flash backend for packed THD MOSS calls.
Prefer the existing private xformers binary; otherwise use torch Flash-only SDPA.
Not Hopper-only FA3, not QKV quantization, not bitwise equivalent to CUTLASS.
No silent backend or multi-segment fallback. The backbone caller retains CFG/SLG scheduling.
"""

import importlib, sys, os, hashlib
from pathlib import Path
import torch

_BINARY_METADATA = {}


def _binary_sha256(binary):
    """Metadata cache only: no Tensor/GPU lifetime or per-request full-file I/O."""
    stat = binary.stat()
    identity = (str(binary), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity not in _BINARY_METADATA:
        _BINARY_METADATA[identity] = hashlib.sha256(binary.read_bytes()).hexdigest()
    return _BINARY_METADATA[identity]


class FlashFP16Backend:
    def __init__(self, device, vendor=None, dll_directory=None):
        capability = torch.cuda.get_device_capability(device)
        if capability < (8, 0):
            raise RuntimeError("Public Flash attention requires SM80+; SM75 uses its Turing backend")
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device(self.device.type, torch.cuda.current_device())
        self.vendor = Path(
            vendor or Path(__file__).parent / "xformers034cu128"
        ).resolve()
        binary = self.vendor / "xformers/flash_attn_3/_C.pyd"
        self.dll_handle = None
        self.closed = False
        if capability != (8, 9) or not binary.is_file():
            self._init_public_flash()
            return
        if "xformers" in sys.modules:
            loaded = Path(sys.modules["xformers"].__file__).resolve()
            if self.vendor not in loaded.parents:
                raise RuntimeError(
                    "Another xformers is loaded; select backend in a fresh process"
                )
        dll_directory = (
            dll_directory
            or "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
        )
        try:
            if os.name == "nt":
                self.dll_handle = os.add_dll_directory(dll_directory)
            if str(self.vendor) not in sys.path:
                sys.path.insert(0, str(self.vendor))
            self.xo = importlib.import_module("xformers.ops")
            self.flash3 = importlib.import_module("xformers.ops.fmha.flash3")
            if not self.flash3.FwOp.is_available():
                raise RuntimeError(
                    "Explicit Flash forward unavailable; no CUTLASS substitution"
                )
        except Exception:
            self.close()
            raise
        self.kind = "private_xformers_flash"
        self.receipt = {
            "backend": "private xformers Flash forward",
            "interface": "xformers_flash3::flash_fwd",
            "xformers": getattr(
                importlib.import_module("xformers"), "__version__", "unknown"
            ),
            "binary_cuda_build_hint": "12.8 (private directory name; not a runtime probe)",
            "host_torch": torch.__version__,
            "host_cuda": torch.version.cuda,
            "vendor": str(self.vendor),
            "flash_binary_sha256": _binary_sha256(binary),
            "binary_hash_policy": "cached per process path/stat identity; first use reads whole binary",
            "DLL_search_directory": str(dll_directory),
            "compute_dtypes": ["FP16", "BF16"],
            "full_dense": True,
            "QKV_quantized": False,
            "bitwise_equivalent_to_SDPA": False,
            "qualified": "SM89 actual-layout synthetic micro; image/long-sequence qualification tracked separately",
            "ownership": "DLL search-directory handle held by this backend until close; loaded extension remains process-owned and is not unloaded or replaced",
        }

    def _init_public_flash(self):
        """Explicit torch Flash backend only: math/efficient/cuDNN are disabled.

        Availability is not qualification. Unsupported runtime shapes fail at the
        call; auto remains the separate existing conservative attention choice.
        """
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            if not torch.backends.cuda.is_flash_attention_available():
                raise RuntimeError("this torch build has no FlashAttention support")
            self.flash_enum = SDPBackend.FLASH_ATTENTION
            self.sdpa_kernel = sdpa_kernel
        except (ImportError, AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                "Explicit Flash is unavailable. Install a compatible CUDA PyTorch build with FlashAttention support, restore the optional private xformers dependency, or select auto. No ordinary SDPA fallback was used."
            ) from exc
        self.kind = "public_torch_flash"
        self.receipt = {
            "backend": "torch FlashAttention only",
            "interface": "torch SDPA with SDPBackend.FLASH_ATTENTION exclusively",
            "host_torch": torch.__version__,
            "host_cuda": torch.version.cuda,
            "vendor": None,
            "compute_dtypes": ["FP16", "BF16"],
            "full_dense": True,
            "QKV_quantized": False,
            "bitwise_equivalent_to_SDPA": False,
            "qualified": "candidate on SM89; public build requires real layout/image qualification",
            "fallback_allowed": False,
        }

    def validate_plan(self, cu_seqlens, max_seqlen, tokens):
        """Call at sequence construction/request boundary, never once per layer.
        Caller may pass original CPU cu values to avoid an extra GPU synchronization.
        """
        values = (
            cu_seqlens.detach().cpu().tolist()
            if isinstance(cu_seqlens, torch.Tensor)
            else list(cu_seqlens)
        )
        if values != [0, tokens] or max_seqlen != tokens:
            raise ValueError(
                "Qualified full dense single sequence requires cu=[0,N], max_seqlen=N"
            )

    def __call__(self, q, k, v, cu_seqlens, max_seqlen, scale):
        if self.closed:
            raise RuntimeError("Flash backend is closed")
        if cu_seqlens.numel() != 2 or max_seqlen != q.shape[0]:
            raise ValueError(
                "Explicit Flash selection supports only packed single-segment full attention"
            )
        if (
            q.ndim != 3
            or q.shape != k.shape
            or q.shape != v.shape
            or q.shape[-1] != 128
        ):
            raise ValueError("Expected equal Q/K/V packed [tokens,heads,128]")
        if any(
            t.dtype not in (torch.float16, torch.bfloat16)
            or t.dtype != q.dtype
            or t.device != self.device
            or t.stride(-1) != 1
            for t in (q, k, v)
        ):
            raise ValueError(
                "Requires equal FP16/BF16 on the selected GPU and contiguous head dimension"
            )
        if self.kind == "public_torch_flash":
            try:
                with self.sdpa_kernel(self.flash_enum):
                    out = torch.nn.functional.scaled_dot_product_attention(
                        *(t.unsqueeze(0).transpose(1, 2) for t in (q, k, v)),
                        dropout_p=0.0,
                        is_causal=False,
                        scale=scale,
                    )
                return out.transpose(1, 2).squeeze(0)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Explicit torch FlashAttention failed for this device/dtype/layout. Use a compatible Flash-enabled CUDA PyTorch build or select auto; no other attention kernel was substituted."
                ) from exc
        return self.xo.memory_efficient_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            p=0.0,
            scale=scale,
            op=(self.flash3.FwOp, None),
        ).squeeze(0)

    def close(self):
        if self.dll_handle is not None:
            self.dll_handle.close()
            self.dll_handle = None
        self.closed = True


def install(bb, **kwargs):
    backend = FlashFP16Backend(bb.video_embedder.weight.device, **kwargs)
    ops = bb.layers[0].ops
    old = ops.attention_varlen
    ops.attention_varlen = backend

    def restore():
        ops.attention_varlen = old
        backend.close()

    return backend.receipt, restore
