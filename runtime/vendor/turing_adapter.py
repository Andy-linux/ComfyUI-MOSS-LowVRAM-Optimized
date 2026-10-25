"""Explicit SM75 dense FP16 backend using the locally built upstream Turing extension.

Default auto SDPA remains unchanged. This callable does not modify projection or
weight contracts. The imported binary stays process-owned after close().
"""
import importlib
import os
from pathlib import Path
import hashlib
import json
import sys


class TuringFP16Backend:
    def __init__(self, device, vendor=None):
        import torch
        if os.name != "nt" or sys.version_info[:2] != (3,12):
            raise RuntimeError("Turing binary requires Windows CPython3.12")
        if torch.__version__.split("+")[0] != "2.10.0" or torch.version.cuda != "13.0":
            raise RuntimeError("Turing binary qualified with torch2.10.0+cu130 only")
        self.device=torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Explicit CUDA device required")
        if self.device.index is None:
            self.device=torch.device("cuda",torch.cuda.current_device())
        if torch.cuda.get_device_capability(self.device) != (7,5):
            raise RuntimeError("Turing backend qualified only on SM75")
        self.vendor=Path(vendor or Path(__file__).parent/"turing_bundle").resolve()
        manifest=json.loads((self.vendor/"MANIFEST.json").read_text())
        binary=self.vendor/manifest["binary"]["filename"]
        digest=hashlib.sha256(binary.read_bytes()).hexdigest()
        if digest != manifest["binary"]["sha256"]:
            raise RuntimeError("Qualified Turing binary SHA mismatch")
        for name in ("flash_attn_turing","flash_attention_interface"):
            if name in sys.modules and Path(sys.modules[name].__file__).resolve().parent != self.vendor:
                raise RuntimeError("Another "+name+" module is loaded; select a fresh process")
        self.dll_handle=None
        self.closed=False
        try:
            self.dll_handle=os.add_dll_directory(str(self.vendor/"dlls"))
            if str(self.vendor) not in sys.path:
                sys.path.insert(0,str(self.vendor))
            self.interface=importlib.import_module("flash_attention_interface")
        except Exception:
            self.close()
            raise
        self.receipt={
            "backend":"ssiu FlashAttention Turing SM75 dense FP16",
            "kernel":"flash_fwd_kernel<Flash_fwd_kernel_traits<128,128,64,8,half>>",
            "binary_sha256":digest,"source_head":manifest["source_head"],
            "host_torch":torch.__version__,"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device":str(self.device),"FP16":True,"full_dense":True,
            "QKV_quantized":False,"bitwise_equivalent_to_SDPA":False,
            "qualification":"Synthetic THD N4133/4362/5894/10206 H24/D128; actual image qualification separate",
            "ownership":"DLL search handle backend-owned; imported extension process-owned",
            "source_license":manifest["source_license"],
        }

    def validate_plan(self,cu_seqlens,max_seqlen,tokens):
        import torch
        values=cu_seqlens.detach().cpu().tolist() if isinstance(cu_seqlens,torch.Tensor) else list(cu_seqlens)
        if values != [0,tokens] or max_seqlen != tokens:
            raise ValueError("Turing adapter requires a single full segment cu=[0,N]")

    def __call__(self,q,k,v,cu_seqlens,max_seqlen,scale):
        import torch
        if self.closed:
            raise RuntimeError("Turing backend is closed")
        if torch.is_grad_enabled() or any(t.requires_grad for t in (q,k,v)):
            raise ValueError("Qualified Turing adapter is inference-only")
        if cu_seqlens.numel()!=2 or max_seqlen!=q.shape[0]:
            raise ValueError("Single full packed segment required")
        if q.ndim!=3 or q.shape!=k.shape or q.shape!=v.shape or tuple(q.shape[1:])!=(24,128):
            raise ValueError("Equal packed Q/K/V THD H24/D128 required")
        if any(t.dtype!=torch.float16 or t.device!=self.device or not t.is_contiguous() for t in (q,k,v)):
            raise ValueError("Contiguous CUDA FP16 THD on selected SM75 required")
        return self.interface.flash_attn_func(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),
                    softmax_scale=scale,causal=False).squeeze(0)

    def close(self):
        if self.dll_handle is not None:
            self.dll_handle.close()
            self.dll_handle=None
        self.closed=True


def install(bb,**kwargs):
    backend=TuringFP16Backend(bb.video_embedder.weight.device,**kwargs)
    ops=bb.layers[0].ops
    old=ops.attention_varlen
    ops.attention_varlen=backend
    def restore():
        ops.attention_varlen=old
        backend.close()
    return backend.receipt,restore
