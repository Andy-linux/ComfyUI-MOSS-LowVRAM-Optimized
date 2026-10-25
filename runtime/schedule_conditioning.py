"""Request-local exact Euler timestep conditioning and 4G weight lifecycle.

Each actual sigma is evaluated by the selected original batch-one numerical path.
The four dense time-conditioning linears are absent from the sampling GPU
working set; their CPU modules are restored at request exit. Cached results
are registered buffers and follow normal Comfy model movement.
"""
from types import MethodType
import torch


class _Placeholder(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(0, dtype=dtype, device=device),
                                         requires_grad=False)
    def forward(self, *args):
        raise RuntimeError("Dense conditioning is unavailable during scheduled sampling")


class _Table(torch.nn.Module):
    def __init__(self, keys, mods, heads):
        super().__init__()
        self.register_buffer("mods", torch.cat(mods, dim=0))
        self.register_buffer("heads", torch.cat(heads, dim=0))
        self.register_buffer("keys", torch.cat(keys).to(dtype=torch.float32))
        self.register_buffer("indices", torch.arange(len(keys), device=self.mods.device))
        self.register_buffer("invalid", torch.zeros((), device=self.mods.device, dtype=torch.bool))
        self.calls = 0

    def forward(self, sigma):
        if sigma.dtype != torch.float32 or tuple(sigma.shape) != (1,):
            raise ValueError("Scheduled conditioning requires the original FP32 batch-one sigma")
        if sigma.device != self.mods.device:
            raise RuntimeError("Conditioning table was not loaded with its model")
        self.calls += 1
        # Exact FP32 lookup stays on CUDA. max chooses the last duplicate,
        # matching the original dictionary semantics without an NFE assumption.
        index = torch.where(self.keys == sigma[0], self.indices, -1).amax().reshape(1)
        self.invalid.logical_or_(index[0] < 0)
        index.clamp_min_(0)
        return self.mods.index_select(0, index), self.heads.index_select(0, index)

    def validate(self):
        # Called once after the existing end-of-sampling synchronize, before any
        # final latent is returned. Unknown sigmas fail the request, never fall
        # back to nearest matching or silently produce an accepted image.
        if self.invalid.item():
            raise RuntimeError("Sampler requested a sigma outside its prepared Euler schedule")


def install(bb, model_sampling, sigmas):
    if torch.is_grad_enabled() or bb.training:
        raise ValueError("Schedule conditioning requires evaluation without autograd")
    if hasattr(bb, "_moss_conditioning_table"):
        raise RuntimeError("Conditioning table already installed")
    if sigmas.ndim != 1 or not 2 <= len(sigmas) <= 65:
        raise ValueError("Schedule conditioning supports 1..64 Euler steps")
    originals = bb.conditioning_linears()
    device = originals[0].weight.device
    if device.type != "cuda" or sigmas.device != device:
        raise ValueError("Prepare conditioning on the original CUDA device")
    original_conditioning = bb.conditioning
    keys, mods, heads = [], [], []
    # Preserve Comfy model_base -> bridge FP32 multiply/divide boundaries.
    for i in range(len(sigmas) - 1):
        timestep = model_sampling.timestep(sigmas[i:i + 1]).float()
        bridge_sigma = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1) / 1000
        mod, head = original_conditioning(bridge_sigma)
        keys.append(bridge_sigma.detach().clone())
        mods.append(mod)
        heads.append(head)
    table = _Table(keys, mods, heads)
    weight_bytes = sum(t.numel() * t.element_size()
                       for module in originals for t in module.parameters())
    dtypes = [module.weight.dtype for module in originals]
    restored = False
    receipt = dict(steps=len(sigmas) - 1, dense_weight_bytes_removed=weight_bytes,
                   table_bytes=sum(x.numel() * x.element_size() for x in table.buffers()),
                   batch_size=1, precision=str(dtypes[0])+" linears and FP32 mod/head",
                   sampler="euler", calls=0, lookup="exact CUDA FP32; validate once before final latent")

    def restore():
        nonlocal restored
        if restored:
            return
        restored = True
        receipt["calls"] = table.calls
        try:
            bb.conditioning = original_conditioning
            bb.time_embedding[0], bb.time_embedding[2] = originals[:2]
            bb.shared_block_modulation, bb.video_head_modulation = originals[2:]
        finally:
            if hasattr(bb, "_moss_conditioning_table"):
                del bb._moss_conditioning_table
            table.mods = torch.empty(0)
            table.heads = torch.empty(0)
            table.keys = torch.empty(0)
            table.indices = torch.empty(0, dtype=torch.int64)
            table.invalid = torch.zeros((), dtype=torch.bool)

    try:
        # CPU owners are outside the registered tree; normal MM cannot reload them.
        for module in originals:
            module.cpu()
        bb.time_embedding[0] = _Placeholder(dtypes[0], device)
        bb.time_embedding[2] = _Placeholder(dtypes[1], device)
        bb.shared_block_modulation = _Placeholder(dtypes[2], device)
        bb.video_head_modulation = _Placeholder(dtypes[3], device)
        bb.add_module("_moss_conditioning_table", table)
        def conditioning(model, sigma):
            return model._moss_conditioning_table(sigma)
        bb.conditioning = MethodType(conditioning, bb)
    except BaseException:
        restore()
        raise
    return receipt, restore, table.validate
