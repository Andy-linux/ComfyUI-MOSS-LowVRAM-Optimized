"""Independent FFN v3 installer: INT8 Tensor Core register-accumulator SiTU up.

Fixed qualified SM89 config BM64/BN64/BK64/warps4/stages2. No autotune.
Full FFN is the default. Down projection and its existing writer, scaling, gate
and residual are unchanged. On packed models the original single INT8 workspace
can be overwritten by down AFTER up completes; no additional decoded weight copy
is needed for tile0. Optional token tiling remains an explicit capacity fallback.
"""
import types
import torch
from ..storage import unpack_base
from .ffn_gemm_v3 import up_situ as register_up_situ

CONFIG = dict(bm=64, bn=64, bk=64, warps=4, stages=2)


def up_situ(x, base):
    resident = hasattr(base, "codes") and hasattr(base, "scales")
    codes = base.codes if resident else unpack_base(base)
    scales = base.scales if resident else base.wscales
    if hasattr(base, "calls"):
        base.calls += 1
    return register_up_situ(x, codes, scales, config=CONFIG)


def install(bb, *, tile=0):
    """Install AFTER final bind_ops; mutually exclusive with installed FFN v2.

    Model execution remains synchronous on its request CUDA stream. Only original
    frozen packed W5/W6 or original resident INT8 bases are accepted. Packed format,
    scales and fixed ConvRot64/256 activation quantization are never changed.
    """
    from .inference_contracts_v3 import validate_model
    inference_contract = validate_model(bb, backend="ffn_v3")
    from kohakumoss.types.context import SHIFT_FFN, SCALE_FFN, GATE_FFN
    if tile < 0:
        raise ValueError("tile must be zero or positive")
    if getattr(bb, "_ffn_v3_installed", False):
        raise RuntimeError("FFN v3 already installed")
    if any(getattr(block.proj["ffn_down"].base.linear, "__module__", "") == "ffn_fused_v2" for block in bb.layers):
        raise RuntimeError("FFN v2 and v3 are mutually exclusive; restore v2 first")
    originals = []
    for block in bb.layers:
        base = block.proj["ffn_up_gate"].base
        if not ((hasattr(base, "qweight") and hasattr(base, "wscales"))
                or (hasattr(base, "codes") and hasattr(base, "scales"))):
            raise ValueError("Only frozen packed W5/W6 or original resident W8 bases")
        originals.append((block, block.ffn_part))

        def ffn(self, x, ctx, _base=base):
            out = torch.empty_like(x)
            step = tile or len(x)
            for start in range(0, len(x), step):
                xx = x[start:start + step]
                seg = ctx.seq.seg_joint[start:start + step]
                y = self._cast(self.ops.rms_norm_mod(
                    xx, self.norm2.weight, ctx.mod[:, SHIFT_FFN],
                    ctx.mod[:, SCALE_FFN], seg, self.norm2.eps))
                act = up_situ(y, _base)
                del y
                out[start:start + step] = self._post(
                    self._ffn_out(act, xx, ctx.mod[:, GATE_FFN], seg, 1.0))
                del act
            return out
        block.ffn_part = types.MethodType(ffn, block)
    bb._ffn_v3_installed = True

    def restore():
        for block, old in originals:
            block.ffn_part = old
        bb._ffn_v3_installed = False
    receipt = dict(
        inference_contract=inference_contract,
        backend="Triton INT8 Tensor Core up+gate SiTU register INT32; native down unchanged",
        config=CONFIG.copy(), tile=tile,
        packed_layout_and_scales_unchanged=True,
        activation_quantizer="original fixed ConvRot64 groupsize256",
        rounding="(INT32.float*activation_scale)*weight_scale -> "+inference_contract["dtype"]+" -> SiTU -> "+inference_contract["dtype"],
        additional_decoded_weight_workspace_bytes=0,
        external_full_upgate_int32_accumulator_bytes=0,
        qualification="SM89 micro INT32/output bitwise; fullimage still required")
    return receipt, restore
