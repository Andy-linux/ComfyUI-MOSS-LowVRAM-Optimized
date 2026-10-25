"""Reuse losslessly decoded W5/W6 weights within a tiled FFNv3 call.

The original 48 MiB owner stores up/gate. A 24 MiB down buffer lives only
during this FFN invocation. Quantizers, GEMMs, FP16 boundaries and the
original scaled writer are unchanged. No persistent CUDA weight cache.
"""
from types import MethodType
import torch
from .storage import FrozenPackedBase, linear, unpack_base
from .owned_scratch import unpack_into
from .vendor.ffn_gemm_v3 import up_situ
from .vendor.ffn_fused_v3 import CONFIG


def install(bb, *, tile):
    from .vendor.inference_contracts_v3 import validate_model, projection_errors
    from kohakumoss.model.stream import scaled_writer
    from kohakumoss.types.context import SHIFT_FFN, SCALE_FFN, GATE_FFN
    validate_model(bb, backend="ffn_v3")
    if tile <= 0 or not getattr(bb, "_ffn_v3_installed", False):
        raise ValueError("Decode reuse requires installed tiled FFNv3")
    originals = []
    plans = []
    receipt = dict(tile=tile, calls=0, decoded_pairs=0,
                   extra_workspace_bytes=3072 * 8192,
                   persistent_cuda_bytes=0, arithmetic="original FP16-fixed v3")
    for block in bb.layers:
        up = block.proj["ffn_up_gate"].base
        down = block.proj["ffn_down"].base
        if type(up) is not FrozenPackedBase or type(down) is not FrozenPackedBase:
            raise ValueError("Tiled decode reuse supports canonical frozen W5/W6")
        if up.decoder is not down.decoder or down.row_tile:
            raise ValueError("One scratch owner and untiled native down required")
        errors = projection_errors(block.proj["ffn_down"])
        if errors:
            raise ValueError("Tiled down projection: " + "; ".join(errors))
        plans.append((block, up, down, bb.stream.block(block.index)))
    for block, up, down, stream in plans:
        originals.append((block, block.ffn_part))

        def ffn(self, x, ctx, _up=up, _down=down, _stream=stream):
            owner = _up.decoder
            up_codes = unpack_base(_up)
            down_codes = torch.empty((_down.qweight.shape[1], _down.k),
                                     device=x.device, dtype=torch.int8)
            # Same launch dimensions as storage.unpack_base for the down shape.
            unpack_into(owner.kernel, _down.qweight, _down.bits, _down.k,
                        down_codes, bn=32, bk=64, num_warps=4)
            owner.stats["unpack_calls"] += 1
            owner.stats["decoded_bytes_written"] += down_codes.numel()

            def write(a, residual, gate, seg, r):
                projected = linear(a, down_codes, _down.wscales)
                _down.calls += 1
                return self.ops.gated_residual(residual, projected, gate, seg, r)

            writer = scaled_writer(write, _stream.in_ffn, _stream.gate_ffn)
            output = torch.empty_like(x)
            for start in range(0, len(x), tile):
                xx = x[start:start + tile]
                seg = ctx.seq.seg_joint[start:start + tile]
                y = self._cast(self.ops.rms_norm_mod(
                    xx, self.norm2.weight, ctx.mod[:, SHIFT_FFN],
                    ctx.mod[:, SCALE_FFN], seg, self.norm2.eps))
                act = up_situ(y, up_codes, _up.wscales, config=CONFIG)
                _up.calls += 1
                del y
                output[start:start + tile] = self._post(
                    writer(act, xx, ctx.mod[:, GATE_FFN], seg, 1.0))
                del act
            receipt["calls"] += 1
            receipt["decoded_pairs"] += 1
            return output

        block.ffn_part = MethodType(ffn, block)

    def restore():
        for block, old in originals:
            block.ffn_part = old
    return receipt, restore
