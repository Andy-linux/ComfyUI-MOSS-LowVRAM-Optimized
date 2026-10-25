"""Fail-closed CPU-inspectable FFN v3 installer preflight.

Validated by installers BEFORE changing any block methods.
Only the measured no-adapter inference contract is accepted.
This helper performs no CUDA tensor allocations. Real runtime architecture query
is only reached after structural checks and explicit CUDA device validation.
"""
import torch


def validate_capability(capability, *, backend="ffn_v3", original_packed_only=False):
    sm=tuple(capability)
    if sm==(8,9):
        return
    if sm==(7,5) and backend=="ffn_v3" and original_packed_only:
        import triton
        if str(triton.__version__).split(".")[:2]==["3","2"]:
            return
        raise ValueError("SM75 FFN v3 requires the separately qualified uniform Triton 3.2 environment")
    raise ValueError("FFN v3 architecture/storage not qualified: SM89 original forms, or SM75 Triton3.2 original packed W5/W6 only")


def projection_errors(up):
    """Structural checks only: CPU-testable without querying CUDA."""
    from kohakumoss.model.linear import MossLinear
    from ..storage import FrozenPackedBase
    from ..storage import FrozenInt8Base
    errors = []
    if type(up) is not MossLinear:
        return ["ffn_up_gate must be exact MossLinear; subclasses/alternate forwards unsupported"]
    if up.bias is not None:
        errors.append("ffn_up_gate bias would be bypassed")
    if up.compute is not None:
        errors.append("ffn_up_gate alternate compute cache unsupported")
    if "forward" in vars(up):
        errors.append("ffn_up_gate instance forward override/LoRA/LyCORIS would be bypassed")
    if up._impl is not MossLinear._stored_nobias:
        errors.append("ffn_up_gate must retain original _stored_nobias implementation")
    if up._forward_hooks or up._forward_pre_hooks:
        errors.append("ffn_up_gate forward/pre-forward hooks would be bypassed")
    if up.training:
        errors.append("FFN v3 is evaluation/inference only")
    if any(p.requires_grad for p in up.parameters()):
        errors.append("FFN v3 cannot preserve trainable-parameter gradients")
    if set(up._modules) != {"base"}:
        errors.append("ffn_up_gate additional child module/adapter unsupported")
    if any(name != "bias" or value is not None for name, value in up._parameters.items()):
        errors.append("ffn_up_gate unexpected dense/additional parameter")
    base = up.base
    if type(base) not in (FrozenPackedBase, FrozenInt8Base):
        errors.append("only exact original FrozenPackedBase or FrozenInt8Base supported")
    elif "linear" in vars(base):
        errors.append("custom base.linear override would be bypassed")
    return errors


def validate_model(bb, *, backend="ffn_v3"):
    all_errors = []
    if not len(bb.layers):
        raise ValueError("Cannot install FFN inference on an empty backbone")
    for i, block in enumerate(bb.layers):
        up = block.proj["ffn_up_gate"]
        all_errors += ["layer " + str(i) + ": " + x for x in projection_errors(up)]
        if up.in_features != 3072 or up.out_features != 16384:
            all_errors.append("layer " + str(i) + ": only validated MOSS D3072/F8192 shape supported")
        down = block.proj["ffn_down"]
        if down.bias is not None or down.compute is not None:
            all_errors.append("layer " + str(i) + ": down bias/alternate compute unsupported")
        from ..storage import FrozenPackedBase
        from ..storage import FrozenInt8Base
        if type(down.base) not in (FrozenPackedBase, FrozenInt8Base):
            all_errors.append("layer " + str(i) + ": down base must be exact original frozen storage")
        elif "linear" in vars(down.base):
            all_errors.append("layer " + str(i) + ": existing down base.linear override/duplicate installation")
        if up.weight_like.dtype not in (torch.float16,torch.bfloat16) or down.weight_like.dtype != up.weight_like.dtype:
            all_errors.append("layer " + str(i) + ": matching FP16/BF16 activation boundary required")
        if up.weight_like.device.type != "cuda":
            all_errors.append("layer " + str(i) + ": install after explicit CUDA model placement")
        for leaf, lin in (("up", up), ("down", down)):
            base = lin.base
            if type(base) is FrozenPackedBase:
                payload, scales = base.qweight, base.wscales
                if base.bits not in (5, 6) or tuple(payload.shape) != ((base.k * base.bits + 7)//8, lin.out_features):
                    all_errors.append("layer " + str(i) + ": original packed shape/bits contract failed")
                expected_dtype = torch.uint8
            elif type(base) is FrozenInt8Base:
                payload, scales = base.codes, base.scales
                if tuple(payload.shape) != (lin.out_features, lin.in_features):
                    all_errors.append("layer " + str(i) + ": original INT8 shape contract failed")
                expected_dtype = torch.int8
            else:
                continue
            if payload.dtype != expected_dtype or not payload.is_contiguous():
                all_errors.append("layer " + str(i) + ": original integer layout/dtype required")
            if scales.dtype != torch.float32 or tuple(scales.shape) != (lin.out_features, 1):
                all_errors.append("layer " + str(i) + ": original FP32 row scales required")
            if payload.device != up.weight_like.device or scales.device != up.weight_like.device:
                all_errors.append("layer " + str(i) + ": FFN payload/scales must share explicit CUDA boundary device")
    if all_errors:
        raise ValueError("Unsupported FFN v3 model state: " + "; ".join(all_errors))
    devices = {block.proj["ffn_up_gate"].weight_like.device for block in bb.layers}
    if len(devices) != 1:
        raise ValueError("All FFN v3 projections must share one CUDA device")
    device = next(iter(devices))
    capability=torch.cuda.get_device_capability(device)
    from ..storage import FrozenPackedBase
    packed_only=all(type(block.proj[key].base) is FrozenPackedBase for block in bb.layers for key in ("ffn_up_gate","ffn_down"))
    if capability[0]<8 and any(block.proj["ffn_up_gate"].weight_like.dtype==torch.bfloat16 for block in bb.layers):
        raise ValueError("BF16 requires SM80 or newer")
    validate_capability(capability,backend=backend,original_packed_only=packed_only)
    return dict(backend=backend,qualified_architecture="SM"+str(capability[0]*10+capability[1]),device=str(device),
                exact_original_MossLinear_stored_nobias=True,bias=False,
                LoRA_or_forward_override=False,forward_hooks=False,
                dtype=str(bb.layers[0].proj["ffn_up_gate"].weight_like.dtype),D=3072,F=8192,check_scope="pre-install; do not attach adapters after install")
