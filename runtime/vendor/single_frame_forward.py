"""Inference-only lifetime change: embed and original plan loop share one frame.

No operation, plan step, runner call, stream cast, mask or head expression changes.
The initial embedded tensor loses its final outer-frame reference after step zero.
"""
from functools import partial
from types import MethodType
import torch
from kohakumoss.model.backbone import MossBackbone, runs_on
from kohakumoss.model.runner import PlainRunner
from kohakumoss.types.context import BlockCtx

def _forward(plan, model, img, text, sigma, seq):
    if torch.is_grad_enabled() or model.training:
        raise RuntimeError("Single-frame forward is inference only")
    mod, head = model.conditioning(sigma)
    x = model.embed(img, text, seq)
    ctx = BlockCtx(seq, mod, model.rope.bands, model.residual_scale)
    for step in plan:
        x = step(model, x, ctx, None)
    return model.head(x, head, seq)

def _forward_masked(plan, model, img, text, sigma, seq, mask):
    if torch.is_grad_enabled() or model.training:
        raise RuntimeError("Single-frame forward is inference only")
    mod, head = model.conditioning(sigma)
    x = model.embed(img, text, seq)
    ctx = BlockCtx(seq, mod, model.rope.bands, model.residual_scale)
    run = runs_on(mask, seq)
    for step in plan:
        x = step(model, x, ctx, run)
    return model.head(x, head, seq)

def install(model):
    if type(model) is not MossBackbone or type(model.runner) is not PlainRunner:
        raise TypeError("Qualified only for the canonical MossBackbone / PlainRunner")
    if "_trunk" in model.__dict__ or "_plan_forward" in model.__dict__:
        raise ValueError("Refuse an already overridden execution plan")
    old_default = model._default
    def plan_forward(self, skip_layers=(), segment_skip_layers=()):
        skip = frozenset(skip_layers)
        if segment_skip_layers:
            segment = frozenset(segment_skip_layers) - skip
            return partial(_forward_masked, self._plan(skip, segment))
        return partial(_forward, self._plan(skip))
    model._plan_forward = MethodType(plan_forward, model)
    model._default = model._plan_forward()
    def restore():
        del model._plan_forward
        model._default = old_default
    return {"mode":"single_frame_original_plan", "inference_only":True,
            "runner":"unchanged PlainRunner", "plan_steps":"original _plan",
            "arithmetic_changed":False, "outer_embed_reference_removed":True}, restore
