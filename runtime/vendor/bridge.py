"""Experimental ComfyUI sampling bridge to official Kohaku Backbone.

Preserves standard Comfy sampler/CFG/SLG/sigma path. Numerical backbone/backend
confounds are explicit; no equivalence with original Comfy DiT is claimed.
"""

import torch
from kohakumoss.types.seqinfo import build_seqinfo


class KohakuComfyBridge(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.dtype = backbone.video_embedder.weight.dtype

    def forward(
        self,
        x,
        timestep,
        context,
        attention_mask=None,
        transformer_options=None,
        **kwargs,
    ):
        options = transformer_options or {}
        replacements = options.get("patches_replace", {}).get("dit", {})
        skip = tuple(k[1] for k in replacements if k[0] == "single_block")
        if any(k[0] != "single_block" or k[1] != 22 for k in replacements):
            raise ValueError("Only explicit SLG22 bridge supported")
        fn = self.backbone.bind_forward(skip_layers=skip)
        times = (
            torch.as_tensor(timestep, device=x.device, dtype=torch.float32)
            .reshape(-1)
            .expand(x.shape[0])
            / 1000
        )
        outputs = []
        for i in range(x.shape[0]):
            payload_owner = getattr(self.backbone, "_moss_payload_owner", None)
            if payload_owner is not None:
                payload_owner.begin_forward(skip)
            text = context[i]
            if attention_mask is not None:
                text = text[attention_mask[i].bool()]
            if not len(text):
                raise ValueError("Empty conditioning")
            h, w = x.shape[-2:]
            seq = build_seqinfo(torch.tensor([[h, w]]), torch.tensor([len(text)])).to(
                x.device
            )
            image = x[i].flatten(1).transpose(0, 1)
            out = fn(image, text, times[i : i + 1], seq)
            outputs.append(out.transpose(0, 1).reshape(1, -1, h, w))
        return torch.cat(outputs)
