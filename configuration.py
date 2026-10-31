"""Unified main entry point; old loader widgets and node IDs remain unchanged."""
from .nodes import MOSSOptimizedSessionLoader
from .runtime.catalog import CHOICES, DEFAULTS, checkpoint_defaults, resolve_configuration, environment_selection, catalog_checkpoint_choices


class MOSSOptimizedConfiguration(MOSSOptimizedSessionLoader):
    FUNCTION = "configure"
    DESCRIPTION = "Main MOSS configuration. Stored weight format and actual matrix computation are separate. Prepare/check inspects dependencies and offers explicit verified model downloads. Sampling parameters change only through Apply recommended sampling."

    @classmethod
    def INPUT_TYPES(cls):
        from .runtime.assets import checkpoint_choices
        old = MOSSOptimizedSessionLoader.INPUT_TYPES()
        required = {}
        for field in ("model_family", "dit_storage", "dit_compute", "te_storage", "te_compute", "profile", "precision", "attention", "vae_precision", "vae_asset"):
            tooltip = {
                "dit_storage": "Packed W5/W6 decode into INT8 weights for real W8A8 GEMM; storage bits are not multiplication bits. Base supports INT8/FP8 only.",
                "dit_compute": "Integer DiT supports W8A8 only. FP8 uses row-scaled FP8 weights with FP8 activations, or dequantized 16-bit matrix operations. Boundary precision is separate.",
                "te_storage": "Sumomo v2 and Comfy INT8 are distinct files, sizes and hashes. Original float downloads all four shards, index/config and tokenizer.",
                "te_compute": "W8A16/W4A16 dequantize weights for 16-bit activation operations; W8A8/W4A8 execute real INT8 GEMM. Historical W4A16 preserves the old GPTQ numerical path.",
                "precision": "Boundary/nonquantized DiT precision. Does not change stored integer codes or select a W8A16 DiT path.",
            }.get(field, "Selected choices are checked against the server's environment policy. Changing family does not change prompt, seed or sampling steps.")
            required[field] = (CHOICES[field], {"default": DEFAULTS[field], "tooltip": tooltip})
        optional = {}
        for field in ("qkv_backend", "ffn_tile", "host_arena"):
            optional[field] = old["required"][field]
        for field, spec in old["optional"].items():
            if field not in required and not field.endswith("_checkpoint"):
                optional[field] = spec
        defaults = checkpoint_defaults({})
        for field, category in (("dit_checkpoint", "diffusion_models"), ("te_checkpoint", "text_encoders"), ("vae_checkpoint", "vae")):
            # Include pinned basenames even on a fresh installation: a workflow
            # can name missing files before explicit Prepare/download is used.
            names = checkpoint_choices(category)
            names = list(dict.fromkeys([defaults[field]] + catalog_checkpoint_choices(category) + names))
            optional[field] = (names, {"default": defaults[field], "tooltip": "Explicit local checkpoint. Prepare shows pinned downloads only for the selected catalog asset; custom files have no automatic source link."})
        return {"required": required, "optional": optional}

    @classmethod
    def VALIDATE_INPUTS(cls, model_family):
        from .runtime.environment_service import check_selection, validation_message
        try:
            selected = environment_selection({})
            return validation_message(check_selection(selected)["evaluation"])
        except (KeyError, TypeError, ValueError) as error:
            return "Invalid MOSS configuration: " + str(error)

    def configure(self, **kwargs):
        # The legacy loader owns caching, lifetime, final environment validation
        # and construction. No intermediate configuration node is required.
        from .runtime.local_models import runtime_checkpoints
        resolved = resolve_configuration(kwargs)
        resolved.update(runtime_checkpoints(kwargs))
        return self.load_session(**resolved)
