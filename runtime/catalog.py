"""Pure, import-light configuration catalog shared by nodes and setup routes.

Names describe checkpoint storage separately from the executed matrix arithmetic.
Hardware eligibility belongs exclusively to runtime.environment.evaluate_selection.
"""
import json
from pathlib import Path

CHOICES = {
    "model_family": ["SFT", "Base"],
    "dit_storage": ["Packed W5", "Packed W6", "INT8 ConvRot", "Kohaku FP8 row"],
    "dit_compute": ["INT8 × INT8 → INT32", "FP8", "16-bit"],
    "te_storage": ["Sumomo INT8 ConvRot v2", "Comfy INT8 ConvRot", "Comfy W4 GPTQ", "Original float"],
    "te_compute": ["W8A16", "W8A8", "W4A16 (legacy)", "W4A8", "FP16", "BF16"],
    "profile": ["4 GB", "6 GB", "8 GB", "12 GB", "16 GB", "unlimited"],
    "precision": ["fp16_fixed", "bf16"],
    "attention": ["auto", "Flash FP16/BF16", "cuDNN dense", "Sage QK8 PV16", "Sage QK8 PV16 + H128", "Sage QK8 / PV FP8", "Sage QK8 / PV FP8 fast", "Sage QK8 / PV FP8 fast + H128", "CK INT8"],
    "vae_precision": ["fp32", "bf16"],
    "vae_asset": ["Original MOSS", "Flux2 small repack"],
}
DEFAULTS = {
    "model_family": "SFT", "dit_storage": "Packed W5",
    "dit_compute": "INT8 × INT8 → INT32",
    "te_storage": "Sumomo INT8 ConvRot v2", "te_compute": "W8A16",
    "profile": "6 GB", "precision": "fp16_fixed", "attention": "auto",
    "vae_precision": "fp32", "vae_asset": "Original MOSS",
    "qkv_backend": "auto", "ffn_tile": "auto", "host_arena": True,
    "adapter_cache": False, "conditioning_memory": False,
    "attention_head_group": "auto", "te_memory": "auto", "vae_admission": False,
    "resident_blocks": -1, "payload_pinning": "staging", "te_resident_gib": 3.0,
}
_DIT = {
    ("SFT", "Packed W5"): ("SFT W5", "sft_w5"),
    ("SFT", "Packed W6"): ("SFT W6", "sft_w6"),
    ("SFT", "INT8 ConvRot"): ("SFT W8", "sft_w8"),
    ("Base", "INT8 ConvRot"): ("Base W8A8 INT8 ConvRot", "base_w8"),
    ("Base", "Kohaku FP8 row"): ("Base Kohaku FP8", "base_fp8"),
    ("SFT", "Kohaku FP8 row"): ("SFT Kohaku FP8", "sft_fp8"),
}
_TE_ASSETS = {"Sumomo INT8 ConvRot v2": "te_int8", "Comfy INT8 ConvRot": "te_int8_comfy", "Comfy W4 GPTQ": "te_w4", "Original float": "te_original"}
_TE_MODES = {"W8A16": "INT8 ConvRot · W8A16", "W8A8": "INT8 ConvRot · W8A8", "W4A16 (legacy)": "W4A8 GPTQ", "W4A8": "W4 GPTQ · W4A8", "FP16": "Original FP16", "BF16": "Original BF16"}
_TE_OPS = {"Sumomo INT8 ConvRot v2": ["W8A16", "W8A8"], "Comfy INT8 ConvRot": ["W8A16", "W8A8"], "Comfy W4 GPTQ": ["W4A16 (legacy)", "W4A8"], "Original float": ["FP16", "BF16"]}
_FALLBACK_NAMES = {"base_fp8": "moss_image_base_kohaku_fp8_row.safetensors", "sft_fp8": "moss_image_sft_kohaku_fp8_row.safetensors"}


def configuration_options(values=None):
    """Structural choices only. Server may disable more using canonical policy."""
    v = {**DEFAULTS, **(values or {})}
    result = {}
    for field, choices in CHOICES.items():
        allowed = choices
        if field == "dit_storage":
            allowed = [s for family, s in _DIT if family == v["model_family"]]
        elif field == "dit_compute":
            allowed = ["FP8", "16-bit"] if v["dit_storage"] == "Kohaku FP8 row" else [CHOICES[field][0]]
        elif field == "te_compute":
            allowed = _TE_OPS.get(v["te_storage"], [])
        result[field] = [{"value": c, "label": c, "enabled": c in allowed, "reason": "" if c in allowed else "Incompatible with selected storage/model family"} for c in choices]
    return result


def configuration_capabilities(values=None):
    return {"options": configuration_options(values), "recommended_sampling": {"steps": 50 if (values or {}).get("model_family") == "Base" else 16, "cfg": 3.0, "sampler_name": "euler", "scheduler": "sgm_uniform", "shift": 5.0, "slg_layer": 22}, "sampling_changes_require_explicit_action": True}


def _selection(values):
    if not isinstance(values, dict):
        raise ValueError("Selection must be an object")
    v = {**DEFAULTS, **values}
    for field, options in configuration_options(v).items():
        if not any(o["value"] == v[field] and o["enabled"] for o in options):
            raise ValueError(f"Invalid {field}: {v[field]!r}; incompatible storage/operation or model family")
    return v


def asset_ids_for_selection(values):
    v = _selection(values)
    return [_DIT[v["model_family"], v["dit_storage"]][1], _TE_ASSETS[v["te_storage"]], "vae" if v["vae_asset"] == "Original MOSS" else "vae_flux2_small"]


def checkpoint_defaults(values):
    ids = asset_ids_for_selection(values)
    manifest = json.loads((Path(__file__).resolve().parents[1] / "model_manifest.json").read_text(encoding="utf-8"))
    assets = {a["id"]: a for a in manifest["assets"]}
    result = {}
    for key, asset_id in zip(("dit_checkpoint", "te_checkpoint", "vae_checkpoint"), ids):
        if asset_id == "te_original":
            name = "MOSS-Image-Base/text_encoder"
        else:
            entry = assets[asset_id]
            name = entry["files"][0]["destination"].split("/", 1)[1] if entry["files"] else entry.get("local_filename", _FALLBACK_NAMES.get(asset_id))
        result[key] = name
    return result


def catalog_checkpoint_choices(category):
    """All recommended catalog names, including absent files on a fresh host.

    Original TE is selected as a directory, never as an individual shard.
    Unrelated manifest alternatives are not silently promoted to catalog choices.
    """
    manifest = json.loads((Path(__file__).resolve().parents[1] / "model_manifest.json").read_text(encoding="utf-8"))
    declared = {asset for _, asset in _DIT.values()} | set(_TE_ASSETS.values()) | {"vae", "vae_flux2_small"}
    result = []
    for asset in manifest["assets"]:
        if asset["id"] not in declared:
            continue
        if asset["id"] == "te_original":
            if category == "text_encoders":
                result.append("MOSS-Image-Base/text_encoder")
            continue
        for row in asset.get("files", []):
            folder, name = row["destination"].split("/", 1)
            if folder == category:
                result.append(name)
        if not asset.get("files") and asset["id"] in _FALLBACK_NAMES and category == "diffusion_models":
            result.append(asset.get("local_filename", _FALLBACK_NAMES[asset["id"]]))
    return list(dict.fromkeys(result))


def resolve_configuration(values):
    """Resolve new UI fields into existing session-loader keyword arguments."""
    v = _selection(values)
    result = {key: v[key] for key in DEFAULTS if key not in ("model_family", "dit_storage", "dit_compute", "te_storage", "te_compute", "vae_asset")}
    result["model"] = _DIT[v["model_family"], v["dit_storage"]][0]
    result["text_encoder"] = _TE_MODES[v["te_compute"]]
    result["fp8_activation_mode"] = {"FP8": "fp8", "16-bit": "16bit"}.get(v["dit_compute"], "checkpoint")
    for field, allowed in {
        "qkv_backend": ("auto", "shared", "native", "register", "register3"),
        "ffn_tile": ("auto", 0, 2048),
        "attention_head_group": ("auto", 0, 8, 12, 24),
        "te_memory": ("auto", "ring", "partial", "resident"),
        "payload_pinning": ("staging", "suffix"),
    }.items():
        if v[field] not in allowed or isinstance(v[field], bool):
            raise ValueError(f"Invalid {field}")
    for field in ("host_arena", "adapter_cache", "conditioning_memory", "vae_admission"):
        if type(v[field]) is not bool:
            raise ValueError(f"{field} must be a boolean")
    if type(v["resident_blocks"]) is not int or not -1 <= v["resident_blocks"] <= 40:
        raise ValueError("resident_blocks must be an integer from -1 to 40")
    if type(v["te_resident_gib"]) not in (int, float) or not 0.25 <= v["te_resident_gib"] <= 6:
        raise ValueError("te_resident_gib must be between 0.25 and 6")
    if v["attention"] in ("auto", "Flash FP16/BF16") and v["attention_head_group"] not in ("auto", 0, 24):
        raise ValueError("Flash/auto attention requires an ungrouped head selection")
    for key, default in checkpoint_defaults(v).items():
        result[key] = v.get(key, default)
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"Invalid {key}")
    # Recommended filenames belonging to another catalog variant are a
    # contradictory selection, not a custom checkpoint with invented provenance.
    defaults = checkpoint_defaults(v)
    for field, category in (("dit_checkpoint", "diffusion_models"), ("te_checkpoint", "text_encoders"), ("vae_checkpoint", "vae")):
        if result[field] in catalog_checkpoint_choices(category) and result[field] != defaults[field]:
            raise ValueError(f"{field} belongs to a different catalog model; select matching storage/family or its recommended checkpoint")
    if v["te_storage"] == "Comfy INT8 ConvRot" and result["te_checkpoint"] == "auto":
        raise ValueError("Comfy INT8 ConvRot requires an explicit checkpoint; auto would select Sumomo")
    return result


def environment_selection(values):
    """Translate through the old authoritative node mapper without torch imports."""
    v = resolve_configuration(values)
    candidates = {"SFT W5": "q5_ridge_gptq", "SFT W6": "q6_direct", "SFT W8": "w8_int8c", "Base W8A8 INT8 ConvRot": "base_w8_int8c", "Base Kohaku FP8": "base_fp8", "SFT Kohaku FP8": "sft_fp8"}
    attention = dict(zip(CHOICES["attention"], ["auto", "flash", "cudnn", "sage_fp16", "sage_fp16_rot", "sage_fp8", "sage_fp8_fast", "sage_fp8_rot", "ck_int8"]))
    te_mode = {"W8A16": "w8a16", "W8A8": "w8a8", "W4A16 (legacy)": "auto", "W4A8": "w4a8", "FP16": "original", "BF16": "original"}[values.get("te_compute", DEFAULTS["te_compute"])]
    slots = v["resident_blocks"]
    if type(slots) is not int or not -1 <= slots <= 40:
        raise ValueError("resident_blocks must be an integer from -1 to 40")
    if v["te_memory"] not in ("auto", "ring", "partial", "resident"):
        raise ValueError("Invalid te_memory")
    return {"candidate": candidates[v["model"]], "profile": "unlimited" if v["profile"] == "unlimited" else v["profile"].split()[0] + "g", "precision": v["precision"], "attention": attention[v["attention"]], "te_mode": te_mode, "te_compute_dtype": "bf16" if values.get("te_compute") == "BF16" else "fp16", "vae_precision": v["vae_precision"], "te_memory": v["te_memory"], "resident_blocks": slots, "fp8_activation_mode": v["fp8_activation_mode"]}
