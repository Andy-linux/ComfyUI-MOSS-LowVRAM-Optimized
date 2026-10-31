"""Portable local model resolution through ComfyUI's registered search paths.

Opening a workflow never downloads files. Optional models.local.json maps asset
IDs to existing local files; it is machine configuration, excluded from releases.
"""
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "sft_w5": ("diffusion_models", ("moss_image_sft_w5.safetensors", "q5_ridge_gptq.safetensors")),
    "sft_w6": ("diffusion_models", ("moss_image_sft_w6.safetensors", "q6_direct.safetensors")),
    "sft_w8": ("diffusion_models", ("moss_image_sft_int8_convrot.safetensors", "moss-image-sft-dit-int8-convrot.safetensors")),
    "base_w8": ("diffusion_models", ("moss_image_base_int8_convrot.safetensors", "moss-image-base-dit-int8-convrot.safetensors")),
    "base_fp8": ("diffusion_models", ("moss_image_base_kohaku_fp8_row.safetensors",)),
    "sft_fp8": ("diffusion_models", ("moss_image_sft_kohaku_fp8_row.safetensors",)),
    "te_int8": ("text_encoders", ("moss-image-qwen35-int8-convrot-v2.safetensors", "moss-image-qwen35-int8-convrot.safetensors", "qwen3.5_9b_int8_convrot.safetensors")),
    "te_int8_comfy": ("text_encoders", ("qwen3.5_9b_int8_convrot.safetensors",)),
    "vae_flux2_small": ("vae", ("flux2-vae_small.safetensors",)),
    "te_w4": ("text_encoders", ("qwen3.5_9b_w4a8_gptq.safetensors",)),
    "te_original": ("text_encoders", ("qwen3.5_9b_bf16.safetensors", "MOSS-Image-Base/text_encoder", "MOSS-Image-SFT/text_encoder")),
    "vae": ("vae", ("moss-image-vae.safetensors", "moss_image_vae.safetensors")),
}


def local_overrides():
    path = Path(os.environ.get("KOHAKU_MOSS_MODEL_CONFIG", PACKAGE_ROOT / "models.local.json"))
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or any(not isinstance(v, str) for v in data.values()):
        raise ValueError(f"Expected asset ID to local path mapping in {path}")
    return data


def _existing(path):
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir() and (path / "model.safetensors.index.json").is_file():
        return path.resolve()
    return None


def _registered(category, name):
    import folder_paths
    # folder_paths owns configured extra_model_paths.yaml and ordinary model dirs.
    path = folder_paths.get_full_path(category, name)
    if path:
        return _existing(path)
    for root in folder_paths.get_folder_paths(category):
        path = _existing(Path(root) / name)
        if path:
            return path
    return None


def resolve_asset(asset_id, selected="auto"):
    if asset_id not in ASSETS:
        raise ValueError(f"Unknown model asset ID: {asset_id}")
    category, names = ASSETS[asset_id]
    if selected and selected != "auto":
        # Absolute paths are useful for explicit local benchmark/config callers.
        path = _existing(selected) if Path(selected).is_absolute() else _registered(category, selected)
        if path:
            return path
        raise FileNotFoundError(f"Selected {category} model is missing: {selected}")
    override = local_overrides().get(asset_id)
    if override:
        path = _existing(override)
        if path:
            return path
        raise FileNotFoundError(f"Local mapping for {asset_id} is missing: {override}")
    for name in names:
        path = _registered(category, name)
        if path:
            return path
    import folder_paths
    for relative in folder_paths.get_filename_list(category):
        if Path(relative).name in names:
            path = _registered(category, relative)
            if path:
                return path
    raise FileNotFoundError(
        f"Missing {asset_id}: put {names[0]} in ComfyUI/models/{category}, "
        "or select an existing checkpoint. See MODEL_FILES.md and "
        "tools/download_models.py --help for verified downloads."
    )


def checkpoint_choices(category):
    import folder_paths
    names = set(folder_paths.get_filename_list(category))
    if category == "text_encoders":
        # HF text encoders are selected as a directory, not as one of its shards.
        for root in folder_paths.get_folder_paths(category):
            root = Path(root)
            if root.is_dir():
                for index in root.rglob("model.safetensors.index.json"):
                    relative = index.parent.relative_to(root).as_posix()
                    if relative != ".":
                        names.add(relative)
    return ["auto"] + sorted(names)


def comfy_root():
    import folder_paths
    return Path(folder_paths.__file__).resolve().parent
