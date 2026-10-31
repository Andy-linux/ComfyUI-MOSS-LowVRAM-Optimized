"""Resolve user-selected files through ComfyUI, independently of download receipts."""
from . import catalog
from .assets import checkpoint_choices, resolve_asset

FIELDS = (("dit_checkpoint", "diffusion_models", "DiT 主模型"),
          ("te_checkpoint", "text_encoders", "文本编码器"),
          ("vae_checkpoint", "vae", "VAE"))


def inspect_models(selection):
    rows = []
    ids = catalog.asset_ids_for_selection(selection)
    for asset_id, (field, category, label) in zip(ids, FIELDS):
        chosen = selection.get(field, "auto")
        try:
            path = resolve_asset(asset_id, chosen)
            ready = True
            message = "已找到本地模型；加载器在运行时检查模型格式，不要求下载记录或推荐哈希。"
        except FileNotFoundError:
            path = None
            ready = False
            message = f"请在 models/{category} 或已注册的额外目录中放入模型，再从下拉框选择；也可使用可选下载。"
        rows.append({"field": field, "asset_id": asset_id, "category": category, "label": label,
                     "selected": chosen, "ready": ready, "message": message,
                     "filename": path.name if path else chosen,
                     "size_bytes": path.stat().st_size if path and path.is_file() else None,
                     "choices": checkpoint_choices(category), "_path": path})
    return rows


def runtime_checkpoints(selection):
    # Each execution stage opens its own file; a TE-only graph does not need VAE/DiT.
    defaults = catalog.checkpoint_defaults(selection)
    return {row["field"]: str(row["_path"]) if row["ready"] else
            defaults[row["field"]] if row["selected"] == "auto" else row["selected"]
            for row in inspect_models(selection)}
