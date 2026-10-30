"""MOSS optimized Base/SFT nodes and normal-queue controls."""
from .nodes import (
    MOSSOptimizedSessionLoader,
    MOSSOptimizedPairedTextEncode,
    MOSSOptimizedSampler,
    MOSSOptimizedVAEDecode,
    MOSSOptimizedReleaseSession,
    MOSSOptimizedControls,
    environment_selection,
)
from .configuration import MOSSOptimizedConfiguration
from .runtime.environment_service import register_routes
from .runtime.setup_service import register_setup_routes

NODE_CLASS_MAPPINGS = {
    "MOSSOptimizedSessionLoader": MOSSOptimizedSessionLoader,
    "MOSSOptimizedPairedTextEncode": MOSSOptimizedPairedTextEncode,
    "MOSSOptimizedSampler": MOSSOptimizedSampler,
    "MOSSOptimizedVAEDecode": MOSSOptimizedVAEDecode,
    "MOSSOptimizedReleaseSession": MOSSOptimizedReleaseSession,
    "MOSSOptimizedControls": MOSSOptimizedControls,
    "MOSSOptimizedConfiguration": MOSSOptimizedConfiguration,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSOptimizedSessionLoader": "MOSS · 低显存模型与预算",
    "MOSSOptimizedPairedTextEncode": "MOSS · 正负提示词编码",
    "MOSSOptimizedSampler": "MOSS · 优化采样",
    "MOSSOptimizedVAEDecode": "MOSS · VAE 解码",
    "MOSSOptimizedReleaseSession": "MOSS · 出图后释放",
    "MOSSOptimizedControls": "MOSS · 预热与卸载",
    "MOSSOptimizedConfiguration": "MOSS · 配置与准备",
}
register_routes(environment_selection)
register_setup_routes()
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
