"""MOSS optimized SFT nodes and normal-queue controls."""
from .nodes import NODE_CLASS_MAPPINGS,NODE_DISPLAY_NAME_MAPPINGS
from .nodes import environment_selection
from .runtime.environment_service import register_routes
register_routes(environment_selection)
from .configuration import MOSSOptimizedConfiguration
from .runtime.setup_service import register_setup_routes
NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, "MOSSOptimizedConfiguration": MOSSOptimizedConfiguration}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, "MOSSOptimizedConfiguration": "MOSS · 配置与准备"}
register_setup_routes()
WEB_DIRECTORY="./web"
__all__=["NODE_CLASS_MAPPINGS","NODE_DISPLAY_NAME_MAPPINGS","WEB_DIRECTORY"]
