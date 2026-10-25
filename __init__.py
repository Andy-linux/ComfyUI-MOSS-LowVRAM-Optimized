"""MOSS optimized SFT nodes and normal-queue controls."""
from .nodes import NODE_CLASS_MAPPINGS,NODE_DISPLAY_NAME_MAPPINGS
from .nodes import environment_selection
from .runtime.environment_service import register_routes
register_routes(environment_selection)
WEB_DIRECTORY="./web"
__all__=["NODE_CLASS_MAPPINGS","NODE_DISPLAY_NAME_MAPPINGS","WEB_DIRECTORY"]
