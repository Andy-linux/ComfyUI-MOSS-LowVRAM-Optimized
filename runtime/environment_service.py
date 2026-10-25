"""Shared read-only checks for the node, CLI and frontend setup dialog."""
from functools import lru_cache
import asyncio
import json

from .environment import inspect_environment, evaluate_selection


@lru_cache(maxsize=1)
def current_environment():
    # Packages cannot be safely swapped in an active ComfyUI process. A refresh
    # observes the current process; installing a package still requires restart.
    return inspect_environment(device_index=0, check_extensions=True)


def check_selection(selection, *, refresh=False):
    if refresh:
        current_environment.cache_clear()
    report = current_environment()
    return {"environment": report, "evaluation": evaluate_selection(report, selection)}


def validation_message(evaluation):
    if not evaluation["errors"]:
        return True
    details = [item["message"] for item in evaluation["errors"]]
    return "MOSS environment: " + "; ".join(details) + ". Use the node's environment check for installation guidance."


def register_routes(selection_mapper):
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/moss_lowvram/environment")
    async def environment_report(request):
        try:
            payload = await request.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("selection", {}), dict):
                raise ValueError("Expected an object containing a selection object")
            selection = selection_mapper(payload.get("selection", {}))
            # Metadata/extension import and nvidia-smi can take seconds. Keep the
            # ComfyUI event loop responsive while performing this read-only check.
            result = await asyncio.to_thread(check_selection, selection, refresh=payload.get("refresh") is True)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response(result)
