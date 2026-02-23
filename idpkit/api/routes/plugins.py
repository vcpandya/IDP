"""IDP Kit plugin management API routes."""

from fastapi import APIRouter, Depends

from idpkit.api.deps import get_current_user
from idpkit.plugins import plugin_manager

router = APIRouter(prefix="/api/plugins", tags=["plugins"])


@router.get("/")
async def list_plugins(user=Depends(get_current_user)):
    """List all loaded plugins."""
    return {"plugins": plugin_manager.list_plugins()}


@router.get("/tools")
async def list_plugin_tools(user=Depends(get_current_user)):
    """List all tools registered by plugins."""
    tools = []
    for name, tool in plugin_manager.tools.items():
        tools.append({
            "name": tool.name,
            "display_name": tool.display_name,
            "description": tool.description,
            "options_schema": tool.options_schema,
        })
    return {"tools": tools}
