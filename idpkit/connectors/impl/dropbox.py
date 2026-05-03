"""Dropbox connector — generated access token."""
from __future__ import annotations

from idpkit.connectors.base import (
    Connector, ConnectorAuthType, ConnectorField, ConnectorTool,
)
from idpkit.connectors.http import request

API = "https://api.dropboxapi.com/2"


def _h(creds: dict) -> dict:
    return {"Authorization": f"Bearer {creds['access_token']}", "Content-Type": "application/json"}


async def health_check(creds: dict) -> tuple[bool, str]:
    data = await request("POST", f"{API}/users/get_current_account", headers=_h(creds), json_body=None)
    return True, data.get("name", {}).get("display_name", "Dropbox")


async def _list_files(args: dict, creds: dict) -> dict:
    path = args.get("path", "")
    data = await request(
        "POST", f"{API}/files/list_folder",
        headers=_h(creds),
        json_body={"path": path, "recursive": False, "limit": min(int(args.get("limit", 50)), 200)},
    )
    return {"entries": [
        {"name": e.get("name"), "path": e.get("path_display"), "tag": e.get(".tag")}
        for e in data.get("entries", [])
    ]}


async def _create_shared_link(args: dict, creds: dict) -> dict:
    path = args.get("path", "")
    if not path:
        return {"error": "path is required"}
    data = await request(
        "POST", f"{API}/sharing/create_shared_link_with_settings",
        headers=_h(creds),
        json_body={"path": path},
    )
    return {"url": data.get("url"), "id": data.get("id")}


CONNECTOR = Connector(
    id="dropbox",
    display_name="Dropbox",
    description="List Dropbox files and create shared links.",
    icon="fa-brands fa-dropbox",
    auth_type=ConnectorAuthType.API_KEY,
    fields=[
        ConnectorField(
            key="access_token", label="Access Token", type="password",
            placeholder="sl....",
            help="Generate an access token at dropbox.com/developers/apps → your app → Generated access token.",
        ),
    ],
    tools=[
        ConnectorTool(
            name="dropbox_list_files",
            description="List files in a Dropbox folder (use empty string for root).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "", "description": "Folder path (e.g. /Documents)"},
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
                },
            },
            executor=_list_files,
        ),
        ConnectorTool(
            name="dropbox_create_shared_link",
            description="Create a shareable link for a Dropbox file or folder.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            executor=_create_shared_link,
        ),
    ],
    docs_url="https://www.dropbox.com/developers/documentation/http/documentation",
    health_check=health_check,
)
