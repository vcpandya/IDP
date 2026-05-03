"""Notion connector — internal integration token."""
from __future__ import annotations

from idpkit.connectors.base import (
    Connector, ConnectorAuthError, ConnectorAuthType, ConnectorField, ConnectorTool,
)
from idpkit.connectors.http import request

API = "https://api.notion.com/v1"
VERSION = "2022-06-28"


def _h(creds: dict) -> dict:
    return {
        "Authorization": f"Bearer {creds['integration_token']}",
        "Notion-Version": VERSION,
        "Content-Type": "application/json",
    }


async def health_check(creds: dict) -> tuple[bool, str]:
    data = await request("GET", f"{API}/users/me", headers=_h(creds))
    name = data.get("name") or data.get("bot", {}).get("workspace_name") or "Notion"
    return True, name


async def _search_pages(args: dict, creds: dict) -> dict:
    query = args.get("query", "")
    page_size = min(int(args.get("page_size", 10)), 50)
    data = await request(
        "POST", f"{API}/search",
        headers=_h(creds),
        json_body={"query": query, "page_size": page_size},
    )
    results = []
    for r in data.get("results", []):
        title = ""
        props = r.get("properties", {})
        for v in props.values():
            if v.get("type") == "title":
                title = "".join(t.get("plain_text", "") for t in v.get("title", []))
                break
        results.append({"id": r.get("id"), "url": r.get("url"), "type": r.get("object"), "title": title})
    return {"results": results}


async def _create_page(args: dict, creds: dict) -> dict:
    parent_page_id = args.get("parent_page_id", "")
    title = args.get("title", "")
    body = args.get("body", "")
    if not parent_page_id or not title:
        return {"error": "parent_page_id and title are required"}
    payload = {
        "parent": {"page_id": parent_page_id},
        "properties": {"title": [{"type": "text", "text": {"content": title}}]},
        "children": [
            {
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": body}}]},
            }
        ] if body else [],
    }
    data = await request("POST", f"{API}/pages", headers=_h(creds), json_body=payload)
    return {"id": data.get("id"), "url": data.get("url")}


CONNECTOR = Connector(
    id="notion",
    display_name="Notion",
    description="Search Notion pages and create new ones.",
    icon="fa-solid fa-book",
    auth_type=ConnectorAuthType.API_KEY,
    fields=[
        ConnectorField(
            key="integration_token", label="Internal Integration Token", type="password",
            placeholder="secret_...",
            help="Create an internal integration at notion.so/my-integrations, share the relevant pages with it, then paste the secret.",
        ),
    ],
    tools=[
        ConnectorTool(
            name="notion_search_pages",
            description="Search shared Notion pages and databases by free-text query.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "page_size": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
            },
            executor=_search_pages,
        ),
        ConnectorTool(
            name="notion_create_page",
            description="Create a new Notion page under an existing parent page.",
            parameters={
                "type": "object",
                "properties": {
                    "parent_page_id": {"type": "string", "description": "ID of the parent page"},
                    "title": {"type": "string"},
                    "body": {"type": "string", "description": "Optional plain-text body"},
                },
                "required": ["parent_page_id", "title"],
            },
            executor=_create_page,
        ),
    ],
    docs_url="https://developers.notion.com/docs/create-a-notion-integration",
    health_check=health_check,
)
