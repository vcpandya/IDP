"""HubSpot connector — Private App token."""
from __future__ import annotations

from idpkit.connectors.base import (
    Connector, ConnectorAuthType, ConnectorField, ConnectorTool,
)
from idpkit.connectors.http import request

API = "https://api.hubapi.com"


def _h(creds: dict) -> dict:
    return {"Authorization": f"Bearer {creds['access_token']}", "Content-Type": "application/json"}


async def health_check(creds: dict) -> tuple[bool, str]:
    await request("GET", f"{API}/crm/v3/objects/contacts", headers=_h(creds), params={"limit": 1})
    return True, "HubSpot"


async def _search_contacts(args: dict, creds: dict) -> dict:
    query = args.get("query", "")
    payload = {"query": query, "limit": min(int(args.get("limit", 10)), 50)}
    data = await request(
        "POST", f"{API}/crm/v3/objects/contacts/search",
        headers=_h(creds), json_body=payload,
    )
    return {"results": [
        {"id": c.get("id"), "properties": c.get("properties", {})}
        for c in data.get("results", [])
    ]}


async def _create_contact(args: dict, creds: dict) -> dict:
    email = args.get("email", "")
    if not email:
        return {"error": "email is required"}
    props = {"email": email}
    for k in ("firstname", "lastname", "company", "phone"):
        v = args.get(k)
        if v:
            props[k] = v
    data = await request(
        "POST", f"{API}/crm/v3/objects/contacts",
        headers=_h(creds), json_body={"properties": props},
    )
    return {"id": data.get("id")}


CONNECTOR = Connector(
    id="hubspot",
    display_name="HubSpot",
    description="Search and create CRM contacts in HubSpot.",
    icon="fa-solid fa-people-group",
    auth_type=ConnectorAuthType.API_KEY,
    fields=[
        ConnectorField(
            key="access_token", label="Private App Access Token", type="password",
            placeholder="pat-...",
            help="Create a Private App at app.hubspot.com → Settings → Integrations → Private Apps with crm.objects.contacts scopes.",
        ),
    ],
    tools=[
        ConnectorTool(
            name="hubspot_search_contacts",
            description="Search HubSpot CRM contacts by free-text query.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
            },
            executor=_search_contacts,
        ),
        ConnectorTool(
            name="hubspot_create_contact",
            description="Create a new HubSpot contact.",
            parameters={
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "firstname": {"type": "string"},
                    "lastname": {"type": "string"},
                    "company": {"type": "string"},
                    "phone": {"type": "string"},
                },
                "required": ["email"],
            },
            executor=_create_contact,
        ),
    ],
    docs_url="https://developers.hubspot.com/docs/api/private-apps",
    health_check=health_check,
)
