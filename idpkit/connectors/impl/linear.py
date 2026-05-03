"""Linear connector — Personal API key."""
from __future__ import annotations

from idpkit.connectors.base import (
    Connector, ConnectorAuthType, ConnectorField, ConnectorTool,
)
from idpkit.connectors.http import request

API = "https://api.linear.app/graphql"


def _h(creds: dict) -> dict:
    return {"Authorization": creds["api_key"], "Content-Type": "application/json"}


async def _gql(creds: dict, query: str, variables: dict | None = None) -> dict:
    return await request(
        "POST", API, headers=_h(creds),
        json_body={"query": query, "variables": variables or {}},
    )


async def health_check(creds: dict) -> tuple[bool, str]:
    data = await _gql(creds, "query { viewer { id name email } }")
    v = data.get("data", {}).get("viewer", {})
    return True, v.get("name") or v.get("email") or "Linear"


async def _list_issues(args: dict, creds: dict) -> dict:
    first = min(int(args.get("first", 20)), 50)
    q = """
    query ($first: Int!) {
      issues(first: $first, orderBy: updatedAt) {
        nodes { id identifier title state { name } url }
      }
    }
    """
    data = await _gql(creds, q, {"first": first})
    return {"issues": data.get("data", {}).get("issues", {}).get("nodes", [])}


async def _create_issue(args: dict, creds: dict) -> dict:
    team_id = args.get("team_id", "")
    title = args.get("title", "")
    description = args.get("description", "")
    if not team_id or not title:
        return {"error": "team_id and title are required"}
    q = """
    mutation ($input: IssueCreateInput!) {
      issueCreate(input: $input) {
        success issue { id identifier url }
      }
    }
    """
    data = await _gql(creds, q, {"input": {"teamId": team_id, "title": title, "description": description}})
    res = data.get("data", {}).get("issueCreate", {})
    if not res.get("success"):
        return {"error": "Linear issueCreate returned success=false"}
    return res.get("issue", {})


CONNECTOR = Connector(
    id="linear",
    display_name="Linear",
    description="Browse and create Linear issues.",
    icon="fa-solid fa-list-check",
    auth_type=ConnectorAuthType.API_KEY,
    fields=[
        ConnectorField(
            key="api_key", label="Personal API Key", type="password",
            placeholder="lin_api_...",
            help="Create a personal key at linear.app → Settings → API → Personal API keys.",
        ),
    ],
    tools=[
        ConnectorTool(
            name="linear_list_issues",
            description="List recent Linear issues (most recently updated first).",
            parameters={
                "type": "object",
                "properties": {"first": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50}},
            },
            executor=_list_issues,
        ),
        ConnectorTool(
            name="linear_create_issue",
            description="Create a new Linear issue under a specific team.",
            parameters={
                "type": "object",
                "properties": {
                    "team_id": {"type": "string", "description": "Linear team ID (UUID)"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["team_id", "title"],
            },
            executor=_create_issue,
        ),
    ],
    docs_url="https://developers.linear.app/docs/graphql/working-with-the-graphql-api",
    health_check=health_check,
)
