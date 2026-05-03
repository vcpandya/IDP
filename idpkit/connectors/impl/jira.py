"""Jira connector — basic auth with email + API token."""
from __future__ import annotations

import base64

from idpkit.connectors.base import (
    Connector, ConnectorAuthType, ConnectorField, ConnectorTool,
)
from idpkit.connectors.http import request


def _h(creds: dict) -> dict:
    auth = base64.b64encode(f"{creds['email']}:{creds['api_token']}".encode("utf-8")).decode("ascii")
    return {
        "Authorization": f"Basic {auth}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _site(creds: dict) -> str:
    site = creds["site"].rstrip("/")
    if not site.startswith("http"):
        site = f"https://{site}"
    return site


async def health_check(creds: dict) -> tuple[bool, str]:
    data = await request("GET", f"{_site(creds)}/rest/api/3/myself", headers=_h(creds))
    return True, data.get("emailAddress", "Jira")


async def _search_issues(args: dict, creds: dict) -> dict:
    jql = args.get("jql", "")
    max_results = min(int(args.get("max_results", 20)), 50)
    data = await request(
        "POST", f"{_site(creds)}/rest/api/3/search",
        headers=_h(creds),
        json_body={"jql": jql, "maxResults": max_results, "fields": ["summary", "status", "assignee"]},
    )
    return {"issues": [
        {
            "key": i.get("key"),
            "summary": i.get("fields", {}).get("summary"),
            "status": i.get("fields", {}).get("status", {}).get("name"),
        }
        for i in data.get("issues", [])
    ]}


async def _create_issue(args: dict, creds: dict) -> dict:
    project_key = args.get("project_key", "")
    summary = args.get("summary", "")
    description = args.get("description", "")
    issue_type = args.get("issue_type", "Task")
    if not project_key or not summary:
        return {"error": "project_key and summary are required"}
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
            "description": {
                "type": "doc", "version": 1,
                "content": [{"type": "paragraph",
                             "content": [{"type": "text", "text": description or ""}]}],
            },
        }
    }
    data = await request(
        "POST", f"{_site(creds)}/rest/api/3/issue",
        headers=_h(creds), json_body=payload,
    )
    return {"key": data.get("key"), "id": data.get("id")}


CONNECTOR = Connector(
    id="jira",
    display_name="Jira",
    description="Search and create Jira issues.",
    icon="fa-brands fa-jira",
    auth_type=ConnectorAuthType.COMPOSITE,
    fields=[
        ConnectorField(key="site", label="Site URL", type="text", placeholder="your-company.atlassian.net"),
        ConnectorField(key="email", label="Account Email", type="text"),
        ConnectorField(
            key="api_token", label="API Token", type="password",
            help="Create at id.atlassian.com/manage-profile/security/api-tokens.",
        ),
    ],
    tools=[
        ConnectorTool(
            name="jira_search_issues",
            description="Search Jira issues using JQL.",
            parameters={
                "type": "object",
                "properties": {
                    "jql": {"type": "string", "description": "JQL query, e.g. 'project = ABC AND status = Open'"},
                    "max_results": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
                },
                "required": ["jql"],
            },
            executor=_search_issues,
        ),
        ConnectorTool(
            name="jira_create_issue",
            description="Create a new Jira issue in a given project.",
            parameters={
                "type": "object",
                "properties": {
                    "project_key": {"type": "string"},
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "issue_type": {"type": "string", "default": "Task"},
                },
                "required": ["project_key", "summary"],
            },
            executor=_create_issue,
        ),
    ],
    docs_url="https://developer.atlassian.com/cloud/jira/platform/basic-auth-for-rest-apis/",
    health_check=health_check,
)
