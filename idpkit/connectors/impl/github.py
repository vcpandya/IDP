"""GitHub connector — Personal Access Token (classic or fine-grained)."""
from __future__ import annotations

from idpkit.connectors.base import (
    Connector, ConnectorAuthType, ConnectorField, ConnectorTool,
)
from idpkit.connectors.http import request

API = "https://api.github.com"


def _h(creds: dict) -> dict:
    return {
        "Authorization": f"Bearer {creds['token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def health_check(creds: dict) -> tuple[bool, str]:
    data = await request("GET", f"{API}/user", headers=_h(creds))
    return True, f"@{data.get('login', 'github')}"


async def _list_repos(args: dict, creds: dict) -> dict:
    per_page = min(int(args.get("per_page", 20)), 100)
    data = await request(
        "GET", f"{API}/user/repos",
        headers=_h(creds),
        params={"per_page": per_page, "sort": "updated"},
    )
    return {"repos": [
        {"full_name": r.get("full_name"), "private": r.get("private"), "url": r.get("html_url")}
        for r in data
    ]}


async def _create_issue(args: dict, creds: dict) -> dict:
    repo = args.get("repo", "")
    title = args.get("title", "")
    body = args.get("body", "")
    if not repo or not title:
        return {"error": "repo (owner/name) and title are required"}
    data = await request(
        "POST", f"{API}/repos/{repo}/issues",
        headers=_h(creds),
        json_body={"title": title, "body": body},
    )
    return {"number": data.get("number"), "url": data.get("html_url")}


async def _search_code(args: dict, creds: dict) -> dict:
    q = args.get("q", "")
    if not q:
        return {"error": "q is required"}
    data = await request(
        "GET", f"{API}/search/code",
        headers=_h(creds),
        params={"q": q, "per_page": 10},
    )
    return {"items": [
        {"path": i.get("path"), "repo": i.get("repository", {}).get("full_name"), "url": i.get("html_url")}
        for i in data.get("items", [])
    ]}


CONNECTOR = Connector(
    id="github",
    display_name="GitHub",
    description="List repos, search code, and create issues on GitHub.",
    icon="fa-brands fa-github",
    auth_type=ConnectorAuthType.API_KEY,
    fields=[
        ConnectorField(
            key="token", label="Personal Access Token", type="password",
            placeholder="ghp_... or github_pat_...",
            help="Create a token at github.com/settings/tokens with repo scope (or fine-grained equivalent).",
        ),
    ],
    tools=[
        ConnectorTool(
            name="github_list_repos",
            description="List the authenticated user's GitHub repositories (most recent first).",
            parameters={
                "type": "object",
                "properties": {"per_page": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100}},
            },
            executor=_list_repos,
        ),
        ConnectorTool(
            name="github_create_issue",
            description="Create a new GitHub issue in a repository.",
            parameters={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/name"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["repo", "title"],
            },
            executor=_create_issue,
        ),
        ConnectorTool(
            name="github_search_code",
            description="Search code across accessible GitHub repositories.",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string", "description": "GitHub code search query"}},
                "required": ["q"],
            },
            executor=_search_code,
        ),
    ],
    docs_url="https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens",
    health_check=health_check,
)
