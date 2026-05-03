"""Slack connector — bot-token auth (xoxb-...)."""
from __future__ import annotations

from idpkit.connectors.base import (
    Connector, ConnectorAuthError, ConnectorAuthType, ConnectorField, ConnectorTool,
)
from idpkit.connectors.http import request

API = "https://slack.com/api"


def _auth_headers(creds: dict) -> dict:
    return {"Authorization": f"Bearer {creds['bot_token']}", "Content-Type": "application/json; charset=utf-8"}


async def health_check(creds: dict) -> tuple[bool, str]:
    data = await request("POST", f"{API}/auth.test", headers=_auth_headers(creds), json_body={})
    if not data.get("ok"):
        raise ConnectorAuthError(f"Slack auth.test failed: {data.get('error', 'unknown')}")
    return True, f"{data.get('team', '')} / {data.get('user', '')}"


async def _send_message(args: dict, creds: dict) -> dict:
    channel = args.get("channel", "")
    text = args.get("text", "")
    if not channel or not text:
        return {"error": "channel and text are required"}
    data = await request(
        "POST", f"{API}/chat.postMessage",
        headers=_auth_headers(creds),
        json_body={"channel": channel, "text": text},
    )
    if not data.get("ok"):
        return {"error": data.get("error", "send failed")}
    return {"ok": True, "channel": data.get("channel"), "ts": data.get("ts")}


async def _list_channels(args: dict, creds: dict) -> dict:
    limit = min(int(args.get("limit", 50)), 200)
    data = await request(
        "GET", f"{API}/conversations.list",
        headers=_auth_headers(creds),
        params={"limit": limit, "exclude_archived": "true", "types": "public_channel,private_channel"},
    )
    if not data.get("ok"):
        return {"error": data.get("error", "list failed")}
    return {"channels": [
        {"id": c["id"], "name": c["name"], "is_private": c.get("is_private", False)}
        for c in data.get("channels", [])
    ]}


CONNECTOR = Connector(
    id="slack",
    display_name="Slack",
    description="Send messages and list channels in your Slack workspace.",
    icon="fa-brands fa-slack",
    auth_type=ConnectorAuthType.API_KEY,
    fields=[
        ConnectorField(
            key="bot_token", label="Bot User OAuth Token", type="password",
            placeholder="xoxb-...",
            help="Create a Slack app at api.slack.com/apps, add bot scopes (chat:write, channels:read), install to your workspace, and paste the Bot User OAuth Token.",
        ),
    ],
    tools=[
        ConnectorTool(
            name="slack_send_message",
            description="Send a message to a Slack channel by ID or name (e.g. C123 or #general).",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Channel ID or #channel-name"},
                    "text": {"type": "string", "description": "Message text (mrkdwn)"},
                },
                "required": ["channel", "text"],
            },
            executor=_send_message,
        ),
        ConnectorTool(
            name="slack_list_channels",
            description="List Slack channels the bot has access to.",
            parameters={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200}},
            },
            executor=_list_channels,
        ),
    ],
    docs_url="https://api.slack.com/authentication/token-types#bot",
    health_check=health_check,
)
