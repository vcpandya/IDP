"""Google Workspace connector — OAuth2 (Gmail + Drive read scopes)."""
from __future__ import annotations

from idpkit.connectors.base import (
    Connector, ConnectorAuthType, ConnectorField, ConnectorTool,
)
from idpkit.connectors.http import request
from idpkit.connectors.oauth import OAuth2Spec, build_authorize_url, exchange_code, refresh_token

SPEC = OAuth2Spec(
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    scopes=[
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/gmail.send",
    ],
    client_id_env="GOOGLE_OAUTH_CLIENT_ID",
    client_secret_env="GOOGLE_OAUTH_CLIENT_SECRET",
)


def _bearer(creds: dict) -> dict:
    return {"Authorization": f"Bearer {creds['access_token']}"}


async def health_check(creds: dict) -> tuple[bool, str]:
    data = await request(
        "GET", "https://www.googleapis.com/oauth2/v3/userinfo",
        headers=_bearer(creds),
    )
    return True, data.get("email", "Google")


async def _drive_search(args: dict, creds: dict) -> dict:
    query = args.get("query", "")
    page_size = min(int(args.get("page_size", 10)), 50)
    params = {
        "q": f"name contains '{query}' and trashed = false" if query else "trashed = false",
        "pageSize": page_size,
        "fields": "files(id,name,mimeType,webViewLink,modifiedTime)",
    }
    data = await request(
        "GET", "https://www.googleapis.com/drive/v3/files",
        headers=_bearer(creds), params=params,
    )
    return {"files": data.get("files", [])}


async def _gmail_send(args: dict, creds: dict) -> dict:
    import base64
    from email.mime.text import MIMEText

    to = args.get("to", "")
    subject = args.get("subject", "")
    body = args.get("body", "")
    if not to or not subject:
        return {"error": "to and subject are required"}
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    data = await request(
        "POST", "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
        headers={**_bearer(creds), "Content-Type": "application/json"},
        json_body={"raw": raw},
    )
    return {"id": data.get("id"), "threadId": data.get("threadId")}


def _build_authorize(state: str, redirect_uri: str) -> str:
    return build_authorize_url(SPEC, state, redirect_uri)


async def _exchange(code: str, redirect_uri: str) -> dict:
    return await exchange_code(SPEC, code, redirect_uri)


async def _refresh(creds: dict) -> dict:
    rt = creds.get("refresh_token")
    if not rt:
        from idpkit.connectors.base import ConnectorAuthError
        raise ConnectorAuthError("No refresh token stored — please reconnect Google.")
    return await refresh_token(SPEC, rt)


CONNECTOR = Connector(
    id="google",
    display_name="Google Workspace",
    description="Read Drive files and send Gmail messages (OAuth).",
    icon="fa-brands fa-google",
    auth_type=ConnectorAuthType.OAUTH2,
    fields=[],
    tools=[
        ConnectorTool(
            name="google_drive_search",
            description="Search the user's Google Drive by file name.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "page_size": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
            },
            executor=_drive_search,
        ),
        ConnectorTool(
            name="google_gmail_send",
            description="Send an email from the user's Gmail account.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject"],
            },
            executor=_gmail_send,
        ),
    ],
    docs_url="https://developers.google.com/identity/protocols/oauth2",
    oauth_authorize_url_builder=_build_authorize,
    oauth_exchange=_exchange,
    oauth_refresh=_refresh,
    health_check=health_check,
)
