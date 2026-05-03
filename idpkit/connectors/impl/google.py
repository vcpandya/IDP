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
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/calendar.events",
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


async def _sheets_read_range(args: dict, creds: dict) -> dict:
    spreadsheet_id = args.get("spreadsheet_id", "")
    rng = args.get("range", "")
    if not spreadsheet_id or not rng:
        return {"error": "spreadsheet_id and range are required"}
    data = await request(
        "GET",
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{rng}",
        headers=_bearer(creds),
    )
    return {"range": data.get("range"), "values": data.get("values", [])}


async def _sheets_append_row(args: dict, creds: dict) -> dict:
    spreadsheet_id = args.get("spreadsheet_id", "")
    rng = args.get("range", "")
    values = args.get("values", [])
    if not spreadsheet_id or not rng or not isinstance(values, list):
        return {"error": "spreadsheet_id, range, and values (list) are required"}
    rows = values if values and isinstance(values[0], list) else [values]
    data = await request(
        "POST",
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{rng}:append",
        headers={**_bearer(creds), "Content-Type": "application/json"},
        params={"valueInputOption": "USER_ENTERED"},
        json_body={"values": rows},
    )
    updates = data.get("updates", {}) if isinstance(data, dict) else {}
    return {"updated_range": updates.get("updatedRange"), "updated_rows": updates.get("updatedRows")}


async def _calendar_list_events(args: dict, creds: dict) -> dict:
    calendar_id = args.get("calendar_id", "primary")
    max_results = min(int(args.get("max_results", 10)), 50)
    params = {"maxResults": max_results, "singleEvents": "true", "orderBy": "startTime"}
    if args.get("time_min"):
        params["timeMin"] = args["time_min"]
    data = await request(
        "GET",
        f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
        headers=_bearer(creds), params=params,
    )
    return {"events": [
        {
            "id": e.get("id"),
            "summary": e.get("summary"),
            "start": e.get("start"),
            "end": e.get("end"),
            "htmlLink": e.get("htmlLink"),
        }
        for e in data.get("items", [])
    ]}


async def _calendar_create_event(args: dict, creds: dict) -> dict:
    calendar_id = args.get("calendar_id", "primary")
    summary = args.get("summary", "")
    start = args.get("start", "")
    end = args.get("end", "")
    if not summary or not start or not end:
        return {"error": "summary, start, and end (RFC3339 datetimes) are required"}
    payload = {
        "summary": summary,
        "description": args.get("description", ""),
        "start": {"dateTime": start},
        "end": {"dateTime": end},
    }
    data = await request(
        "POST",
        f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
        headers={**_bearer(creds), "Content-Type": "application/json"},
        json_body=payload,
    )
    return {"id": data.get("id"), "htmlLink": data.get("htmlLink")}


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
        ConnectorTool(
            name="google_sheets_read_range",
            description="Read a range of cells from a Google Sheets spreadsheet.",
            parameters={
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string"},
                    "range": {"type": "string", "description": "A1 notation, e.g. 'Sheet1!A1:C20'"},
                },
                "required": ["spreadsheet_id", "range"],
            },
            executor=_sheets_read_range,
        ),
        ConnectorTool(
            name="google_sheets_append_row",
            description="Append one or more rows to a Google Sheets range.",
            parameters={
                "type": "object",
                "properties": {
                    "spreadsheet_id": {"type": "string"},
                    "range": {"type": "string", "description": "A1 notation of the table, e.g. 'Sheet1!A:C'"},
                    "values": {
                        "type": "array",
                        "description": "A single row (array of strings) or an array of rows.",
                        "items": {},
                    },
                },
                "required": ["spreadsheet_id", "range", "values"],
            },
            executor=_sheets_append_row,
        ),
        ConnectorTool(
            name="google_calendar_list_events",
            description="List upcoming events on a Google Calendar.",
            parameters={
                "type": "object",
                "properties": {
                    "calendar_id": {"type": "string", "default": "primary"},
                    "max_results": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                    "time_min": {"type": "string", "description": "RFC3339 lower bound, e.g. 2025-01-01T00:00:00Z"},
                },
            },
            executor=_calendar_list_events,
        ),
        ConnectorTool(
            name="google_calendar_create_event",
            description="Create an event on a Google Calendar.",
            parameters={
                "type": "object",
                "properties": {
                    "calendar_id": {"type": "string", "default": "primary"},
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "start": {"type": "string", "description": "RFC3339 start datetime"},
                    "end": {"type": "string", "description": "RFC3339 end datetime"},
                },
                "required": ["summary", "start", "end"],
            },
            executor=_calendar_create_event,
        ),
    ],
    docs_url="https://developers.google.com/identity/protocols/oauth2",
    oauth_authorize_url_builder=_build_authorize,
    oauth_exchange=_exchange,
    oauth_refresh=_refresh,
    health_check=health_check,
)
