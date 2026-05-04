"""Connector / Connection management API.

Endpoints:
- GET    /api/connectors                    — list available connectors (definitions only)
- GET    /api/connectors/connections        — list current user's connections
- POST   /api/connectors/{cid}/connect      — save credentials for an api-key/composite connector
- POST   /api/connectors/{cid}/test         — test credentials without persisting
- DELETE /api/connectors/connections/{id}   — disconnect a connection
- GET    /api/connectors/{cid}/oauth/start  — begin OAuth2 flow
- GET    /api/connectors/oauth/callback     — OAuth2 redirect-target

Credential rule: cleartext credentials enter only through these POST endpoints,
get encrypted, and are never echoed back. GETs return only public metadata.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.api.deps import get_current_user, get_db
from idpkit.connectors import (
    ConnectorAuthError, ConnectorAuthType, ConnectorError,
    decrypt_credentials, encrypt_credentials, get_connector, list_connectors,
)
from idpkit.connectors.oauth import consume_state, new_state
from idpkit.db.models import Connection, ConnectionAuditLog, User, UserRole, utcnow


def _is_admin(user: User) -> bool:
    return user.role in (UserRole.ADMIN.value, UserRole.SUPERADMIN.value)


def _serialize_connection(row: Connection, viewer: User) -> dict:
    is_owner = row.owner_id == viewer.id
    return {
        "id": row.id,
        "connector_id": row.connector_id,
        "display_name": row.display_name,
        "status": row.status,
        "metadata": row.connection_metadata or {},
        "scope": row.scope or "private",
        "is_shared": (row.scope == "org"),
        "is_owner": is_owner,
        "owner_org": row.owner_org,
        "last_checked_at": row.last_checked_at.isoformat() if row.last_checked_at else None,
        "last_error": row.last_error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/connectors", tags=["connectors"])


@router.get("", summary="List all available connectors")
async def list_all(user: User = Depends(get_current_user)):
    return {"connectors": [c.public_metadata() for c in list_connectors()]}


@router.get("/connections", summary="List connections visible to the current user")
async def list_user_connections(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all connections this user can see.

    Includes the user's own connections plus any connection an admin has
    shared at org level (``scope='org'``). Non-owners may attach skills
    to a shared connection but cannot edit, disconnect, or read its
    credentials.
    """
    from sqlalchemy import or_ as _or
    rows = (await db.execute(
        select(Connection).where(
            _or(Connection.owner_id == user.id, Connection.scope == "org"),
        ).order_by(Connection.created_at.desc())
    )).scalars().all()
    return {"connections": [_serialize_connection(r, user) for r in rows]}


class CredentialBody(BaseModel):
    credentials: dict


def _validate_credentials(connector, credentials: dict) -> dict:
    if not isinstance(credentials, dict):
        raise HTTPException(400, "credentials must be a JSON object")
    sanitized: dict = {}
    for f in connector.fields:
        v = credentials.get(f.key)
        if f.required and not v:
            raise HTTPException(400, f"Missing required field: {f.label} ({f.key})")
        if v is not None:
            sanitized[f.key] = v
    return sanitized


@router.post("/{connector_id}/test", summary="Test credentials without saving")
async def test_credentials(
    connector_id: str,
    body: CredentialBody,
    user: User = Depends(get_current_user),
):
    connector = get_connector(connector_id)
    if not connector:
        raise HTTPException(404, f"Unknown connector: {connector_id}")
    if connector.auth_type == ConnectorAuthType.OAUTH2:
        raise HTTPException(400, "OAuth connectors are tested via the OAuth flow, not credential POST.")
    creds = _validate_credentials(connector, body.credentials)
    if not connector.health_check:
        return {"ok": True, "message": "No health check available; credentials accepted."}
    try:
        ok, label = await connector.health_check(creds)
    except ConnectorAuthError as exc:
        return {"ok": False, "error": str(exc)}
    except ConnectorError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": ok, "account": label}


@router.post("/{connector_id}/connect", summary="Save credentials for an api-key / composite connector")
async def connect(
    connector_id: str,
    body: CredentialBody,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    connector = get_connector(connector_id)
    if not connector:
        raise HTTPException(404, f"Unknown connector: {connector_id}")
    if connector.auth_type == ConnectorAuthType.OAUTH2:
        raise HTTPException(400, "Use the OAuth flow for this connector.")
    creds = _validate_credentials(connector, body.credentials)

    label = ""
    if connector.health_check:
        try:
            ok, label = await connector.health_check(creds)
            if not ok:
                raise HTTPException(400, "Health check failed; credentials rejected.")
        except ConnectorAuthError as exc:
            raise HTTPException(401, f"Authentication failed: {exc}")
        except ConnectorError as exc:
            raise HTTPException(400, f"Connection failed: {exc}")

    return await _persist_connection(db, user.id, connector_id, creds, display_name=label)


async def _persist_connection(
    db: AsyncSession,
    user_id: str,
    connector_id: str,
    creds: dict,
    display_name: str = "",
    metadata: Optional[dict] = None,
    expires_at=None,
) -> dict:
    existing = (await db.execute(
        select(Connection).where(
            Connection.owner_id == user_id,
            Connection.connector_id == connector_id,
        ).order_by(Connection.created_at.desc())
    )).scalars().first()
    encrypted = encrypt_credentials(creds)
    if existing:
        existing.encrypted_credentials = encrypted
        existing.display_name = display_name or existing.display_name
        existing.connection_metadata = metadata or existing.connection_metadata
        existing.status = "active"
        existing.last_checked_at = utcnow()
        existing.last_error = None
        existing.expires_at = expires_at
        row = existing
    else:
        row = Connection(
            owner_id=user_id,
            connector_id=connector_id,
            display_name=display_name or None,
            encrypted_credentials=encrypted,
            connection_metadata=metadata,
            status="active",
            last_checked_at=utcnow(),
            expires_at=expires_at,
        )
        db.add(row)
    await db.commit()
    await db.refresh(row)
    return {
        "id": row.id,
        "connector_id": row.connector_id,
        "display_name": row.display_name,
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.delete("/connections/{connection_id}", summary="Disconnect (delete) a connection")
async def disconnect(
    connection_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    row = (await db.execute(
        select(Connection).where(Connection.id == connection_id)
    )).scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Connection not found")
    # The owner can always remove their own connection. Admins may also
    # remove an org-shared connection that another admin set up.
    if row.owner_id != user.id and not (row.scope == "org" and _is_admin(user)):
        raise HTTPException(403, "Not allowed to disconnect this connection")
    await db.delete(row)
    await db.commit()
    return {"deleted": True, "id": connection_id}


# ---------------------------------------------------------------------------
# Org-level sharing — admins can promote a connection to "scope=org" so
# any user in the deployment may attach skills to it.
# ---------------------------------------------------------------------------

class ShareBody(BaseModel):
    owner_org: Optional[str] = None  # tenant identifier; defaults to "default"


@router.post(
    "/connections/{connection_id}/share",
    summary="Admin: share a connection org-wide",
)
async def share_connection(
    connection_id: str,
    body: ShareBody = ShareBody(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not _is_admin(user):
        raise HTTPException(403, "Only admins can share connections org-wide")
    row = (await db.execute(
        select(Connection).where(Connection.id == connection_id)
    )).scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Connection not found")
    # Only the connection's owner (who is also an admin) can share — this
    # avoids one admin silently exposing another admin's personal connection.
    if row.owner_id != user.id:
        raise HTTPException(403, "Only the connection owner can share it")
    row.scope = "org"
    row.owner_org = (body.owner_org or row.owner_org or "default")[:100]
    await db.commit()
    await db.refresh(row)
    return _serialize_connection(row, user)


@router.post(
    "/connections/{connection_id}/unshare",
    summary="Admin: revoke org-wide sharing",
)
async def unshare_connection(
    connection_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not _is_admin(user):
        raise HTTPException(403, "Only admins can change sharing")
    row = (await db.execute(
        select(Connection).where(Connection.id == connection_id)
    )).scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Connection not found")
    if row.owner_id != user.id:
        raise HTTPException(403, "Only the connection owner can unshare it")
    row.scope = "private"
    await db.commit()
    await db.refresh(row)
    return _serialize_connection(row, user)


@router.get(
    "/connections/{connection_id}/audit",
    summary="Owner/admin: list audit events for a shared connection",
)
async def connection_audit(
    connection_id: str,
    limit: int = Query(100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    row = (await db.execute(
        select(Connection).where(Connection.id == connection_id)
    )).scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Connection not found")
    if row.owner_id != user.id and not _is_admin(user):
        raise HTTPException(403, "Not allowed to view audit log")
    events = (await db.execute(
        select(ConnectionAuditLog)
        .where(ConnectionAuditLog.connection_id == connection_id)
        .order_by(ConnectionAuditLog.created_at.desc())
        .limit(limit)
    )).scalars().all()
    # Resolve usernames in one round-trip for nicer display.
    user_ids = sorted({e.user_id for e in events if e.user_id})
    name_by_id: dict[str, str] = {}
    if user_ids:
        users = (await db.execute(
            select(User.id, User.username).where(User.id.in_(user_ids))
        )).all()
        name_by_id = {uid: uname for uid, uname in users}
    return {"events": [
        {
            "id": e.id,
            "user_id": e.user_id,
            "username": name_by_id.get(e.user_id) if e.user_id else None,
            "tool_name": e.tool_name,
            "success": bool(e.success),
            "error": e.error,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in events
    ]}


# ---------------------------------------------------------------------------
# OAuth 2.0 flow
# ---------------------------------------------------------------------------

def _oauth_redirect_uri(request: Request) -> str:
    """Build the OAuth callback URL.

    To prevent open-redirect / host-spoofing, production deployments MUST set
    `OAUTH_REDIRECT_BASE_URL` (the canonical https origin of the deployment).
    A comma-separated `OAUTH_ALLOWED_HOSTS` env var may additionally restrict
    which request hosts are acceptable when no explicit base is set
    (useful for multi-domain dev). Falls back to the request host only when
    neither is configured (dev convenience).
    """
    explicit = os.environ.get("OAUTH_REDIRECT_BASE_URL", "").strip().rstrip("/")
    if explicit:
        return f"{explicit}/api/connectors/oauth/callback"
    allowed = [
        h.strip().lower() for h in os.environ.get("OAUTH_ALLOWED_HOSTS", "").split(",")
        if h.strip()
    ]
    request_host = (request.url.hostname or "").lower()
    if allowed and request_host not in allowed:
        raise HTTPException(
            400,
            f"OAuth redirect host '{request_host}' is not in OAUTH_ALLOWED_HOSTS.",
        )
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/connectors/oauth/callback"


@router.get("/{connector_id}/oauth/start", summary="Begin OAuth2 authorization for a connector")
async def oauth_start(
    connector_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    connector = get_connector(connector_id)
    if not connector or connector.auth_type != ConnectorAuthType.OAUTH2:
        raise HTTPException(400, "Connector does not use OAuth2")
    if not connector.oauth_authorize_url_builder:
        raise HTTPException(500, "Connector OAuth not configured")
    state = await new_state(db, {"user_id": user.id, "connector_id": connector_id})
    redirect_uri = _oauth_redirect_uri(request)
    try:
        auth_url = connector.oauth_authorize_url_builder(state, redirect_uri)
    except ConnectorError as exc:
        raise HTTPException(503, str(exc))
    return {"authorize_url": auth_url}


@router.get("/oauth/callback", summary="OAuth2 redirect target — completes any connector's flow")
async def oauth_callback(
    request: Request,
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    if error:
        return RedirectResponse(f"/connections?oauth_error={error}", status_code=302)
    if not code or not state:
        raise HTTPException(400, "Missing code or state")
    payload = await consume_state(db, state)
    if not payload:
        raise HTTPException(400, "Invalid or expired state token")
    connector = get_connector(payload["connector_id"])
    if not connector or not connector.oauth_exchange:
        raise HTTPException(400, "Unknown OAuth connector")
    redirect_uri = _oauth_redirect_uri(request)
    try:
        token_payload = await connector.oauth_exchange(code, redirect_uri)
    except ConnectorError as exc:
        return RedirectResponse(f"/connections?oauth_error={exc}", status_code=302)
    label = ""
    if connector.health_check:
        try:
            _, label = await connector.health_check(token_payload)
        except ConnectorError:
            label = ""
    await _persist_connection(
        db, payload["user_id"], connector.id, token_payload,
        display_name=label, metadata={"granted_scopes": token_payload.get("scope")},
    )
    return RedirectResponse("/connections?oauth_ok=1", status_code=302)
