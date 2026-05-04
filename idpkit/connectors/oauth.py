"""Generic OAuth2 authorization-code helpers.

These are scaffolding for OAuth2 connectors (Google, Jira, Linear OAuth, etc.).
Each connector wires its own `client_id`/`client_secret`/`scopes` via env vars
and supplies the authorize/token URLs.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional
from urllib.parse import urlencode

import httpx
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.connectors.base import ConnectorAuthError, ConnectorError
from idpkit.connectors.http import DEFAULT_TIMEOUT
from idpkit.db.models import OAuthState, utcnow

# OAuth state tokens are short-lived; users typically complete consent in <2 min.
STATE_TTL = timedelta(minutes=10)


@dataclass
class OAuth2Spec:
    authorize_url: str
    token_url: str
    scopes: list[str]
    client_id_env: str
    client_secret_env: str
    extra_authorize_params: Optional[dict[str, str]] = None
    audience: Optional[str] = None


async def _prune_expired_states(db: AsyncSession) -> None:
    """Best-effort sweep of stale state rows."""
    await db.execute(delete(OAuthState).where(OAuthState.expires_at < utcnow()))


async def new_state(db: AsyncSession, payload: dict) -> str:
    """Issue an opaque CSRF state token bound to a payload (e.g. user_id, connector_id).

    Persists the token to the database with a short TTL so any worker can
    consume it during the OAuth callback. Also prunes expired rows.
    """
    await _prune_expired_states(db)
    token = secrets.token_urlsafe(24)
    db.add(OAuthState(
        token=token,
        payload=payload,
        expires_at=utcnow() + STATE_TTL,
    ))
    await db.commit()
    return token


async def consume_state(db: AsyncSession, token: str) -> Optional[dict]:
    """Atomically pop a state token; returns None if unknown / expired / already used.

    Uses a single ``DELETE ... RETURNING`` statement so concurrent callbacks
    racing on the same token can never both succeed (supported on PostgreSQL
    and on SQLite >= 3.35, which covers our deployment targets).
    """
    await _prune_expired_states(db)
    result = await db.execute(
        delete(OAuthState)
        .where(OAuthState.token == token)
        .returning(OAuthState.payload, OAuthState.expires_at)
    )
    row = result.first()
    await db.commit()
    if row is None:
        return None
    payload, expires_at = row
    if expires_at < utcnow():
        return None
    return dict(payload) if isinstance(payload, dict) else payload


def build_authorize_url(spec: OAuth2Spec, state: str, redirect_uri: str) -> str:
    client_id = os.environ.get(spec.client_id_env)
    if not client_id:
        raise ConnectorError(
            f"OAuth not configured: missing env var {spec.client_id_env}. "
            f"Ask your administrator to set up this integration."
        )
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(spec.scopes),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    if spec.extra_authorize_params:
        params.update(spec.extra_authorize_params)
    return f"{spec.authorize_url}?{urlencode(params)}"


async def exchange_code(spec: OAuth2Spec, code: str, redirect_uri: str) -> dict:
    client_id = os.environ.get(spec.client_id_env)
    client_secret = os.environ.get(spec.client_secret_env)
    if not client_id or not client_secret:
        raise ConnectorError("OAuth not configured: missing client credentials")
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                spec.token_url, data=data,
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        raise ConnectorError(f"Token exchange network error: {exc}") from exc
    if resp.status_code >= 400:
        raise ConnectorAuthError(f"Token exchange failed: HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


async def refresh_token(spec: OAuth2Spec, refresh_tok: str) -> dict:
    client_id = os.environ.get(spec.client_id_env)
    client_secret = os.environ.get(spec.client_secret_env)
    if not client_id or not client_secret:
        raise ConnectorError("OAuth not configured: missing client credentials")
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                spec.token_url,
                data={
                    "refresh_token": refresh_tok,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "refresh_token",
                },
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        raise ConnectorError(f"Token refresh network error: {exc}") from exc
    if resp.status_code >= 400:
        raise ConnectorAuthError(
            f"Token refresh failed: HTTP {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json()
