"""Generic OAuth2 authorization-code helpers.

These are scaffolding for OAuth2 connectors (Google, Jira, Linear OAuth, etc.).
Each connector wires its own `client_id`/`client_secret`/`scopes` via env vars
and supplies the authorize/token URLs.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

import httpx

from idpkit.connectors.base import ConnectorAuthError, ConnectorError
from idpkit.connectors.http import DEFAULT_TIMEOUT


@dataclass
class OAuth2Spec:
    authorize_url: str
    token_url: str
    scopes: list[str]
    client_id_env: str
    client_secret_env: str
    extra_authorize_params: Optional[dict[str, str]] = None
    audience: Optional[str] = None


_STATE_STORE: dict[str, dict] = {}


def new_state(payload: dict) -> str:
    """Issue an opaque CSRF state token bound to a payload (e.g. user_id, connector_id)."""
    token = secrets.token_urlsafe(24)
    _STATE_STORE[token] = payload
    return token


def consume_state(token: str) -> Optional[dict]:
    """Pop a state token; returns None if unknown / already used."""
    return _STATE_STORE.pop(token, None)


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
