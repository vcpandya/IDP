"""Generic OAuth2 authorization-code helpers.

These are scaffolding for OAuth2 connectors (Google, Jira, Linear OAuth, etc.).
Each connector wires its own `client_id`/`client_secret`/`scopes` via env vars
and supplies the authorize/token URLs.

State storage backend
---------------------
Short-lived CSRF state tokens are persisted via a pluggable backend so that
multi-worker / multi-server deployments can complete callbacks regardless of
which worker began the flow:

* Default: PostgreSQL/SQLite via the ``OAuthState`` table. A periodic prune
  removes expired rows on each access.
* When ``REDIS_URL`` is set: Redis with a native TTL. No extra DB writes
  per login attempt and no prune needed — the server expires keys for us.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional
from urllib.parse import urlencode

import httpx
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.connectors.base import ConnectorAuthError, ConnectorError
from idpkit.connectors.http import DEFAULT_TIMEOUT
from idpkit.db.models import OAuthState, utcnow

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Pluggable state store
# ---------------------------------------------------------------------------


class OAuthStateStore(ABC):
    """Backend for short-lived OAuth CSRF state tokens."""

    @abstractmethod
    async def put(self, db: AsyncSession, token: str, payload: dict) -> None:
        ...

    @abstractmethod
    async def pop(self, db: AsyncSession, token: str) -> Optional[dict]:
        ...


class DBOAuthStateStore(OAuthStateStore):
    """Stores state in the application database (default backend)."""

    async def _prune_expired(self, db: AsyncSession) -> None:
        await db.execute(delete(OAuthState).where(OAuthState.expires_at < utcnow()))

    async def put(self, db: AsyncSession, token: str, payload: dict) -> None:
        await self._prune_expired(db)
        db.add(OAuthState(
            token=token,
            payload=payload,
            expires_at=utcnow() + STATE_TTL,
        ))
        await db.commit()

    async def pop(self, db: AsyncSession, token: str) -> Optional[dict]:
        # Atomic DELETE ... RETURNING so concurrent callbacks racing on the
        # same token can never both succeed (PostgreSQL + SQLite >= 3.35).
        await self._prune_expired(db)
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


class RedisOAuthStateStore(OAuthStateStore):
    """Stores state in Redis with a native TTL.

    Avoids a per-login Postgres write+commit and removes the need for a
    periodic prune — Redis expires keys for us. Atomic single-use semantics
    are provided by ``GETDEL`` (Redis >= 6.2).
    """

    KEY_PREFIX = "idpkit:oauth_state:"

    def __init__(self, client) -> None:
        self._client = client

    def _key(self, token: str) -> str:
        return f"{self.KEY_PREFIX}{token}"

    async def put(self, db: AsyncSession, token: str, payload: dict) -> None:
        # ``db`` is intentionally unused — kept for interface parity.
        await self._client.set(
            self._key(token),
            json.dumps(payload),
            ex=int(STATE_TTL.total_seconds()),
        )

    async def pop(self, db: AsyncSession, token: str) -> Optional[dict]:
        raw = await self._client.getdel(self._key(token))
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None


_state_store: Optional[OAuthStateStore] = None


def _build_state_store() -> OAuthStateStore:
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if redis_url:
        try:
            from redis import asyncio as redis_asyncio  # type: ignore
        except ImportError:
            logger.warning(
                "REDIS_URL is set but the 'redis' package is not installed; "
                "falling back to the database-backed OAuth state store."
            )
        else:
            try:
                client = redis_asyncio.from_url(redis_url, decode_responses=True)
            except Exception as exc:  # pragma: no cover - depends on env
                logger.warning(
                    "Failed to initialise Redis OAuth state store (%s); "
                    "falling back to the database-backed store.", exc,
                )
            else:
                logger.info("OAuth state store: Redis (%s)", redis_url.split("@")[-1])
                return RedisOAuthStateStore(client)
    logger.info("OAuth state store: database")
    return DBOAuthStateStore()


def get_state_store() -> OAuthStateStore:
    """Return the process-wide state store, building it on first use."""
    global _state_store
    if _state_store is None:
        _state_store = _build_state_store()
    return _state_store


def reset_state_store() -> None:
    """Drop the cached state store. Intended for tests."""
    global _state_store
    _state_store = None


async def new_state(db: AsyncSession, payload: dict) -> str:
    """Issue an opaque CSRF state token bound to a payload.

    Persists the token via the configured backend (DB by default, Redis when
    ``REDIS_URL`` is set) with a short TTL so any worker can consume it
    during the OAuth callback.
    """
    token = secrets.token_urlsafe(24)
    await get_state_store().put(db, token, payload)
    return token


async def consume_state(db: AsyncSession, token: str) -> Optional[dict]:
    """Atomically pop a state token; returns None if unknown / expired / used."""
    return await get_state_store().pop(db, token)


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
