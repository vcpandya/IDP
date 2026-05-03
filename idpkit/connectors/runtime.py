"""Runtime helpers — load active connections, decrypt creds just-in-time,
build LLM tool definitions, dispatch connector tool calls.

Architectural rule: credentials are read fresh from DB and decrypted on
*every* tool call. They are never cached at process scope.
"""
from __future__ import annotations

import logging
from datetime import timezone
from typing import Awaitable, Callable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.connectors.base import (
    Connector, ConnectorAuthError, ConnectorAuthType, ConnectorError, ConnectorTool,
)
from idpkit.connectors.crypto import decrypt_credentials, encrypt_credentials
from idpkit.connectors.registry import REGISTRY, get_connector, tool_to_connector_map
from idpkit.db.models import Connection, utcnow

logger = logging.getLogger(__name__)


async def list_active_connections(db: AsyncSession, user_id: str) -> list[Connection]:
    rows = (await db.execute(
        select(Connection).where(
            Connection.owner_id == user_id,
            Connection.status == "active",
        ).order_by(Connection.connector_id)
    )).scalars().all()
    return list(rows)


async def get_active_connection(
    db: AsyncSession, user_id: str, connector_id: str,
) -> Optional[Connection]:
    return (await db.execute(
        select(Connection).where(
            Connection.owner_id == user_id,
            Connection.connector_id == connector_id,
            Connection.status == "active",
        ).order_by(Connection.created_at.desc())
    )).scalars().first()


def build_runtime_tools(active_connections: list[Connection]) -> list[dict]:
    """Build OpenAI-format tool definitions for the LLM, one per active connection's tools."""
    seen_connector_ids = {c.connector_id for c in active_connections}
    out: list[dict] = []
    for cid in seen_connector_ids:
        conn = get_connector(cid)
        if not conn:
            continue
        for tool in conn.tools:
            out.append(tool.to_openai_function())
    return out


def build_runtime_executors(
    db: AsyncSession,
    user_id: str,
) -> dict[str, Callable[..., Awaitable[dict]]]:
    """Return a {tool_name: executor(args, llm, db)} dict ready to merge with the agent dispatcher.

    Each executor:
    - Looks up the user's active connection for the tool's connector.
    - Decrypts the credentials (just-in-time).
    - Calls the connector tool's executor.
    - On ConnectorAuthError, marks the connection disconnected and returns a structured error.
    """
    tool_to_conn = tool_to_connector_map()

    def _make_executor(tool: ConnectorTool, connector_id: str):
        async def _runner(args: dict, llm, _db: AsyncSession) -> dict:
            sanitized = {k: v for k, v in (args or {}).items() if not k.startswith("_")}
            connection = await get_active_connection(_db, user_id, connector_id)
            if connection is None:
                return {
                    "error": (
                        f"You are not connected to {connector_id}. "
                        f"Visit /connections to connect this integration."
                    )
                }
            try:
                creds = decrypt_credentials(connection.encrypted_credentials)
            except ValueError as exc:
                logger.warning("Could not decrypt connection %s: %s", connection.id, exc)
                await _mark_disconnected(_db, connection, str(exc))
                return {"error": "Stored credentials are unreadable; reconnect this integration."}

            try:
                result = await tool.executor(sanitized, creds)
            except ConnectorAuthError as exc:
                # OAuth2 connectors: attempt one-shot refresh before giving up.
                connector_def = get_connector(connector_id)
                can_refresh = bool(
                    connector_def
                    and connector_def.auth_type == ConnectorAuthType.OAUTH2
                    and connector_def.oauth_refresh
                    and creds.get("refresh_token")
                )
                refreshed = await _try_oauth_refresh(
                    _db, connection, connector_def, creds, exc,
                ) if can_refresh else None
                if refreshed is not None:
                    try:
                        result = await tool.executor(sanitized, refreshed)
                    except ConnectorAuthError as exc2:
                        logger.info("Connector %s auth failure after refresh: %s", connector_id, exc2)
                        await _mark_disconnected(_db, connection, str(exc2))
                        return {
                            "error": (
                                f"Auth failed for {connector_id} after token refresh; "
                                f"reconnect at /connections."
                            )
                        }
                    except ConnectorError as exc2:
                        return {"error": str(exc2)}
                    return result
                # If a refresh was attempted, _try_oauth_refresh has already
                # marked the connection disconnected with the refresh-failure
                # reason — don't overwrite it with the original auth error.
                if not can_refresh:
                    logger.info("Connector %s auth failure: %s", connector_id, exc)
                    await _mark_disconnected(_db, connection, str(exc))
                return {"error": f"Auth failed for {connector_id}: reconnect at /connections"}
            except ConnectorError as exc:
                logger.info("Connector %s error: %s", connector_id, exc)
                return {"error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected connector error for %s", connector_id)
                return {"error": f"Connector failure: {type(exc).__name__}"}
            return result

        return _runner

    out: dict[str, Callable[..., Awaitable[dict]]] = {}
    for tool_name, connector_id in tool_to_conn.items():
        conn = get_connector(connector_id)
        if not conn:
            continue
        for tool in conn.tools:
            if tool.name == tool_name:
                out[tool_name] = _make_executor(tool, connector_id)
                break
    return out


async def _mark_disconnected(
    db: AsyncSession, connection: Connection, error: str,
) -> None:
    connection.status = "disconnected"
    connection.last_error = error[:1000]
    connection.last_checked_at = utcnow()
    try:
        await db.commit()
    except Exception:  # noqa: BLE001
        await db.rollback()


async def _try_oauth_refresh(
    db: AsyncSession,
    connection: Connection,
    connector_def: Optional[Connector],
    current_creds: dict,
    original_error: Exception,
) -> Optional[dict]:
    """If the connector supports OAuth2 refresh and we have a refresh token,
    attempt a single refresh, persist the new token (encrypted), and return
    the merged credentials. On failure, mark the connection disconnected
    and return None.
    """
    if (
        not connector_def
        or connector_def.auth_type != ConnectorAuthType.OAUTH2
        or not connector_def.oauth_refresh
        or not current_creds.get("refresh_token")
    ):
        return None
    try:
        new_token = await connector_def.oauth_refresh(current_creds)
    except (ConnectorAuthError, ConnectorError) as exc:
        logger.info("OAuth refresh failed for %s: %s", connector_def.id, exc)
        await _mark_disconnected(db, connection, f"refresh failed: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error refreshing OAuth for %s", connector_def.id)
        await _mark_disconnected(db, connection, f"refresh error: {exc}")
        return None
    # Refresh responses commonly omit `refresh_token` — preserve the old one.
    merged = {**current_creds, **(new_token or {})}
    if "refresh_token" not in (new_token or {}):
        merged["refresh_token"] = current_creds["refresh_token"]
    try:
        connection.encrypted_credentials = encrypt_credentials(merged)
        connection.last_checked_at = utcnow()
        connection.last_error = None
        await db.commit()
    except Exception:  # noqa: BLE001
        await db.rollback()
    return merged


def build_capability_prompt_section(
    active_connections: list[Connection],
    active_skills: Optional[list[dict]] = None,
) -> str:
    """Append-to-system-prompt: lists the user's connected integrations
    and explicitly calls out which connectors are *required by an installed
    skill* but not connected. Other registry connectors are not mentioned —
    the goal is to keep the prompt focused on what the user actually wants.
    """
    if not REGISTRY:
        return ""
    active_ids = {c.connector_id for c in active_connections}
    available = [REGISTRY[cid] for cid in sorted(active_ids) if cid in REGISTRY]

    # Compute connectors required by the user's installed skills.
    required_by_skill: dict[str, list[str]] = {}
    for s in active_skills or []:
        req = (s.get("requirements") or {}).get("connectors") or []
        for cid in req:
            required_by_skill.setdefault(cid, []).append(s.get("name", "?"))
    missing_required = [
        cid for cid in sorted(required_by_skill.keys())
        if cid in REGISTRY and cid not in active_ids
    ]

    lines = ["\n\n### Connector Availability"]
    if available:
        lines.append("**Connected** (you may call these tools):")
        for c in available:
            tool_list = ", ".join(t.name for t in c.tools)
            lines.append(f"- {c.display_name} → {tool_list}")
    else:
        lines.append("**Connected:** _(none — no external integrations connected)_")

    if missing_required:
        lines.append(
            "\n**Required by installed skills but NOT connected — do NOT call their tools; "
            "tell the user to connect at /connections first:**"
        )
        for cid in missing_required:
            disp = REGISTRY[cid].display_name
            users = ", ".join(sorted(set(required_by_skill[cid])))
            lines.append(f"- {disp} (needed by: {users})")

    lines.append(
        "\nNever fabricate data from unavailable services. If a task requires a "
        "disconnected integration, tell the user which one to connect."
    )
    return "\n".join(lines)
