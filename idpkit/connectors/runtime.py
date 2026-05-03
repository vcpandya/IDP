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
    Connector, ConnectorAuthError, ConnectorError, ConnectorTool,
)
from idpkit.connectors.crypto import decrypt_credentials
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


def build_capability_prompt_section(
    active_connections: list[Connection],
) -> str:
    """Append-to-system-prompt: tells the LLM which connectors are available vs unavailable."""
    if not REGISTRY:
        return ""
    active_ids = {c.connector_id for c in active_connections}
    available = [REGISTRY[c] for c in sorted(active_ids) if c in REGISTRY]
    unavailable = [c for cid, c in sorted(REGISTRY.items()) if cid not in active_ids]

    lines = ["\n\n### Connector Availability"]
    if available:
        lines.append("**Connected** (you may call these tools):")
        for c in available:
            tool_list = ", ".join(t.name for t in c.tools)
            lines.append(f"- {c.display_name} → {tool_list}")
    else:
        lines.append("**Connected:** _(none — no external integrations connected)_")
    if unavailable:
        names = ", ".join(c.display_name for c in unavailable)
        lines.append(
            f"\n**Not connected** (do NOT call these — tell the user to connect at /connections first): {names}"
        )
    lines.append(
        "\nNever fabricate data from unavailable services. If a task requires a "
        "disconnected integration, tell the user which one to connect."
    )
    return "\n".join(lines)
