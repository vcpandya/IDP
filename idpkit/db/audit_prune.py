"""Background pruning for the shared-connection audit log.

Each call through a shared (org-scoped) connection writes a row to
``connection_audit_log``. On a busy deployment that table grows without bound
and slows down both inserts and the audit modal query.

This module provides a tiny scheduler that periodically deletes rows older
than ``CONNECTION_AUDIT_RETENTION_DAYS`` (default: 90 days). The retention
window is configurable via env var so operators can tune it per deployment.

Design notes
------------
* No external scheduler/dependency — we just spawn an asyncio task in the
  FastAPI lifespan. One worker is enough; the DELETE is idempotent so even
  if multiple Gunicorn workers each fire it, the second one is a no-op.
* The first prune runs on startup so a freshly-deployed server immediately
  trims any backlog inherited from before the feature shipped.
* Errors are logged and swallowed — we never want a transient DB hiccup to
  kill the background task and let the table grow forever again.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete

from .models import ConnectionAuditLog

_log = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 90
DEFAULT_INTERVAL_SECONDS = 24 * 60 * 60  # once per day


def _retention_days() -> int:
    raw = os.getenv("CONNECTION_AUDIT_RETENTION_DAYS")
    if not raw:
        return DEFAULT_RETENTION_DAYS
    try:
        val = int(raw)
    except ValueError:
        _log.warning(
            "Invalid CONNECTION_AUDIT_RETENTION_DAYS=%r; falling back to %d",
            raw, DEFAULT_RETENTION_DAYS,
        )
        return DEFAULT_RETENTION_DAYS
    if val <= 0:
        _log.warning(
            "CONNECTION_AUDIT_RETENTION_DAYS must be positive; got %d, "
            "falling back to %d", val, DEFAULT_RETENTION_DAYS,
        )
        return DEFAULT_RETENTION_DAYS
    return val


async def prune_connection_audit_log(session_factory, *, retention_days: int | None = None) -> int:
    """Delete audit rows older than the retention window.

    Returns the number of rows deleted (best-effort — some drivers don't
    report rowcount accurately, in which case 0 is returned).
    """
    days = retention_days if retention_days is not None else _retention_days()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    async with session_factory() as db:
        try:
            result = await db.execute(
                delete(ConnectionAuditLog).where(
                    ConnectionAuditLog.created_at < cutoff
                )
            )
            await db.commit()
        except Exception as exc:
            await db.rollback()
            _log.warning("Audit-log prune failed: %s", exc)
            return 0
    deleted = result.rowcount or 0
    if deleted < 0:
        deleted = 0
    if deleted:
        _log.info(
            "Pruned %d connection_audit_log row(s) older than %d days",
            deleted, days,
        )
    return deleted


async def _scheduler_loop(session_factory, interval_seconds: int) -> None:
    while True:
        try:
            await prune_connection_audit_log(session_factory)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover — defensive
            _log.exception("Unexpected error in audit prune scheduler: %s", exc)
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise


def start_audit_prune_scheduler(
    session_factory,
    *,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> asyncio.Task:
    """Spawn the periodic prune task and return its asyncio.Task handle."""
    return asyncio.create_task(
        _scheduler_loop(session_factory, interval_seconds),
        name="connection-audit-prune",
    )
