"""Background task that flips envelopes past their expires_at to EXPIRED.

Runs once on startup and then every ESIGN_EXPIRY_SWEEP_INTERVAL seconds
(default: 1h). Uses a PostgreSQL advisory lock so only one Gunicorn worker
performs the sweep at a time. Falls back to best-effort on non-PG (SQLite tests).
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

from sqlalchemy import text, update

from idpkit.esign.models import EnvelopeStatus, SignatureEnvelope

logger = logging.getLogger(__name__)

_ADVISORY_LOCK_KEY = 0x65_53_69_67_6E_45_78  # "eSignEx"
_DEFAULT_INTERVAL_SEC = 60 * 60  # 1 hour

# Statuses that should still be eligible to flip to EXPIRED.
_ACTIVE_STATUSES = (
    EnvelopeStatus.SENT.value,
    EnvelopeStatus.VIEWED.value,
    EnvelopeStatus.PARTIALLY_SIGNED.value,
)


async def _sweep_once(session_factory) -> int:
    """Return the number of envelopes flipped to EXPIRED."""
    now = datetime.now(timezone.utc)
    async with session_factory() as db:
        # Try advisory lock (no-op on SQLite — try/except swallows the error)
        got_lock = True
        try:
            res = await db.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": _ADVISORY_LOCK_KEY})
            got_lock = bool(res.scalar())
        except Exception:
            got_lock = True  # non-PG; just proceed

        if not got_lock:
            return 0

        try:
            result = await db.execute(
                update(SignatureEnvelope)
                .where(
                    SignatureEnvelope.status.in_(_ACTIVE_STATUSES),
                    SignatureEnvelope.expires_at.isnot(None),
                    SignatureEnvelope.expires_at < now,
                )
                .values(status=EnvelopeStatus.EXPIRED.value)
                .execution_options(synchronize_session=False)
            )
            count = result.rowcount or 0
            await db.commit()
            return count
        finally:
            try:
                await db.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _ADVISORY_LOCK_KEY})
                await db.commit()
            except Exception:
                pass


async def _scheduler_loop(session_factory, interval_sec: int) -> None:
    while True:
        try:
            n = await _sweep_once(session_factory)
            if n:
                logger.info("E-sign expiry sweep: marked %d envelope(s) as EXPIRED", n)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("E-sign expiry sweep failed: %s", exc)
        await asyncio.sleep(interval_sec)


def start_expiry_sweep_scheduler(session_factory) -> asyncio.Task:
    """Kick off the periodic sweep. Returns the asyncio.Task for cancellation."""
    interval = int(os.getenv("ESIGN_EXPIRY_SWEEP_INTERVAL", str(_DEFAULT_INTERVAL_SEC)))
    return asyncio.create_task(_scheduler_loop(session_factory, interval))
