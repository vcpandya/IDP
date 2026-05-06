"""Background runner for bulk-send batches.

For each row in an EnvelopeBatch:
  1. Build role_assignments + merge_values from the row dict using the column map.
  2. Instantiate a draft envelope from the template.
  3. Optionally send the envelope (dispatching invitation emails).
  4. Update the EnvelopeBatchItem status + the batch rollup counters.

Concurrency is bounded by an asyncio.Semaphore so we don't hammer the email
provider. Failures are isolated per row — a bad row never blocks the rest.

Cancellation is honored at row boundaries: if a user clicks "Cancel" while the
batch is running, queued and not-yet-started rows are marked cancelled and any
in-flight rows finish in their natural state (best-effort).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from idpkit.api.deps import get_storage
from idpkit.db.session import async_session
from idpkit.esign.models import (
    EnvelopeBatch,
    EnvelopeBatchItem,
    EnvelopeStatus,
    EnvelopeTemplate,
    SignatureEnvelope,
    SignerStatus,
)
from idpkit.esign.templates_lib import instantiate_template

logger = logging.getLogger(__name__)

BULK_CONCURRENCY = max(1, int(os.getenv("ESIGN_BATCH_CONCURRENCY", "3")))


def build_assignments_and_merge(
    row: Dict[str, str],
    column_map: Dict[str, Dict[str, str]],
    merge_keys: List[str],
) -> tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Translate a raw row + column mapping into role_assignments + merge_values.

    column_map shape:
        {
          "roles": { "<role_key>": {"name_col": "...", "email_col": "..."} },
          "merge": { "<merge_key>": "<column_name>" }
        }
    """
    role_map = column_map.get("roles") or {}
    merge_map = column_map.get("merge") or {}

    assignments: Dict[str, Dict[str, str]] = {}
    for role_key, cols in role_map.items():
        name_col = cols.get("name_col") or ""
        email_col = cols.get("email_col") or ""
        assignments[role_key] = {
            "name": (row.get(name_col, "") or "").strip(),
            "email": (row.get(email_col, "") or "").strip(),
        }

    merge_values: Dict[str, str] = {}
    for mk in merge_keys:
        col = merge_map.get(mk)
        if col:
            merge_values[mk] = (row.get(col, "") or "").strip()
    return assignments, merge_values


async def _send_envelope_emails(
    *,
    env: SignatureEnvelope,
    sender_name: str,
    base_url: str,
    db,
) -> int:
    """Mark signers SENT, generate tokens, and dispatch invites. Returns invites sent."""
    from idpkit.esign.email import send_signing_invitation

    sorted_signers = sorted(env.signers, key=lambda s: s.order_index)
    if env.signing_order == "sequential" and sorted_signers:
        first_order = sorted_signers[0].order_index
        active_now = {s.id for s in sorted_signers if s.order_index == first_order}
    else:
        active_now = {s.id for s in env.signers}

    invites = 0
    for signer in env.signers:
        raw_token = secrets.token_urlsafe(32)
        signer.token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        if signer.id in active_now:
            signer.status = SignerStatus.SENT.value
            db.add(signer)
            try:
                await send_signing_invitation(
                    signer_name=signer.name,
                    signer_email=signer.email,
                    sender_name=sender_name,
                    envelope_title=env.title,
                    signing_url=f"{base_url}/sign/{raw_token}",
                    message=env.message,
                    expires_at=env.expires_at.strftime("%B %d, %Y") if env.expires_at else None,
                )
                invites += 1
            except Exception as e:
                logger.warning("bulk-send invitation email failed for %s: %s", signer.email, e)
        else:
            signer.status = SignerStatus.PENDING.value
            db.add(signer)
    return invites


async def _is_cancelled(batch_id: str) -> bool:
    """Cheap, isolated cancellation check."""
    async with async_session() as db:
        b = await db.get(EnvelopeBatch, batch_id)
        return bool(b and b.status == "cancelled")


async def _process_item(
    *,
    item_id: str,
    batch_id: str,
    template_id: str,
    owner_id: str,
    sender_name: str,
    column_map: Dict[str, Any],
    merge_keys: List[str],
    base_url: str,
    send_immediately: bool,
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        # Pre-flight cancellation check (cheap; avoids creating envelopes on cancelled batches)
        if await _is_cancelled(batch_id):
            async with async_session() as db:
                it = await db.get(EnvelopeBatchItem, item_id)
                if it and it.status == "pending":
                    it.status = "cancelled"
                    it.updated_at = datetime.now(timezone.utc)
                    await db.commit()
            return

        async with async_session() as db:
            try:
                item = await db.get(EnvelopeBatchItem, item_id)
                if not item or item.status != "pending":
                    return
                row = json.loads(item.raw_row_json)
                assignments, merge_values = build_assignments_and_merge(row, column_map, merge_keys)
                storage = get_storage()
                env = await instantiate_template(
                    template_id=template_id,
                    owner_id=owner_id,
                    role_assignments=assignments,
                    merge_values=merge_values,
                    db=db,
                    storage=storage,
                )

                invites = 0
                if send_immediately:
                    # Reload with eager-loaded signers/fields for sending
                    result = await db.execute(
                        select(SignatureEnvelope)
                        .options(
                            selectinload(SignatureEnvelope.signers),
                            selectinload(SignatureEnvelope.fields),
                        )
                        .where(SignatureEnvelope.id == env.id)
                    )
                    env = result.scalar_one()
                    env.status = EnvelopeStatus.SENT.value
                    invites = await _send_envelope_emails(
                        env=env, sender_name=sender_name, base_url=base_url, db=db
                    )

                item.envelope_id = env.id
                if send_immediately:
                    item.status = "sent" if invites > 0 else "created"
                else:
                    item.status = "created"   # draft mode — envelope stays DRAFT
                item.error = None
                item.updated_at = datetime.now(timezone.utc)
                await db.commit()
            except Exception as e:
                logger.exception("bulk-send item %s failed: %s", item_id, e)
                try:
                    async with async_session() as dbf:
                        it = await dbf.get(EnvelopeBatchItem, item_id)
                        if it:
                            it.status = "failed"
                            it.error = str(e)[:1000]
                            it.updated_at = datetime.now(timezone.utc)
                            await dbf.commit()
                except Exception:
                    pass


async def run_bulk_batch(
    *,
    batch_id: str,
    base_url: str,
    sender_name: str,
) -> None:
    """Top-level entry point — schedule all items with bounded concurrency.

    Uses a CAS-style transition: only flips ``pending`` → ``running``. If the
    batch was already cancelled (or is running on a different worker), we exit
    without taking any action.
    """
    # CAS: claim the batch only if still pending. Prevents cancel/start races and
    # double-execution if startup recovery fires concurrently.
    async with async_session() as db:
        cas = await db.execute(
            update(EnvelopeBatch)
            .where(EnvelopeBatch.id == batch_id, EnvelopeBatch.status == "pending")
            .values(status="running", started_at=datetime.now(timezone.utc))
        )
        await db.commit()
        if cas.rowcount == 0:
            logger.info("bulk batch %s skipped (not in pending state)", batch_id)
            return

        batch = await db.get(EnvelopeBatch, batch_id)
        if not batch:
            return
        template_id = batch.template_id
        owner_id = batch.owner_id
        send_immediately = bool(batch.send_immediately)
        try:
            column_map = json.loads(batch.column_map_json or "{}")
        except Exception:
            column_map = {}
        if not template_id:
            batch.status = "failed"
            batch.finished_at = datetime.now(timezone.utc)
            await db.commit()
            return

        tpl = await db.get(EnvelopeTemplate, template_id)
        merge_keys: List[str] = []
        if tpl:
            from idpkit.esign.merge import extract_merge_keys
            from idpkit.esign.templates_lib import merge_fields_schema_load
            schema = merge_fields_schema_load(tpl.merge_fields_json)
            merge_keys = [m["key"] for m in schema]
            for k in extract_merge_keys(tpl.title, tpl.message):
                if k.lower() not in {m.lower() for m in merge_keys}:
                    merge_keys.append(k)

        items_q = await db.execute(
            select(EnvelopeBatchItem.id)
            .where(EnvelopeBatchItem.batch_id == batch_id, EnvelopeBatchItem.status == "pending")
            .order_by(EnvelopeBatchItem.row_index)
        )
        item_ids = [row[0] for row in items_q.all()]

    sem = asyncio.Semaphore(BULK_CONCURRENCY)
    tasks = [
        asyncio.create_task(_process_item(
            item_id=iid,
            batch_id=batch_id,
            template_id=template_id,
            owner_id=owner_id,
            sender_name=sender_name,
            column_map=column_map,
            merge_keys=merge_keys,
            base_url=base_url,
            send_immediately=send_immediately,
            sem=sem,
        ))
        for iid in item_ids
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Finalize: recount from items for accuracy. Set finished_at unconditionally
    # so the delete endpoint knows in-flight tasks have terminated.
    async with async_session() as db:
        batch = await db.get(EnvelopeBatch, batch_id)
        if not batch:
            return
        items_q = await db.execute(
            select(EnvelopeBatchItem).where(EnvelopeBatchItem.batch_id == batch_id)
        )
        items = items_q.scalars().all()
        batch.created_count = sum(1 for i in items if i.status in ("created", "sent", "completed"))
        batch.sent_count = sum(1 for i in items if i.status in ("sent", "completed"))
        batch.failed_count = sum(1 for i in items if i.status == "failed")
        if batch.status == "cancelled":
            # Mark any still-pending items as cancelled (defensive — they should already be)
            for i in items:
                if i.status == "pending":
                    i.status = "cancelled"
        else:
            if batch.failed_count == batch.total_rows and batch.total_rows > 0:
                batch.status = "failed"
            else:
                batch.status = "completed"
        batch.finished_at = datetime.now(timezone.utc)
        await db.commit()


async def recover_pending_batches() -> List[str]:
    """Resume any batches stranded in pending/running on process startup.

    Resets ``running`` batches back to ``pending`` (so the CAS in ``run_bulk_batch``
    can re-claim them) and schedules a runner for each. Returns the list of
    batch ids resumed.
    """
    base_url = os.getenv("DEPLOYED_DOMAIN") or ""
    if base_url and not base_url.startswith("http"):
        base_url = "https://" + base_url
    base_url = base_url.rstrip("/")

    resumed: List[str] = []
    async with async_session() as db:
        # Flip running -> pending so CAS in run_bulk_batch will re-claim them.
        await db.execute(
            update(EnvelopeBatch)
            .where(EnvelopeBatch.status == "running")
            .values(status="pending", started_at=None)
        )
        await db.commit()

        result = await db.execute(
            select(EnvelopeBatch).where(EnvelopeBatch.status == "pending")
        )
        batches = result.scalars().all()
        from idpkit.db.models import User
        for b in batches:
            user = await db.get(User, b.owner_id)
            sender_name = user.username if user else "IDP Kit"
            asyncio.create_task(run_bulk_batch(
                batch_id=b.id, base_url=base_url, sender_name=sender_name,
            ))
            resumed.append(b.id)
    if resumed:
        logger.info("Resumed %d e-sign bulk batch(es) on startup", len(resumed))
    return resumed
