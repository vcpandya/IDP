"""E-sign Bulk Send (Batch Signing) — instantiate one envelope per recipient row."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from idpkit.api.deps import get_current_user
from idpkit.db.models import User
from idpkit.db.session import get_db
from idpkit.esign.bulk_runner import run_bulk_batch
from idpkit.esign.merge import extract_merge_keys
from idpkit.esign.models import (
    EnvelopeBatch,
    EnvelopeBatchItem,
    EnvelopeTemplate,
    SignatureEnvelope,
)
from idpkit.esign.recipient_parsers import MAX_ROWS, parse_recipients
from idpkit.esign.templates_lib import merge_fields_schema_load

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/esign/batches", tags=["esign-batches"])


# ----- Preview parsing -----------------------------------------------------

@router.post("/preview", summary="Parse a CSV/XLSX upload or pasted table without creating a batch")
async def preview_recipients(
    file: Optional[UploadFile] = File(None),
    paste: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
):
    MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
    content = None
    if file and file.filename:
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)")
    fname = file.filename if file else None
    if paste and len(paste) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Pasted text too large")
    try:
        headers, rows, source_label = parse_recipients(fname, content, paste)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse recipients: {e}")
    if not headers:
        raise HTTPException(status_code=400, detail="No header row detected")
    if not rows:
        raise HTTPException(status_code=400, detail="No data rows detected")
    return {
        "headers": headers,
        "rows": rows[:50],            # preview first 50 only
        "row_count": len(rows),
        "truncated_to": MAX_ROWS if len(rows) >= MAX_ROWS else None,
        "source_label": source_label,
        "all_rows": rows,             # full list returned for client to round-trip on submit
    }


# ----- Create batch --------------------------------------------------------

class BatchCreate(BaseModel):
    template_id: str
    name: str = Field(..., min_length=1, max_length=200)
    source_label: Optional[str] = Field(None, max_length=200)
    rows: List[Dict[str, str]]
    column_map: Dict[str, Any]   # {"roles": {role_key: {name_col, email_col}}, "merge": {merge_key: column}}
    send_immediately: bool = True


@router.post("", status_code=201, summary="Create + start a bulk-send batch")
async def create_batch(
    payload: BatchCreate,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="At least one row is required")
    if len(payload.rows) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"Too many rows (max {MAX_ROWS})")

    # Verify template ownership
    tpl = await db.get(EnvelopeTemplate, payload.template_id)
    if not tpl or tpl.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Template not found")

    # Validate column_map references real role keys
    role_q = await db.execute(
        select(EnvelopeBatch).where(EnvelopeBatch.id == "__noop__")  # noop just to use db
    )
    _ = role_q.scalar_one_or_none()
    role_keys_q = await db.execute(
        select(EnvelopeTemplate)
        .options(selectinload(EnvelopeTemplate.roles))
        .where(EnvelopeTemplate.id == payload.template_id)
    )
    tpl_full = role_keys_q.scalar_one()
    role_keys = {r.role_key for r in tpl_full.roles}
    mapped_roles = (payload.column_map.get("roles") or {})
    missing_roles = role_keys - set(mapped_roles.keys())
    if missing_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Column mapping missing role(s): {', '.join(sorted(missing_roles))}",
        )
    for rk, cols in mapped_roles.items():
        if rk not in role_keys:
            continue
        if not (cols.get("name_col") and cols.get("email_col")):
            raise HTTPException(
                status_code=400,
                detail=f"Role '{rk}' needs both a name column and an email column",
            )

    batch = EnvelopeBatch(
        owner_id=user.id,
        template_id=payload.template_id,
        name=payload.name.strip(),
        source_label=payload.source_label,
        status="pending",
        column_map_json=json.dumps(payload.column_map),
        send_immediately=payload.send_immediately,
        total_rows=len(payload.rows),
    )
    db.add(batch)
    await db.flush()

    for idx, row in enumerate(payload.rows):
        db.add(EnvelopeBatchItem(
            batch_id=batch.id,
            row_index=idx,
            raw_row_json=json.dumps(dict(row)),
            status="pending",
        ))
    await db.commit()

    base_url = str(request.base_url).rstrip("/")
    sender_name = user.username
    # Schedule the runner as a background task; it owns its own DB sessions.
    asyncio.create_task(run_bulk_batch(
        batch_id=batch.id,
        base_url=base_url,
        sender_name=sender_name,
    ))
    return {"id": batch.id, "status": batch.status, "total_rows": batch.total_rows}


# ----- Status / listing ----------------------------------------------------

def _batch_summary(b: EnvelopeBatch) -> dict:
    return {
        "id": b.id,
        "name": b.name,
        "template_id": b.template_id,
        "source_label": b.source_label,
        "status": b.status,
        "total_rows": b.total_rows,
        "created_count": b.created_count,
        "sent_count": b.sent_count,
        "completed_count": b.completed_count,
        "failed_count": b.failed_count,
        "created_at": b.created_at.isoformat() if b.created_at else None,
        "started_at": b.started_at.isoformat() if b.started_at else None,
        "finished_at": b.finished_at.isoformat() if b.finished_at else None,
    }


@router.get("", summary="List batches")
async def list_batches(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EnvelopeBatch)
        .where(EnvelopeBatch.owner_id == user.id)
        .order_by(EnvelopeBatch.created_at.desc())
    )
    return [_batch_summary(b) for b in result.scalars().all()]


@router.get("/{batch_id}", summary="Batch detail with per-row status")
async def get_batch(
    batch_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EnvelopeBatch)
        .options(selectinload(EnvelopeBatch.items))
        .where(EnvelopeBatch.id == batch_id, EnvelopeBatch.owner_id == user.id)
    )
    b = result.scalar_one_or_none()
    if not b:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Also pull current envelope statuses for any items that have envelopes
    env_ids = [i.envelope_id for i in b.items if i.envelope_id]
    env_status_map: Dict[str, str] = {}
    if env_ids:
        env_q = await db.execute(
            select(SignatureEnvelope.id, SignatureEnvelope.status).where(
                SignatureEnvelope.id.in_(env_ids)
            )
        )
        env_status_map = {row.id: row.status for row in env_q.all()}

    summary = _batch_summary(b)
    summary["items"] = [
        {
            "id": i.id,
            "row_index": i.row_index,
            "envelope_id": i.envelope_id,
            "envelope_status": env_status_map.get(i.envelope_id) if i.envelope_id else None,
            "status": i.status,
            "error": i.error,
            "raw_row": json.loads(i.raw_row_json),
            "updated_at": i.updated_at.isoformat() if i.updated_at else None,
        }
        for i in sorted(b.items, key=lambda x: x.row_index)
    ]
    return summary


@router.post("/{batch_id}/cancel", summary="Cancel a running batch (best-effort)")
async def cancel_batch(
    batch_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    b = await db.get(EnvelopeBatch, batch_id)
    if not b or b.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Batch not found")
    if b.status in ("completed", "failed", "cancelled"):
        return {"status": b.status, "detail": "Batch already finished"}
    b.status = "cancelled"
    await db.commit()
    return {"status": b.status}


@router.delete("/{batch_id}", status_code=204, summary="Delete a batch and its item rows")
async def delete_batch(
    batch_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    b = await db.get(EnvelopeBatch, batch_id)
    if not b or b.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Batch not found")
    if b.status == "running":
        raise HTTPException(status_code=400, detail="Cancel the batch before deleting")
    # Block delete on cancelled-but-not-yet-finalized batches: in-flight tasks may
    # still be writing envelopes. The runner sets finished_at when it terminates.
    if b.status == "cancelled" and b.finished_at is None:
        raise HTTPException(
            status_code=409,
            detail="Batch is cancelling — wait until in-flight items finish, then retry.",
        )
    await db.delete(b)
    await db.commit()
    return None


@router.get("/{batch_id}/results.csv", summary="Download per-row results as CSV")
async def download_results(
    batch_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EnvelopeBatch)
        .options(selectinload(EnvelopeBatch.items))
        .where(EnvelopeBatch.id == batch_id, EnvelopeBatch.owner_id == user.id)
    )
    b = result.scalar_one_or_none()
    if not b:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Collect all source columns from row 0
    source_cols: List[str] = []
    if b.items:
        first = json.loads(b.items[0].raw_row_json)
        source_cols = list(first.keys())

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["row_index", *source_cols, "envelope_id", "status", "error"])
    for it in sorted(b.items, key=lambda x: x.row_index):
        row = json.loads(it.raw_row_json)
        writer.writerow([
            it.row_index,
            *(row.get(c, "") for c in source_cols),
            it.envelope_id or "",
            it.status,
            it.error or "",
        ])
    out.seek(0)
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="batch-{batch_id[:8]}-results.csv"'},
    )
