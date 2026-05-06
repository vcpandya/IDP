"""E-sign template management — DocuSign-style reusable envelope blueprints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from idpkit.api.deps import get_current_user, get_storage
from idpkit.core.storage import StorageBackend
from idpkit.db.models import User
from idpkit.db.session import get_db
from idpkit.esign.models import (
    EnvelopeTemplate,
    EnvelopeTemplateField,
    EnvelopeTemplateRole,
)
from idpkit.esign.templates_lib import (
    instantiate_template,
    merge_fields_schema_dump,
    merge_fields_schema_load,
    snapshot_envelope_to_template,
    template_response,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/esign/templates", tags=["esign-templates"])


# ----- Schemas -------------------------------------------------------------

class MergeFieldSpec(BaseModel):
    key: str
    label: str = ""
    type: str = "text"
    required: bool = False


class RoleUpdate(BaseModel):
    role_key: str
    role_label: str
    order_index: int = 0
    default_name: Optional[str] = None
    default_email: Optional[str] = None


class TemplateCreateFromEnvelope(BaseModel):
    envelope_id: str
    name: str = Field(..., min_length=1, max_length=200)
    merge_fields: List[MergeFieldSpec] = Field(default_factory=list)


class TemplateUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    title: Optional[str] = Field(None, max_length=500)
    message: Optional[str] = None
    signing_order: Optional[str] = None
    expiry_days: Optional[int] = Field(None, ge=1, le=365)
    merge_fields: Optional[List[MergeFieldSpec]] = None
    roles: Optional[List[RoleUpdate]] = None


class TemplateInstantiate(BaseModel):
    role_assignments: Dict[str, Dict[str, str]]  # role_key -> {name, email}
    merge_values: Dict[str, Any] = Field(default_factory=dict)
    title_override: Optional[str] = Field(None, max_length=500)
    message_override: Optional[str] = None
    expiry_days: Optional[int] = Field(None, ge=1, le=365)
    send_immediately: bool = False


# ----- Routes --------------------------------------------------------------

@router.post("", status_code=201, summary="Create template by snapshotting an existing envelope")
async def create_template_from_envelope(
    payload: TemplateCreateFromEnvelope,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    try:
        tpl = await snapshot_envelope_to_template(
            envelope_id=payload.envelope_id,
            owner_id=user.id,
            template_name=payload.name,
            db=db,
            storage=storage,
            merge_fields=[m.model_dump() for m in payload.merge_fields],
        )
        await db.commit()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = await db.execute(
        select(EnvelopeTemplate)
        .options(selectinload(EnvelopeTemplate.roles), selectinload(EnvelopeTemplate.fields))
        .where(EnvelopeTemplate.id == tpl.id)
    )
    return template_response(result.scalar_one())


@router.get("", summary="List the caller's templates")
async def list_templates(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EnvelopeTemplate)
        .options(selectinload(EnvelopeTemplate.roles), selectinload(EnvelopeTemplate.fields))
        .where(EnvelopeTemplate.owner_id == user.id)
        .order_by(EnvelopeTemplate.updated_at.desc())
    )
    return [template_response(t) for t in result.scalars().all()]


@router.get("/{template_id}", summary="Template detail")
async def get_template(
    template_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EnvelopeTemplate)
        .options(selectinload(EnvelopeTemplate.roles), selectinload(EnvelopeTemplate.fields))
        .where(EnvelopeTemplate.id == template_id, EnvelopeTemplate.owner_id == user.id)
    )
    tpl = result.scalar_one_or_none()
    if not tpl:
        raise HTTPException(status_code=404, detail="Template not found")
    return template_response(tpl)


@router.put("/{template_id}", summary="Update template header / roles / merge fields")
async def update_template(
    template_id: str,
    payload: TemplateUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EnvelopeTemplate)
        .options(selectinload(EnvelopeTemplate.roles), selectinload(EnvelopeTemplate.fields))
        .where(EnvelopeTemplate.id == template_id, EnvelopeTemplate.owner_id == user.id)
    )
    tpl = result.scalar_one_or_none()
    if not tpl:
        raise HTTPException(status_code=404, detail="Template not found")

    if payload.name is not None:
        tpl.name = payload.name.strip() or tpl.name
    if payload.title is not None:
        tpl.title = payload.title.strip() or tpl.title
    if payload.message is not None:
        tpl.message = payload.message
    if payload.signing_order in ("parallel", "sequential"):
        tpl.signing_order = payload.signing_order
    if payload.expiry_days is not None:
        tpl.expiry_days = int(payload.expiry_days)
    if payload.merge_fields is not None:
        tpl.merge_fields_json = merge_fields_schema_dump([m.model_dump() for m in payload.merge_fields])

    if payload.roles is not None:
        # Reject duplicate role keys before mutating anything (DB unique constraint
        # would also catch this, but a friendly error is better than a 500).
        seen_keys: set[str] = set()
        for r_in in payload.roles:
            k = (r_in.role_key or "").strip()
            if not k:
                raise HTTPException(status_code=400, detail="Role key cannot be empty")
            if k in seen_keys:
                raise HTTPException(status_code=400, detail=f"Duplicate role key: {k}")
            seen_keys.add(k)
        # Re-key fields if any role_key changed
        old_role_keys = {r.role_key for r in tpl.roles}
        new_keys_by_id = {r.role_key for r in payload.roles}
        # Map by role_key (stable identifier across the update)
        existing_by_key = {r.role_key: r for r in tpl.roles}
        # Apply updates / adds
        for r_in in payload.roles:
            existing = existing_by_key.get(r_in.role_key)
            if existing:
                existing.role_label = r_in.role_label
                existing.order_index = r_in.order_index
                existing.default_name = r_in.default_name
                existing.default_email = r_in.default_email
            else:
                db.add(EnvelopeTemplateRole(
                    template_id=tpl.id,
                    role_key=r_in.role_key,
                    role_label=r_in.role_label,
                    order_index=r_in.order_index,
                    default_name=r_in.default_name,
                    default_email=r_in.default_email,
                ))
        # Reject removals if any field still references the role
        to_remove = old_role_keys - new_keys_by_id
        if to_remove:
            field_q = await db.execute(
                select(EnvelopeTemplateField).where(
                    EnvelopeTemplateField.template_id == tpl.id,
                    EnvelopeTemplateField.role_key.in_(to_remove),
                )
            )
            referenced = {f.role_key for f in field_q.scalars().all()}
            blocked = referenced & to_remove
            if blocked:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot remove role(s) still bound to fields: {', '.join(blocked)}",
                )
            for k in to_remove:
                r = existing_by_key.get(k)
                if r:
                    await db.delete(r)

    await db.commit()
    result = await db.execute(
        select(EnvelopeTemplate)
        .options(selectinload(EnvelopeTemplate.roles), selectinload(EnvelopeTemplate.fields))
        .where(EnvelopeTemplate.id == tpl.id)
    )
    return template_response(result.scalar_one())


@router.delete("/{template_id}", status_code=204, summary="Delete a template")
async def delete_template(
    template_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    tpl = await db.get(EnvelopeTemplate, template_id)
    if not tpl or tpl.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Template not found")
    pdf_key = tpl.pdf_storage_key
    await db.delete(tpl)
    await db.commit()
    if pdf_key:
        try:
            storage.delete(pdf_key)
        except Exception as e:
            logger.warning("template pdf cleanup failed for %s: %s", template_id, e)
    return None


@router.get("/{template_id}/pdf", summary="Download the template's source PDF")
async def template_pdf(
    template_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    from fastapi.responses import Response
    tpl = await db.get(EnvelopeTemplate, template_id)
    if not tpl or tpl.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Template not found")
    try:
        data = storage.load(tpl.pdf_storage_key)
    except Exception:
        raise HTTPException(status_code=404, detail="Template PDF missing")
    return Response(content=data, media_type="application/pdf")


@router.post("/{template_id}/instantiate", summary="Create a draft envelope from this template")
async def use_template(
    template_id: str,
    payload: TemplateInstantiate,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    try:
        env = await instantiate_template(
            template_id=template_id,
            owner_id=user.id,
            role_assignments=payload.role_assignments,
            merge_values=payload.merge_values,
            db=db,
            storage=storage,
            title_override=payload.title_override,
            message_override=payload.message_override,
            expiry_days_override=payload.expiry_days,
        )
        await db.commit()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if payload.send_immediately:
        # Reuse the existing send_envelope route logic by calling it via HTTP would be heavy;
        # instead, set up tokens + dispatch invites inline, mirroring esign.send_envelope.
        from idpkit.esign.bulk_runner import _send_envelope_emails
        from idpkit.esign.models import EnvelopeStatus, SignatureEnvelope
        result = await db.execute(
            select(SignatureEnvelope)
            .options(selectinload(SignatureEnvelope.signers), selectinload(SignatureEnvelope.fields))
            .where(SignatureEnvelope.id == env.id)
        )
        env = result.scalar_one()
        env.status = EnvelopeStatus.SENT.value
        base_url = str(request.base_url).rstrip("/")
        await _send_envelope_emails(
            env=env,
            sender_name=user.username,
            base_url=base_url,
            db=db,
        )
        await db.commit()
    return {"envelope_id": env.id, "status": env.status}
