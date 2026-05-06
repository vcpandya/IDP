"""Snapshot/instantiate logic for envelope templates.

A template captures: a frozen PDF blob, a list of signer roles, field placements
bound to those roles, default title/message text (with ``{{merge}}`` placeholders),
and a declared merge-fields schema.

Instantiation produces a fresh ``SignatureEnvelope`` (status=draft) with concrete
signers and fields bound to that envelope. The PDF blob is *copied* into the new
envelope's storage namespace so future template edits never affect existing
envelopes.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from idpkit.core.storage import StorageBackend
from idpkit.esign.merge import extract_merge_keys, render_merge
from idpkit.esign.models import (
    EnvelopeSigner,
    EnvelopeStatus,
    EnvelopeTemplate,
    EnvelopeTemplateField,
    EnvelopeTemplateRole,
    SignatureEnvelope,
    SignatureField,
    SignerStatus,
)


_VALID_KEY_RE = re.compile(r"[^a-z0-9_]+")


def normalize_role_key(label: str, taken: set[str]) -> str:
    """Turn a free-form label like 'Customer #1' into 'customer_1', avoiding collisions."""
    base = _VALID_KEY_RE.sub("_", (label or "role").strip().lower()).strip("_") or "role"
    candidate = base
    n = 2
    while candidate in taken:
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def merge_fields_schema_load(raw: str | None) -> List[Dict[str, Any]]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        if not key:
            continue
        out.append({
            "key": key,
            "label": str(item.get("label") or key),
            "type": str(item.get("type") or "text"),
            "required": bool(item.get("required", False)),
        })
    return out


def merge_fields_schema_dump(fields: List[Dict[str, Any]] | None) -> str | None:
    return json.dumps(fields or [])


# ---------------------------------------------------------------------------
# Snapshot: envelope -> template
# ---------------------------------------------------------------------------

async def snapshot_envelope_to_template(
    *,
    envelope_id: str,
    owner_id: str,
    template_name: str,
    db: AsyncSession,
    storage: StorageBackend,
    merge_fields: Optional[List[Dict[str, Any]]] = None,
) -> EnvelopeTemplate:
    """Create a new EnvelopeTemplate by snapshotting an existing envelope."""
    result = await db.execute(
        select(SignatureEnvelope)
        .options(
            selectinload(SignatureEnvelope.signers),
            selectinload(SignatureEnvelope.fields),
        )
        .where(SignatureEnvelope.id == envelope_id, SignatureEnvelope.owner_id == owner_id)
    )
    env = result.scalar_one_or_none()
    if not env:
        raise ValueError("Envelope not found")
    if not env.original_file_key:
        raise ValueError("Envelope has no source PDF to snapshot")

    pdf_bytes = storage.load(env.original_file_key)

    template_id = str(uuid.uuid4())
    pdf_key = f"esign/templates/{owner_id}/{template_id}/source.pdf"
    storage.save(pdf_key, pdf_bytes)

    # Build role rows from signers (one role per distinct signer slot).
    role_for_signer: Dict[str, str] = {}
    role_rows: List[EnvelopeTemplateRole] = []
    taken: set[str] = set()
    sorted_signers = sorted(env.signers or [], key=lambda s: (s.order_index, s.name))
    for s in sorted_signers:
        # role label defaults to the human's name; user can rename later
        label = (s.name or "Signer").strip() or "Signer"
        key = normalize_role_key(label, taken)
        taken.add(key)
        role_for_signer[s.id] = key
        role_rows.append(EnvelopeTemplateRole(
            template_id=template_id,
            role_key=key,
            role_label=label,
            order_index=s.order_index or 0,
            default_name=s.name,
            default_email=s.email,
        ))

    field_rows: List[EnvelopeTemplateField] = []
    for f in env.fields or []:
        rkey = role_for_signer.get(f.signer_id) if f.signer_id else None
        if not rkey:
            # orphan fields (unassigned) — skip silently; templates require role binding
            continue
        field_rows.append(EnvelopeTemplateField(
            template_id=template_id,
            role_key=rkey,
            field_type=f.field_type,
            page=f.page,
            x_pct=f.x_pct,
            y_pct=f.y_pct,
            w_pct=f.w_pct,
            h_pct=f.h_pct,
            label=f.label,
            is_required=f.is_required,
            bulk_group_id=f.bulk_group_id,
            default_value=f.value,  # carry over any pre-filled text
        ))

    tpl = EnvelopeTemplate(
        id=template_id,
        owner_id=owner_id,
        name=template_name.strip() or env.title,
        title=env.title,
        message=env.message,
        signing_order=env.signing_order or "parallel",
        expiry_days=30,
        pdf_storage_key=pdf_key,
        doc_sha256=env.doc_sha256,
        page_count=env.page_count or 1,
        merge_fields_json=merge_fields_schema_dump(merge_fields or []),
    )
    db.add(tpl)
    for r in role_rows:
        db.add(r)
    for f in field_rows:
        db.add(f)
    await db.flush()
    return tpl


# ---------------------------------------------------------------------------
# Instantiate: template -> envelope
# ---------------------------------------------------------------------------

async def instantiate_template(
    *,
    template_id: str,
    owner_id: str,
    role_assignments: Dict[str, Dict[str, str]],  # role_key -> {name, email}
    merge_values: Optional[Dict[str, Any]],
    db: AsyncSession,
    storage: StorageBackend,
    title_override: Optional[str] = None,
    message_override: Optional[str] = None,
    expiry_days_override: Optional[int] = None,
) -> SignatureEnvelope:
    """Materialize a draft envelope from a template + per-role recipient data + merge values."""
    result = await db.execute(
        select(EnvelopeTemplate)
        .options(
            selectinload(EnvelopeTemplate.roles),
            selectinload(EnvelopeTemplate.fields),
        )
        .where(EnvelopeTemplate.id == template_id, EnvelopeTemplate.owner_id == owner_id)
    )
    tpl = result.scalar_one_or_none()
    if not tpl:
        raise ValueError("Template not found")
    if not tpl.roles:
        raise ValueError("Template has no signer roles defined")
    if not tpl.fields:
        raise ValueError("Template has no signature fields defined")

    # Validate every role has a recipient
    missing: List[str] = []
    for role in tpl.roles:
        ra = role_assignments.get(role.role_key) or {}
        if not (ra.get("email") or "").strip() or not (ra.get("name") or "").strip():
            missing.append(role.role_label)
    if missing:
        raise ValueError(f"Missing name/email for role(s): {', '.join(missing)}")

    # Copy PDF blob to a new envelope-scoped key (so template edits never affect this envelope)
    pdf_bytes = storage.load(tpl.pdf_storage_key)
    env_id = str(uuid.uuid4())
    file_key = f"esign/{owner_id}/{env_id}/original.pdf"
    storage.save(file_key, pdf_bytes)

    # Render title + message with merge values
    rendered_title = render_merge(title_override or tpl.title, merge_values) or tpl.title
    rendered_message = render_merge(message_override if message_override is not None else tpl.message, merge_values)

    expiry_days = expiry_days_override if expiry_days_override is not None else (tpl.expiry_days or 30)
    expires_at = datetime.now(timezone.utc) + timedelta(days=int(expiry_days))

    env = SignatureEnvelope(
        id=env_id,
        owner_id=owner_id,
        title=rendered_title[:500],
        message=rendered_message,
        signing_order=tpl.signing_order or "parallel",
        doc_sha256=tpl.doc_sha256,
        original_file_key=file_key,
        page_count=tpl.page_count or 1,
        status=EnvelopeStatus.DRAFT.value,
        expires_at=expires_at,
    )
    db.add(env)
    await db.flush()

    # Create signers
    import hashlib
    import secrets
    role_to_signer: Dict[str, EnvelopeSigner] = {}
    for role in sorted(tpl.roles, key=lambda r: r.order_index):
        ra = role_assignments.get(role.role_key) or {}
        signer = EnvelopeSigner(
            envelope_id=env_id,
            name=(ra.get("name") or "").strip(),
            email=(ra.get("email") or "").strip(),
            order_index=role.order_index or 0,
            status=SignerStatus.PENDING.value,
            download_token_hash=hashlib.sha256(secrets.token_urlsafe(32).encode()).hexdigest(),
        )
        db.add(signer)
        await db.flush()
        role_to_signer[role.role_key] = signer

    # Create fields, applying merge substitution to default_value
    for tf in tpl.fields:
        signer = role_to_signer.get(tf.role_key)
        if not signer:
            continue
        rendered_default = render_merge(tf.default_value, merge_values) if tf.default_value else None
        db.add(SignatureField(
            envelope_id=env_id,
            signer_id=signer.id,
            field_type=tf.field_type,
            page=tf.page,
            x_pct=tf.x_pct,
            y_pct=tf.y_pct,
            w_pct=tf.w_pct,
            h_pct=tf.h_pct,
            label=tf.label,
            is_required=tf.is_required,
            bulk_group_id=tf.bulk_group_id,
            value=rendered_default,  # pre-filled text for text/date fields
        ))

    await db.flush()
    return env


def template_response(tpl: EnvelopeTemplate) -> dict:
    return {
        "id": tpl.id,
        "name": tpl.name,
        "title": tpl.title,
        "message": tpl.message,
        "signing_order": tpl.signing_order,
        "expiry_days": tpl.expiry_days,
        "page_count": tpl.page_count,
        "merge_fields": merge_fields_schema_load(tpl.merge_fields_json),
        "created_at": tpl.created_at.isoformat() if tpl.created_at else None,
        "updated_at": tpl.updated_at.isoformat() if tpl.updated_at else None,
        "roles": [
            {
                "id": r.id,
                "role_key": r.role_key,
                "role_label": r.role_label,
                "order_index": r.order_index,
                "default_name": r.default_name,
                "default_email": r.default_email,
            }
            for r in (tpl.roles or [])
        ],
        "fields": [
            {
                "id": f.id,
                "role_key": f.role_key,
                "field_type": f.field_type,
                "page": f.page,
                "x_pct": f.x_pct,
                "y_pct": f.y_pct,
                "w_pct": f.w_pct,
                "h_pct": f.h_pct,
                "label": f.label,
                "is_required": f.is_required,
                "bulk_group_id": f.bulk_group_id,
                "default_value": f.default_value,
            }
            for f in (tpl.fields or [])
        ],
        "detected_merge_keys": extract_merge_keys(
            tpl.title,
            tpl.message,
            *[f.default_value for f in (tpl.fields or [])],
        ),
    }
