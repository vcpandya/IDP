"""E-Signature API routes."""

import hashlib
import io
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from idpkit.api.deps import get_current_user, get_storage
from idpkit.core.storage import StorageBackend
from idpkit.db.models import Document, User
from idpkit.db.session import get_db
from idpkit.esign.models import (
    EnvelopeAuditEvent,
    EnvelopeSigner,
    EnvelopeStatus,
    FieldType,
    SignatureEnvelope,
    SignatureField,
    SignerStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/esign", tags=["esign"])

ESIGN_EXPIRY_DAYS = int(os.getenv("ESIGN_EXPIRY_DAYS", "30"))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SignerIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: str = Field(..., min_length=3)
    order_index: int = 0


class FieldIn(BaseModel):
    id: Optional[str] = None
    signer_id: Optional[str] = None
    field_type: str = "signature"
    page: int = 1
    x_pct: float = 0.0
    y_pct: float = 0.0
    w_pct: float = 15.0
    h_pct: float = 5.0
    label: Optional[str] = None
    is_required: int = 1


class CreateEnvelopeIn(BaseModel):
    document_id: Optional[str] = None
    title: str = Field(..., min_length=1, max_length=500)
    message: Optional[str] = None
    signing_order: str = "parallel"
    signers: list[SignerIn] = []


class UpdateFieldsIn(BaseModel):
    fields: list[FieldIn]


class SubmitSignatureIn(BaseModel):
    fields: list[dict]
    canvas_fingerprint_hash: Optional[str] = None
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_secret_key() -> str:
    return os.getenv("SECRET_KEY") or os.getenv("SESSION_SECRET") or os.getenv("IDP_SECRET_KEY") or "changeme"


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _parse_ua(ua: str) -> dict:
    """Very simple UA parser — avoids adding ua-parser dependency."""
    ua_lower = ua.lower()
    browser = "Unknown"
    browser_ver = ""
    os_name = "Unknown"

    if "edg/" in ua_lower:
        browser = "Edge"
        m = ua.split("Edg/")[-1].split(" ")[0]
        browser_ver = m
    elif "chrome/" in ua_lower and "chromium" not in ua_lower:
        browser = "Chrome"
        m = ua.split("Chrome/")[-1].split(" ")[0]
        browser_ver = m
    elif "firefox/" in ua_lower:
        browser = "Firefox"
        m = ua.split("Firefox/")[-1].split(" ")[0]
        browser_ver = m
    elif "safari/" in ua_lower and "chrome" not in ua_lower:
        browser = "Safari"
        m = ua.split("Version/")[-1].split(" ")[0] if "Version/" in ua else ""
        browser_ver = m

    if "windows" in ua_lower:
        os_name = "Windows"
    elif "mac os" in ua_lower or "macos" in ua_lower:
        os_name = "macOS"
    elif "android" in ua_lower:
        os_name = "Android"
    elif "iphone" in ua_lower or "ipad" in ua_lower:
        os_name = "iOS"
    elif "linux" in ua_lower:
        os_name = "Linux"

    return {"browser_name": browser, "browser_version": browser_ver, "os_name": os_name}


async def _geo_lookup(ip: str) -> dict:
    """Best-effort IP geolocation using ip-api.com (free, no key needed)."""
    if ip in ("unknown", "127.0.0.1", "::1"):
        return {}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"http://ip-api.com/json/{ip}?fields=country,city,status")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    return {"geo_country": data.get("country", ""), "geo_city": data.get("city", "")}
    except Exception:
        pass
    return {}


async def _log_event(
    db: AsyncSession,
    envelope_id: str,
    event_type: str,
    actor_email: str = "",
    request: Optional[Request] = None,
    extra: Optional[dict] = None,
) -> None:
    ip = ""
    ua_str = ""
    ua_info = {}
    geo = {}
    if request:
        ip = _client_ip(request)
        ua_str = request.headers.get("user-agent", "")
        ua_info = _parse_ua(ua_str)
        geo = await _geo_lookup(ip)

    ev = EnvelopeAuditEvent(
        envelope_id=envelope_id,
        actor_email=actor_email,
        event_type=event_type,
        ip_address=ip,
        user_agent=ua_str[:500],
        geo_country=geo.get("geo_country", ""),
        geo_city=geo.get("geo_city", ""),
        **(ua_info or {}),
        **(extra or {}),
    )
    db.add(ev)


def _envelope_response(env: SignatureEnvelope) -> dict:
    return {
        "id": env.id,
        "title": env.title,
        "message": env.message,
        "status": env.status,
        "signing_order": env.signing_order,
        "doc_sha256": env.doc_sha256,
        "page_count": env.page_count,
        "expires_at": env.expires_at.isoformat() if env.expires_at else None,
        "created_at": env.created_at.isoformat() if env.created_at else None,
        "completed_at": env.completed_at.isoformat() if env.completed_at else None,
        "signers": [
            {
                "id": s.id,
                "name": s.name,
                "email": s.email,
                "order_index": s.order_index,
                "status": s.status,
                "viewed_at": s.viewed_at.isoformat() if s.viewed_at else None,
                "signed_at": s.signed_at.isoformat() if s.signed_at else None,
            }
            for s in (env.signers or [])
        ],
        "fields": [
            {
                "id": f.id,
                "signer_id": f.signer_id,
                "field_type": f.field_type,
                "page": f.page,
                "x_pct": f.x_pct,
                "y_pct": f.y_pct,
                "w_pct": f.w_pct,
                "h_pct": f.h_pct,
                "label": f.label,
                "is_required": f.is_required,
                "has_value": bool(f.value),
            }
            for f in (env.fields or [])
        ],
        "has_finalized": bool(env.finalized_file_key),
        "has_audit_report": bool(env.audit_report_key),
    }


def _load_envelope_pdf(env: SignatureEnvelope, storage: StorageBackend) -> bytes:
    key = env.original_file_key
    if not key:
        raise HTTPException(status_code=400, detail="Envelope has no source document")
    return storage.load(key)


# ---------------------------------------------------------------------------
# Sender Routes (authenticated)
# ---------------------------------------------------------------------------

@router.post("/envelopes", status_code=201)
async def create_envelope(
    body: CreateEnvelopeIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    """Create a new envelope from an existing document."""
    from idpkit.esign.pdf_utils import compute_sha256, get_page_count

    if not body.document_id:
        raise HTTPException(status_code=400, detail="document_id is required")

    result = await db.execute(
        select(Document).where(Document.id == body.document_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.file_path:
        raise HTTPException(status_code=400, detail="Document has no stored file")

    pdf_bytes = storage.load(doc.file_path)
    sha = compute_sha256(pdf_bytes)
    pages = get_page_count(pdf_bytes)

    env = SignatureEnvelope(
        owner_id=user.id,
        document_id=doc.id,
        title=body.title or doc.filename,
        message=body.message,
        signing_order=body.signing_order,
        doc_sha256=sha,
        original_file_key=doc.file_path,
        page_count=pages,
        status=EnvelopeStatus.DRAFT.value,
        expires_at=datetime.now(timezone.utc) + timedelta(days=ESIGN_EXPIRY_DAYS),
    )
    db.add(env)
    await db.flush()

    for s in body.signers:
        signer = EnvelopeSigner(
            envelope_id=env.id,
            name=s.name,
            email=s.email,
            order_index=s.order_index,
        )
        db.add(signer)

    await _log_event(db, env.id, "envelope_created", actor_email=user.email or user.username)
    await db.commit()
    await db.refresh(env)

    result2 = await db.execute(
        select(SignatureEnvelope)
        .options(selectinload(SignatureEnvelope.signers), selectinload(SignatureEnvelope.fields))
        .where(SignatureEnvelope.id == env.id)
    )
    env = result2.scalar_one()
    return _envelope_response(env)


@router.post("/envelopes/upload", status_code=201)
async def create_envelope_with_upload(
    title: str,
    message: Optional[str] = None,
    signing_order: str = "parallel",
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    """Create a new envelope by uploading a PDF directly."""
    from idpkit.esign.pdf_utils import compute_sha256, get_page_count

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for e-signature")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    sha = compute_sha256(content)
    try:
        pages = get_page_count(content)
    except Exception:
        pages = 1

    env_id = str(uuid.uuid4())
    file_key = f"esign/{user.id}/{env_id}/original.pdf"
    storage.save(file_key, content)

    env = SignatureEnvelope(
        id=env_id,
        owner_id=user.id,
        title=title or file.filename,
        message=message,
        signing_order=signing_order,
        doc_sha256=sha,
        original_file_key=file_key,
        page_count=pages,
        status=EnvelopeStatus.DRAFT.value,
        expires_at=datetime.now(timezone.utc) + timedelta(days=ESIGN_EXPIRY_DAYS),
    )
    db.add(env)
    await _log_event(db, env_id, "envelope_created", actor_email=user.email or user.username)
    await db.commit()
    await db.refresh(env)

    result = await db.execute(
        select(SignatureEnvelope)
        .options(selectinload(SignatureEnvelope.signers), selectinload(SignatureEnvelope.fields))
        .where(SignatureEnvelope.id == env.id)
    )
    env = result.scalar_one()
    return _envelope_response(env)


@router.get("/envelopes")
async def list_envelopes(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SignatureEnvelope)
        .options(selectinload(SignatureEnvelope.signers), selectinload(SignatureEnvelope.fields))
        .where(SignatureEnvelope.owner_id == user.id)
        .order_by(SignatureEnvelope.created_at.desc())
    )
    envelopes = result.scalars().all()
    return [_envelope_response(e) for e in envelopes]


@router.get("/envelopes/{envelope_id}")
async def get_envelope(
    envelope_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SignatureEnvelope)
        .options(
            selectinload(SignatureEnvelope.signers),
            selectinload(SignatureEnvelope.fields),
            selectinload(SignatureEnvelope.audit_events),
        )
        .where(SignatureEnvelope.id == envelope_id, SignatureEnvelope.owner_id == user.id)
    )
    env = result.scalar_one_or_none()
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")
    resp = _envelope_response(env)
    resp["audit_events"] = [
        {
            "id": ev.id,
            "actor_email": ev.actor_email,
            "event_type": ev.event_type,
            "ip_address": ev.ip_address,
            "browser_name": ev.browser_name,
            "browser_version": ev.browser_version,
            "os_name": ev.os_name,
            "geo_country": ev.geo_country,
            "geo_city": ev.geo_city,
            "canvas_fingerprint_hash": ev.canvas_fingerprint_hash,
            "screen_resolution": ev.screen_resolution,
            "timezone": ev.timezone,
            "language": ev.language,
            "session_id": ev.session_id,
            "created_at": ev.created_at.isoformat() if ev.created_at else None,
        }
        for ev in (env.audit_events or [])
    ]
    return resp


@router.put("/envelopes/{envelope_id}/signers")
async def update_signers(
    envelope_id: str,
    signers: list[SignerIn],
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SignatureEnvelope).where(
            SignatureEnvelope.id == envelope_id, SignatureEnvelope.owner_id == user.id
        )
    )
    env = result.scalar_one_or_none()
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")
    if env.status != EnvelopeStatus.DRAFT.value:
        raise HTTPException(status_code=400, detail="Cannot edit signers after envelope is sent")

    # Delete existing signers
    existing = await db.execute(
        select(EnvelopeSigner).where(EnvelopeSigner.envelope_id == envelope_id)
    )
    for s in existing.scalars().all():
        await db.delete(s)
    await db.flush()

    for s in signers:
        signer = EnvelopeSigner(
            envelope_id=envelope_id,
            name=s.name,
            email=s.email,
            order_index=s.order_index,
        )
        db.add(signer)

    await db.commit()
    return {"detail": "Signers updated"}


@router.put("/envelopes/{envelope_id}/fields")
async def update_fields(
    envelope_id: str,
    body: UpdateFieldsIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SignatureEnvelope).where(
            SignatureEnvelope.id == envelope_id, SignatureEnvelope.owner_id == user.id
        )
    )
    env = result.scalar_one_or_none()
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")
    if env.status != EnvelopeStatus.DRAFT.value:
        raise HTTPException(status_code=400, detail="Cannot edit fields after envelope is sent")

    existing = await db.execute(
        select(SignatureField).where(SignatureField.envelope_id == envelope_id)
    )
    for f in existing.scalars().all():
        await db.delete(f)
    await db.flush()

    for f in body.fields:
        field = SignatureField(
            envelope_id=envelope_id,
            signer_id=f.signer_id,
            field_type=f.field_type,
            page=f.page,
            x_pct=f.x_pct,
            y_pct=f.y_pct,
            w_pct=f.w_pct,
            h_pct=f.h_pct,
            label=f.label,
            is_required=f.is_required,
        )
        db.add(field)

    await db.commit()
    return {"detail": "Fields saved"}


@router.post("/envelopes/{envelope_id}/send")
async def send_envelope(
    envelope_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    result = await db.execute(
        select(SignatureEnvelope)
        .options(selectinload(SignatureEnvelope.signers), selectinload(SignatureEnvelope.fields))
        .where(SignatureEnvelope.id == envelope_id, SignatureEnvelope.owner_id == user.id)
    )
    env = result.scalar_one_or_none()
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")
    if env.status != EnvelopeStatus.DRAFT.value:
        raise HTTPException(status_code=400, detail="Envelope already sent")
    if not env.signers:
        raise HTTPException(status_code=400, detail="Add at least one signer before sending")
    if not env.fields:
        raise HTTPException(status_code=400, detail="Add at least one signature field before sending")

    from idpkit.esign.email import send_signing_invitation

    base_url = str(request.base_url).rstrip("/")
    env.status = EnvelopeStatus.SENT.value

    for signer in env.signers:
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        signer.token_hash = token_hash
        signer.status = SignerStatus.SENT.value
        db.add(signer)

        signing_url = f"{base_url}/sign/{raw_token}"
        expires_str = env.expires_at.strftime("%B %d, %Y") if env.expires_at else None

        await send_signing_invitation(
            signer_name=signer.name,
            signer_email=signer.email,
            sender_name=user.username,
            envelope_title=env.title,
            signing_url=signing_url,
            message=env.message,
            expires_at=expires_str,
        )
        await _log_event(
            db, envelope_id, "invitation_sent",
            actor_email=user.email or user.username,
            extra={"session_id": signer.id},
        )

    db.add(env)
    await db.commit()
    return {"detail": "Envelope sent", "signer_count": len(env.signers)}


@router.get("/envelopes/{envelope_id}/pdf-page/{page}")
async def get_pdf_page(
    envelope_id: str,
    page: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    from idpkit.esign.pdf_utils import render_page_to_image

    result = await db.execute(
        select(SignatureEnvelope).where(
            SignatureEnvelope.id == envelope_id, SignatureEnvelope.owner_id == user.id
        )
    )
    env = result.scalar_one_or_none()
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")

    pdf_bytes = _load_envelope_pdf(env, storage)
    img_b64 = await _run_sync(render_page_to_image, pdf_bytes, page)
    return {"page": page, "image_b64": img_b64, "total_pages": env.page_count}


@router.get("/envelopes/{envelope_id}/download")
async def download_signed_pdf(
    envelope_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    result = await db.execute(
        select(SignatureEnvelope).where(
            SignatureEnvelope.id == envelope_id, SignatureEnvelope.owner_id == user.id
        )
    )
    env = result.scalar_one_or_none()
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")
    if env.status != EnvelopeStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Envelope is not yet completed")
    if not env.finalized_file_key:
        raise HTTPException(status_code=404, detail="Signed document not found")

    pdf_bytes = storage.load(env.finalized_file_key)
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in env.title)
    filename = f"signed_{safe_title[:60]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/envelopes/{envelope_id}/audit-report")
async def download_audit_report(
    envelope_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    result = await db.execute(
        select(SignatureEnvelope)
        .options(selectinload(SignatureEnvelope.signers), selectinload(SignatureEnvelope.audit_events))
        .where(SignatureEnvelope.id == envelope_id, SignatureEnvelope.owner_id == user.id)
    )
    env = result.scalar_one_or_none()
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")

    # Regenerate on-the-fly if needed (also try cached version)
    if env.audit_report_key and storage.exists(env.audit_report_key):
        pdf_bytes = storage.load(env.audit_report_key)
    else:
        pdf_bytes = await _generate_and_store_audit_report(env, storage, db)

    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in env.title)
    filename = f"audit_report_{safe_title[:50]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/envelopes/{envelope_id}/void")
async def void_envelope(
    envelope_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SignatureEnvelope)
        .options(selectinload(SignatureEnvelope.signers))
        .where(SignatureEnvelope.id == envelope_id, SignatureEnvelope.owner_id == user.id)
    )
    env = result.scalar_one_or_none()
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")
    if env.status == EnvelopeStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Cannot void a completed envelope")

    from idpkit.esign.email import send_void_notice

    env.status = EnvelopeStatus.VOIDED.value
    # Invalidate tokens
    for s in env.signers:
        if s.status not in (SignerStatus.SIGNED.value,):
            s.token_hash = None
            db.add(s)
        if s.status != SignerStatus.SIGNED.value:
            await send_void_notice(
                recipient_email=s.email,
                recipient_name=s.name,
                envelope_title=env.title,
                voided_by=user.username,
            )

    db.add(env)
    await _log_event(db, envelope_id, "envelope_voided", actor_email=user.email or user.username, request=request)
    await db.commit()
    return {"detail": "Envelope voided"}


@router.post("/envelopes/{envelope_id}/resend/{signer_id}")
async def resend_invitation(
    envelope_id: str,
    signer_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SignatureEnvelope).where(
            SignatureEnvelope.id == envelope_id, SignatureEnvelope.owner_id == user.id
        )
    )
    env = result.scalar_one_or_none()
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")
    if env.status not in (EnvelopeStatus.SENT.value,):
        raise HTTPException(status_code=400, detail="Envelope is not in sent state")

    result2 = await db.execute(
        select(EnvelopeSigner).where(
            EnvelopeSigner.id == signer_id, EnvelopeSigner.envelope_id == envelope_id
        )
    )
    signer = result2.scalar_one_or_none()
    if not signer:
        raise HTTPException(status_code=404, detail="Signer not found")
    if signer.status == SignerStatus.SIGNED.value:
        raise HTTPException(status_code=400, detail="Signer has already signed")

    from idpkit.esign.email import send_signing_invitation

    raw_token = secrets.token_urlsafe(32)
    signer.token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    db.add(signer)

    base_url = str(request.base_url).rstrip("/")
    signing_url = f"{base_url}/sign/{raw_token}"
    await send_signing_invitation(
        signer_name=signer.name,
        signer_email=signer.email,
        sender_name=user.username,
        envelope_title=env.title,
        signing_url=signing_url,
        message=env.message,
    )
    await _log_event(db, envelope_id, "invitation_resent", actor_email=user.email or user.username)
    await db.commit()
    return {"detail": "Invitation resent"}


# ---------------------------------------------------------------------------
# Public Signing Routes (no auth required)
# ---------------------------------------------------------------------------

def _find_signer_by_token(token: str, db):
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    return select(EnvelopeSigner).where(EnvelopeSigner.token_hash == token_hash)


@router.get("/sign/{token}")
async def get_signing_context(
    token: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    """Public: validate token, return signer context and fields."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    result = await db.execute(
        select(EnvelopeSigner)
        .options(selectinload(EnvelopeSigner.envelope).selectinload(SignatureEnvelope.fields))
        .where(EnvelopeSigner.token_hash == token_hash)
    )
    signer = result.scalar_one_or_none()
    if not signer:
        raise HTTPException(status_code=404, detail="Invalid or expired signing link")

    env = signer.envelope
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")

    now = datetime.now(timezone.utc)
    if env.status == EnvelopeStatus.VOIDED.value:
        raise HTTPException(status_code=410, detail="This envelope has been voided")
    if env.status == EnvelopeStatus.EXPIRED.value or (env.expires_at and now > env.expires_at):
        env.status = EnvelopeStatus.EXPIRED.value
        db.add(env)
        await db.commit()
        raise HTTPException(status_code=410, detail="This signing link has expired")
    if signer.status == SignerStatus.SIGNED.value:
        return {
            "already_signed": True,
            "envelope_title": env.title,
            "signer_name": signer.name,
            "completed": env.status == EnvelopeStatus.COMPLETED.value,
            "download_available": env.status == EnvelopeStatus.COMPLETED.value,
        }

    # Check sequential order
    if env.signing_order == "sequential" and signer.order_index > 0:
        prev_result = await db.execute(
            select(EnvelopeSigner).where(
                EnvelopeSigner.envelope_id == env.id,
                EnvelopeSigner.order_index < signer.order_index,
            )
        )
        prev_signers = prev_result.scalars().all()
        if any(s.status != SignerStatus.SIGNED.value for s in prev_signers):
            raise HTTPException(status_code=400, detail="Waiting for previous signers to complete first")

    # Log viewed event
    if not signer.viewed_at:
        signer.viewed_at = now
        signer.status = SignerStatus.VIEWED.value
        db.add(signer)
        await _log_event(db, env.id, "document_viewed", actor_email=signer.email, request=request)
        await db.commit()

    my_fields = [f for f in (env.fields or []) if f.signer_id == signer.id]

    return {
        "already_signed": False,
        "envelope_id": env.id,
        "envelope_title": env.title,
        "envelope_message": env.message,
        "signer_id": signer.id,
        "signer_name": signer.name,
        "signer_email": signer.email,
        "page_count": env.page_count,
        "fields": [
            {
                "id": f.id,
                "field_type": f.field_type,
                "page": f.page,
                "x_pct": f.x_pct,
                "y_pct": f.y_pct,
                "w_pct": f.w_pct,
                "h_pct": f.h_pct,
                "label": f.label,
                "is_required": f.is_required,
            }
            for f in my_fields
        ],
    }


@router.get("/sign/{token}/page/{page}")
async def get_signing_page_image(
    token: str,
    page: int,
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    """Public: get a rendered PDF page for the signing UI."""
    from idpkit.esign.pdf_utils import render_page_to_image

    token_hash = hashlib.sha256(token.encode()).hexdigest()
    result = await db.execute(
        select(EnvelopeSigner)
        .options(selectinload(EnvelopeSigner.envelope))
        .where(EnvelopeSigner.token_hash == token_hash)
    )
    signer = result.scalar_one_or_none()
    if not signer or not signer.envelope:
        raise HTTPException(status_code=404, detail="Invalid or expired signing link")

    env = signer.envelope
    if env.status in (EnvelopeStatus.VOIDED.value, EnvelopeStatus.EXPIRED.value):
        raise HTTPException(status_code=410, detail="Envelope is no longer active")

    pdf_bytes = _load_envelope_pdf(env, storage)
    img_b64 = await _run_sync(render_page_to_image, pdf_bytes, page)
    return {"page": page, "image_b64": img_b64, "total_pages": env.page_count}


@router.post("/sign/{token}/submit")
async def submit_signature(
    token: str,
    body: SubmitSignatureIn,
    request: Request,
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    """Public: submit signed fields, finalize envelope if all signers done."""
    from idpkit.esign.pdf_utils import (
        overlay_signatures,
        append_audit_certificate_page,
        render_page_to_image,
    )

    token_hash = hashlib.sha256(token.encode()).hexdigest()
    result = await db.execute(
        select(EnvelopeSigner)
        .options(
            selectinload(EnvelopeSigner.envelope).selectinload(SignatureEnvelope.fields),
            selectinload(EnvelopeSigner.envelope).selectinload(SignatureEnvelope.signers),
        )
        .where(EnvelopeSigner.token_hash == token_hash)
    )
    signer = result.scalar_one_or_none()
    if not signer:
        raise HTTPException(status_code=404, detail="Invalid or expired signing link")

    env = signer.envelope
    if not env:
        raise HTTPException(status_code=404, detail="Envelope not found")

    now = datetime.now(timezone.utc)
    if env.status == EnvelopeStatus.VOIDED.value:
        raise HTTPException(status_code=410, detail="Envelope has been voided")
    if signer.status == SignerStatus.SIGNED.value:
        raise HTTPException(status_code=400, detail="Already signed")

    ip = _client_ip(request)
    ua_str = request.headers.get("user-agent", "")
    ua_info = _parse_ua(ua_str)
    geo = await _geo_lookup(ip)

    # Store field values
    field_map = {f.id: f for f in (env.fields or []) if f.signer_id == signer.id}
    for submitted in body.fields:
        fid = submitted.get("id")
        value = submitted.get("value", "")
        if fid and fid in field_map:
            field_map[fid].value = value
            db.add(field_map[fid])

    # Mark signer complete
    signer.signed_at = now
    signer.status = SignerStatus.SIGNED.value
    signer.ip_address = ip
    signer.user_agent = ua_str[:500]
    db.add(signer)

    # Audit event with forensic data
    ev = EnvelopeAuditEvent(
        envelope_id=env.id,
        actor_email=signer.email,
        event_type="document_signed",
        ip_address=ip,
        user_agent=ua_str[:500],
        browser_name=ua_info.get("browser_name"),
        browser_version=ua_info.get("browser_version"),
        os_name=ua_info.get("os_name"),
        geo_country=geo.get("geo_country", ""),
        geo_city=geo.get("geo_city", ""),
        canvas_fingerprint_hash=body.canvas_fingerprint_hash,
        screen_resolution=body.screen_resolution,
        timezone=body.timezone,
        language=body.language,
        session_id=body.session_id,
    )
    db.add(ev)
    await db.flush()

    # Check if all signers have signed
    all_signers = env.signers or []
    unsigned = [s for s in all_signers if s.id != signer.id and s.status != SignerStatus.SIGNED.value]

    if not unsigned:
        # All signed — finalize
        env.status = EnvelopeStatus.COMPLETED.value
        env.completed_at = now
        db.add(env)
        await db.flush()

        # Reload all events for audit report
        ev_result = await db.execute(
            select(EnvelopeAuditEvent)
            .where(EnvelopeAuditEvent.envelope_id == env.id)
            .order_by(EnvelopeAuditEvent.created_at)
        )
        all_events = ev_result.scalars().all()

        # Build overlay data from all fields
        all_fields_result = await db.execute(
            select(SignatureField).where(SignatureField.envelope_id == env.id)
        )
        all_fields = all_fields_result.scalars().all()

        overlay_data = [
            {
                "page": f.page,
                "x_pct": f.x_pct,
                "y_pct": f.y_pct,
                "w_pct": f.w_pct,
                "h_pct": f.h_pct,
                "field_type": f.field_type,
                "value": f.value,
            }
            for f in all_fields
        ]

        signers_data = [
            {
                "name": s.name,
                "email": s.email,
                "status": s.status,
                "ip_address": s.ip_address,
                "user_agent": s.user_agent,
                "signed_at": s.signed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if s.signed_at else None,
            }
            for s in all_signers
        ]

        events_data = [
            {
                "actor_email": e.actor_email,
                "event_type": e.event_type,
                "ip_address": e.ip_address,
                "browser_name": e.browser_name,
                "browser_version": e.browser_version,
                "os_name": e.os_name,
                "geo_country": e.geo_country,
                "geo_city": e.geo_city,
                "canvas_fingerprint_hash": e.canvas_fingerprint_hash,
                "screen_resolution": e.screen_resolution,
                "timezone": e.timezone,
                "language": e.language,
                "session_id": e.session_id,
                "created_at": e.created_at.strftime("%Y-%m-%d %H:%M:%S") if e.created_at else "",
            }
            for e in all_events
        ]

        try:
            orig_pdf = _load_envelope_pdf(env, storage)
            signed_pdf = await _run_sync(overlay_signatures, orig_pdf, overlay_data)
            signed_pdf_with_cert = await _run_sync(
                append_audit_certificate_page,
                signed_pdf,
                env.id,
                env.title,
                env.doc_sha256 or "",
                signers_data,
                events_data,
            )
        except Exception as exc:
            logger.error("PDF finalization failed: %s", exc)
            signed_pdf_with_cert = _load_envelope_pdf(env, storage)

        finalized_key = f"esign/{env.owner_id}/{env.id}/signed.pdf"
        storage.save(finalized_key, signed_pdf_with_cert)
        env.finalized_file_key = finalized_key
        db.add(env)

        # Generate audit report
        try:
            report_pdf = await _generate_and_store_audit_report(env, storage, db, events_data=events_data, signers_data=signers_data)
        except Exception as exc:
            logger.error("Audit report generation failed: %s", exc)

        await db.flush()

        # Send completion emails
        from idpkit.esign.email import send_completion_notice

        base_url = str(request.base_url).rstrip("/")
        download_url = f"{base_url}/esign"
        for s in all_signers:
            await send_completion_notice(
                recipient_email=s.email,
                recipient_name=s.name,
                envelope_title=env.title,
                download_url=download_url,
                pdf_bytes=signed_pdf_with_cert,
                filename=f"signed_{env.title[:40]}.pdf",
            )

        # Also notify the sender
        result_owner = await db.execute(
            select(User).where(User.id == env.owner_id)
        )
        owner = result_owner.scalar_one_or_none()
        if owner and owner.email:
            await send_completion_notice(
                recipient_email=owner.email,
                recipient_name=owner.username,
                envelope_title=env.title,
                download_url=f"{base_url}/esign",
                pdf_bytes=signed_pdf_with_cert,
            )

        await _log_event(db, env.id, "envelope_completed")
        await db.commit()

        return {"signed": True, "completed": True, "envelope_title": env.title}

    await db.commit()
    return {"signed": True, "completed": False, "envelope_title": env.title}


@router.get("/sign/{token}/download")
async def signer_download(
    token: str,
    db: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage),
):
    """Public: allow signer to download the finalized document."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    result = await db.execute(
        select(EnvelopeSigner)
        .options(selectinload(EnvelopeSigner.envelope))
        .where(EnvelopeSigner.token_hash == token_hash)
    )
    signer = result.scalar_one_or_none()
    if not signer or not signer.envelope:
        raise HTTPException(status_code=404, detail="Invalid or expired signing link")

    env = signer.envelope
    if env.status != EnvelopeStatus.COMPLETED.value or not env.finalized_file_key:
        raise HTTPException(status_code=400, detail="Document is not yet complete")

    pdf_bytes = storage.load(env.finalized_file_key)
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in env.title)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="signed_{safe_title[:60]}.pdf"'},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _run_sync(fn, *args):
    """Run a sync function in a thread pool to avoid blocking the event loop."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


async def _generate_and_store_audit_report(
    env: SignatureEnvelope,
    storage: StorageBackend,
    db: AsyncSession,
    events_data: Optional[list] = None,
    signers_data: Optional[list] = None,
) -> bytes:
    from idpkit.esign.pdf_utils import generate_audit_report_pdf

    if events_data is None:
        ev_result = await db.execute(
            select(EnvelopeAuditEvent)
            .where(EnvelopeAuditEvent.envelope_id == env.id)
            .order_by(EnvelopeAuditEvent.created_at)
        )
        events_data = [
            {
                "actor_email": e.actor_email,
                "event_type": e.event_type,
                "ip_address": e.ip_address,
                "browser_name": e.browser_name,
                "browser_version": e.browser_version,
                "os_name": e.os_name,
                "geo_country": e.geo_country,
                "geo_city": e.geo_city,
                "canvas_fingerprint_hash": e.canvas_fingerprint_hash,
                "screen_resolution": e.screen_resolution,
                "timezone": e.timezone,
                "language": e.language,
                "session_id": e.session_id,
                "created_at": e.created_at.strftime("%Y-%m-%d %H:%M:%S") if e.created_at else "",
            }
            for e in ev_result.scalars().all()
        ]

    if signers_data is None:
        s_result = await db.execute(
            select(EnvelopeSigner).where(EnvelopeSigner.envelope_id == env.id)
        )
        signers_data = [
            {
                "name": s.name,
                "email": s.email,
                "status": s.status,
                "ip_address": s.ip_address,
                "signed_at": s.signed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if s.signed_at else None,
            }
            for s in s_result.scalars().all()
        ]

    secret = _get_secret_key()
    report_pdf = await _run_sync(
        generate_audit_report_pdf,
        env.id,
        env.title,
        env.doc_sha256 or "",
        signers_data,
        events_data,
        secret,
        f"",
    )

    report_key = f"esign/{env.owner_id}/{env.id}/audit_report.pdf"
    storage.save(report_key, report_pdf)
    env.audit_report_key = report_key
    db.add(env)
    await db.flush()

    return report_pdf
