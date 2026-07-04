"""E-Signature database models — appended to Base.metadata without touching existing tables."""

import uuid
import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from idpkit.db.models import Base, TZDateTime, generate_uuid, utcnow


class EnvelopeStatus(str, enum.Enum):
    DRAFT = "draft"
    SENT = "sent"
    VIEWED = "viewed"
    PARTIALLY_SIGNED = "partially_signed"
    COMPLETED = "completed"
    DECLINED = "declined"
    VOIDED = "voided"
    EXPIRED = "expired"


class SignerStatus(str, enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    VIEWED = "viewed"
    SIGNED = "signed"
    DECLINED = "declined"


class FieldType(str, enum.Enum):
    SIGNATURE = "signature"
    INITIALS = "initials"
    DATE = "date"
    TEXT = "text"


class SignatureEnvelope(Base):
    __tablename__ = "signature_envelopes"

    id = Column(String, primary_key=True, default=generate_uuid)
    owner_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=True)
    status = Column(String(20), default=EnvelopeStatus.DRAFT.value, index=True)
    signing_order = Column(String(20), default="parallel")  # parallel | sequential
    doc_sha256 = Column(String(64), nullable=True)
    original_file_key = Column(String(1000), nullable=True)
    finalized_file_key = Column(String(1000), nullable=True)
    audit_report_key = Column(String(1000), nullable=True)
    page_count = Column(Integer, default=1)
    expires_at = Column(TZDateTime, nullable=True)
    created_at = Column(TZDateTime, default=utcnow)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)
    completed_at = Column(TZDateTime, nullable=True)

    owner = relationship("User", foreign_keys=[owner_id])
    document = relationship("Document", foreign_keys=[document_id])
    signers = relationship("EnvelopeSigner", back_populates="envelope", cascade="all, delete-orphan", order_by="EnvelopeSigner.order_index")
    fields = relationship("SignatureField", back_populates="envelope", cascade="all, delete-orphan")
    audit_events = relationship("EnvelopeAuditEvent", back_populates="envelope", cascade="all, delete-orphan", order_by="EnvelopeAuditEvent.created_at")


class EnvelopeSigner(Base):
    __tablename__ = "envelope_signers"

    id = Column(String, primary_key=True, default=generate_uuid)
    envelope_id = Column(String, ForeignKey("signature_envelopes.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(255), nullable=False)
    order_index = Column(Integer, default=0)
    status = Column(String(20), default=SignerStatus.PENDING.value)
    token_hash = Column(String(64), nullable=True, unique=True, index=True)
    download_token_hash = Column(String(64), nullable=True, unique=True, index=True)
    download_consumed_at = Column(TZDateTime, nullable=True)
    viewed_at = Column(TZDateTime, nullable=True)
    last_viewed_at = Column(TZDateTime, nullable=True)
    signed_at = Column(TZDateTime, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(TZDateTime, default=utcnow)

    envelope = relationship("SignatureEnvelope", back_populates="signers")
    fields = relationship("SignatureField", back_populates="signer")


class SignatureField(Base):
    __tablename__ = "signature_fields"

    id = Column(String, primary_key=True, default=generate_uuid)
    envelope_id = Column(String, ForeignKey("signature_envelopes.id", ondelete="CASCADE"), nullable=False, index=True)
    signer_id = Column(String, ForeignKey("envelope_signers.id", ondelete="CASCADE"), nullable=True, index=True)
    field_type = Column(String(20), nullable=False)  # signature, initials, date, text
    page = Column(Integer, default=1)
    x_pct = Column(Float, default=0.0)
    y_pct = Column(Float, default=0.0)
    w_pct = Column(Float, default=15.0)
    h_pct = Column(Float, default=5.0)
    label = Column(String(200), nullable=True)
    value = Column(Text, nullable=True)
    is_required = Column(Integer, default=1)
    bulk_group_id = Column(String(36), nullable=True, index=True)
    created_at = Column(TZDateTime, default=utcnow)

    envelope = relationship("SignatureEnvelope", back_populates="fields")
    signer = relationship("EnvelopeSigner", back_populates="fields")


class EnvelopeAuditEvent(Base):
    __tablename__ = "envelope_audit_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    envelope_id = Column(String, ForeignKey("signature_envelopes.id", ondelete="CASCADE"), nullable=False, index=True)
    actor_email = Column(String(255), nullable=True)
    event_type = Column(String(50), nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(1000), nullable=True)
    browser_name = Column(String(100), nullable=True)
    browser_version = Column(String(50), nullable=True)
    os_name = Column(String(100), nullable=True)
    geo_country = Column(String(100), nullable=True)
    geo_city = Column(String(100), nullable=True)
    canvas_fingerprint_hash = Column(String(64), nullable=True)
    screen_resolution = Column(String(30), nullable=True)
    timezone = Column(String(100), nullable=True)
    language = Column(String(50), nullable=True)
    session_id = Column(String(64), nullable=True)
    notes = Column(String(500), nullable=True)
    extra_json = Column(Text, nullable=True)
    created_at = Column(TZDateTime, default=utcnow)

    envelope = relationship("SignatureEnvelope", back_populates="audit_events")

    __table_args__ = (
        Index("ix_audit_events_envelope_created", "envelope_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# Reusable Templates + Bulk Send (DocuSign-style)
# ---------------------------------------------------------------------------

class EnvelopeTemplate(Base):
    """A reusable envelope blueprint: PDF + role-based signers + field placements."""
    __tablename__ = "envelope_templates"

    id = Column(String, primary_key=True, default=generate_uuid)
    owner_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    title = Column(String(500), nullable=False)            # default envelope title (supports {{merge}})
    message = Column(Text, nullable=True)                   # default email message (supports {{merge}})
    signing_order = Column(String(20), default="parallel")
    expiry_days = Column(Integer, default=30)
    pdf_storage_key = Column(String(1000), nullable=False)
    doc_sha256 = Column(String(64), nullable=True)
    page_count = Column(Integer, default=1)
    merge_fields_json = Column(Text, nullable=True)         # JSON list[{key,label,type,required}]
    created_at = Column(TZDateTime, default=utcnow)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)

    owner = relationship("User", foreign_keys=[owner_id])
    roles = relationship(
        "EnvelopeTemplateRole",
        back_populates="template",
        cascade="all, delete-orphan",
        order_by="EnvelopeTemplateRole.order_index",
    )
    fields = relationship("EnvelopeTemplateField", back_populates="template", cascade="all, delete-orphan")


class EnvelopeTemplateRole(Base):
    """A signer slot on a template (e.g. 'Customer', 'Manager') — bound to a real person at use-time."""
    __tablename__ = "envelope_template_roles"

    id = Column(String, primary_key=True, default=generate_uuid)
    template_id = Column(String, ForeignKey("envelope_templates.id", ondelete="CASCADE"), nullable=False, index=True)
    role_key = Column(String(50), nullable=False)
    role_label = Column(String(100), nullable=False)
    order_index = Column(Integer, default=0)
    default_name = Column(String(200), nullable=True)
    default_email = Column(String(255), nullable=True)

    template = relationship("EnvelopeTemplate", back_populates="roles")

    __table_args__ = (
        UniqueConstraint("template_id", "role_key", name="uq_envelope_template_role_key"),
    )


class EnvelopeTemplateField(Base):
    """A field placement on a template, bound to a role_key (logical, not FK)."""
    __tablename__ = "envelope_template_fields"

    id = Column(String, primary_key=True, default=generate_uuid)
    template_id = Column(String, ForeignKey("envelope_templates.id", ondelete="CASCADE"), nullable=False, index=True)
    role_key = Column(String(50), nullable=False)
    field_type = Column(String(20), nullable=False)
    page = Column(Integer, default=1)
    x_pct = Column(Float, default=0.0)
    y_pct = Column(Float, default=0.0)
    w_pct = Column(Float, default=15.0)
    h_pct = Column(Float, default=5.0)
    label = Column(String(200), nullable=True)
    is_required = Column(Integer, default=1)
    bulk_group_id = Column(String(36), nullable=True, index=True)
    default_value = Column(Text, nullable=True)             # supports {{merge_key}} substitution

    template = relationship("EnvelopeTemplate", back_populates="fields")


class EnvelopeBatch(Base):
    """A bulk-send job: instantiates one envelope per recipient row from a template."""
    __tablename__ = "envelope_batches"

    id = Column(String, primary_key=True, default=generate_uuid)
    owner_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    template_id = Column(String, ForeignKey("envelope_templates.id", ondelete="SET NULL"), nullable=True)
    name = Column(String(200), nullable=False)
    source_label = Column(String(200), nullable=True)       # e.g. "recipients.csv" / "Pasted (12 rows)"
    status = Column(String(20), default="pending", index=True)  # pending|running|completed|failed|cancelled
    column_map_json = Column(Text, nullable=True)           # {"roles": {...}, "merge": {...}}
    send_immediately = Column(Boolean, default=True)
    total_rows = Column(Integer, default=0)
    created_count = Column(Integer, default=0)
    sent_count = Column(Integer, default=0)
    completed_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    created_at = Column(TZDateTime, default=utcnow)
    started_at = Column(TZDateTime, nullable=True)
    finished_at = Column(TZDateTime, nullable=True)

    template = relationship("EnvelopeTemplate")
    items = relationship(
        "EnvelopeBatchItem",
        back_populates="batch",
        cascade="all, delete-orphan",
        order_by="EnvelopeBatchItem.row_index",
    )


class EnvelopeBatchItem(Base):
    """Per-row state for a bulk-send job."""
    __tablename__ = "envelope_batch_items"

    id = Column(String, primary_key=True, default=generate_uuid)
    batch_id = Column(String, ForeignKey("envelope_batches.id", ondelete="CASCADE"), nullable=False, index=True)
    row_index = Column(Integer, nullable=False)
    envelope_id = Column(String, ForeignKey("signature_envelopes.id", ondelete="SET NULL"), nullable=True)
    raw_row_json = Column(Text, nullable=False)
    status = Column(String(20), default="pending", index=True)  # pending|created|sent|completed|failed|cancelled
    error = Column(Text, nullable=True)
    created_at = Column(TZDateTime, default=utcnow)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)

    batch = relationship("EnvelopeBatch", back_populates="items")
