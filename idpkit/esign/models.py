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
    viewed_at = Column(TZDateTime, nullable=True)
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
