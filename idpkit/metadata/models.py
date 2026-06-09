"""Database model for smart document metadata facets.

A *facet* is one extracted key-value pair for a document (e.g. key="judge",
value="Justice A. Sharma"). List-typed fields produce one facet row per value.
``value_norm`` is a lowercased/trimmed form used for grouping and matching so
that "High Court" and "high court " collapse into one facet value.
"""

from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from idpkit.db.models import Base, TZDateTime, generate_uuid, utcnow


class DocumentFacet(Base):
    """One extracted typed key-value pair attached to a document."""

    __tablename__ = "document_facets"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category = Column(String(100), nullable=True, index=True)
    key = Column(String(100), nullable=False, index=True)
    label = Column(String(200), nullable=True)
    value = Column(String(1000), nullable=False)
    value_norm = Column(String(1000), nullable=False, index=True)
    confidence = Column(Integer, default=80)  # 0-100
    created_at = Column(TZDateTime, default=utcnow)

    __table_args__ = (
        Index("ix_facets_key_value", "key", "value_norm"),
        Index("ix_facets_doc_key", "document_id", "key"),
        # Guarantees a document never accumulates duplicate facets even if two
        # extractions race (e.g. background post-index hook + manual reprocess).
        UniqueConstraint(
            "document_id", "key", "value_norm", name="uq_facet_doc_key_value"
        ),
    )


class MetadataJob(Base):
    """Tracks a background smart-metadata (re)extraction run for live progress.

    A single row per "Reprocess" / "Add to Document Map" action. The runner
    updates ``processed``/``failed``/``skipped``/``current`` as it works so any
    Gunicorn worker can serve a poll request (progress lives in the DB, not in
    a worker's memory).
    """

    __tablename__ = "metadata_jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    owner_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(20), default="pending", index=True)  # pending|running|completed|failed
    scope = Column(String(20), nullable=True)  # all|missing|selection|tag
    label = Column(String(200), nullable=True)  # human label, e.g. a KB name
    total = Column(Integer, default=0)
    processed = Column(Integer, default=0)
    failed = Column(Integer, default=0)
    skipped = Column(Integer, default=0)
    current = Column(String(300), nullable=True)  # filename currently being processed
    error = Column(Text, nullable=True)
    created_at = Column(TZDateTime, default=utcnow, index=True)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        Index("ix_metadata_jobs_owner_created", "owner_id", "created_at"),
    )
