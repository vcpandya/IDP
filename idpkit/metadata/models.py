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
