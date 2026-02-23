"""Knowledge Graph database models — entities, edges, and mentions."""

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)

from idpkit.db.models import Base, TZDateTime, generate_uuid, utcnow


class Entity(Base):
    """Global entity registry — one row per unique entity across all documents."""

    __tablename__ = "entities"

    id = Column(String, primary_key=True, default=generate_uuid)
    canonical_name = Column(String(500), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False, index=True)
    description = Column(String(2000), nullable=True)
    aliases = Column(JSON, nullable=True)  # list of alternative names
    first_document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    document_count = Column(Integer, default=1)
    created_at = Column(TZDateTime, default=utcnow)
    updated_at = Column(TZDateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        Index("ix_entities_name_type", "canonical_name", "entity_type"),
    )


class EntityMention(Base):
    """Links entities to specific tree nodes in specific documents."""

    __tablename__ = "entity_mentions"

    id = Column(String, primary_key=True, default=generate_uuid)
    entity_id = Column(String, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    node_id = Column(String, nullable=False, index=True)
    node_title = Column(String(500), nullable=True)
    mention_text = Column(String(500), nullable=True)
    start_page = Column(Integer, nullable=True)
    end_page = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_mentions_entity_doc", "entity_id", "document_id"),
        Index("ix_mentions_doc_node", "document_id", "node_id"),
    )


class GraphEdge(Base):
    """Relationships between entities, both within and across documents."""

    __tablename__ = "graph_edges"

    id = Column(String, primary_key=True, default=generate_uuid)
    source_entity_id = Column(String, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    source_document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    source_node_id = Column(String, nullable=True)
    target_entity_id = Column(String, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    target_document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    target_node_id = Column(String, nullable=True)
    relation_type = Column(String(100), nullable=False, index=True)
    scope = Column(String(20), nullable=False, default="intra")  # "intra" or "inter"
    weight = Column(Integer, default=1)
    confidence = Column(Integer, default=80)  # 0-100
    context_snippet = Column(String(500), nullable=True)

    __table_args__ = (
        Index("ix_edges_source_target", "source_entity_id", "target_entity_id"),
        Index("ix_edges_doc_scope", "source_document_id", "scope"),
        CheckConstraint("confidence >= 0 AND confidence <= 100", name="ck_edge_confidence"),
    )
