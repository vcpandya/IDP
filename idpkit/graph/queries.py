"""Graph query functions — entity lookup, mentions, neighbors, cross-doc links."""

import logging

from sqlalchemy import select, and_, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from idpkit.db.models import Document
from .models import Entity, EntityMention, GraphEdge

logger = logging.getLogger(__name__)

# Default caps for query results to prevent memory issues.
_DEFAULT_LIMIT = 50
_MAX_IN_CLAUSE = 500


async def search_entities(
    db: AsyncSession,
    name: str | None = None,
    entity_type: str | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict]:
    """Search entities by name (partial match) and/or type, with mention counts."""
    stmt = (
        select(Entity, func.count(EntityMention.id).label("mention_count"))
        .outerjoin(EntityMention, Entity.id == EntityMention.entity_id)
        .group_by(Entity.id)
    )

    if name:
        search_term = name[:200].lower()
        stmt = stmt.where(func.lower(Entity.canonical_name).contains(search_term))
    if entity_type:
        stmt = stmt.where(Entity.entity_type == entity_type.upper()[:50])

    stmt = stmt.order_by(Entity.document_count.desc()).limit(limit)
    rows = (await db.execute(stmt)).all()
    results = []
    for entity, mc in rows:
        entity.mention_count = mc
        results.append(entity)
    return results


async def get_entity_detail(
    db: AsyncSession,
    entity_id: str,
) -> dict | None:
    """Get full entity details including mentions and edges."""
    entity = (await db.execute(
        select(Entity).where(Entity.id == entity_id)
    )).scalar_one_or_none()

    if not entity:
        return None

    mention_rows = (await db.execute(
        select(EntityMention, Document.filename, Document.format, Document.source_url)
        .join(Document, EntityMention.document_id == Document.id)
        .where(EntityMention.entity_id == entity_id)
        .limit(_DEFAULT_LIMIT)
    )).all()

    enriched_mentions = []
    for mention, doc_filename, doc_format, doc_source_url in mention_rows:
        mention.document_filename = doc_filename
        mention.document_format = doc_format
        mention.source_url = doc_source_url
        enriched_mentions.append(mention)

    SourceEntity = aliased(Entity)
    TargetEntity = aliased(Entity)
    edge_rows = (await db.execute(
        select(
            GraphEdge,
            SourceEntity.canonical_name.label("source_entity_name"),
            SourceEntity.entity_type.label("source_entity_type"),
            TargetEntity.canonical_name.label("target_entity_name"),
            TargetEntity.entity_type.label("target_entity_type"),
        )
        .outerjoin(SourceEntity, GraphEdge.source_entity_id == SourceEntity.id)
        .outerjoin(TargetEntity, GraphEdge.target_entity_id == TargetEntity.id)
        .where(
            or_(
                GraphEdge.source_entity_id == entity_id,
                GraphEdge.target_entity_id == entity_id,
            )
        )
        .limit(_DEFAULT_LIMIT)
    )).all()

    enriched_edges = []
    for edge, src_name, src_type, tgt_name, tgt_type in edge_rows:
        edge.source_entity_name = src_name
        edge.source_entity_type = src_type
        edge.target_entity_name = tgt_name
        edge.target_entity_type = tgt_type
        enriched_edges.append(edge)

    return {"entity": entity, "mentions": enriched_mentions, "edges": enriched_edges}


async def get_entity_mentions(
    db: AsyncSession,
    entity_id: str,
    limit: int = _DEFAULT_LIMIT,
) -> list[EntityMention]:
    """Get all tree nodes where an entity is mentioned."""
    rows = (await db.execute(
        select(EntityMention, Document.filename, Document.format, Document.source_url)
        .join(Document, EntityMention.document_id == Document.id)
        .where(EntityMention.entity_id == entity_id)
        .limit(min(limit, 200))
    )).all()

    enriched = []
    for mention, doc_filename, doc_format, doc_source_url in rows:
        mention.document_filename = doc_filename
        mention.document_format = doc_format
        mention.source_url = doc_source_url
        enriched.append(mention)
    return enriched


async def get_entity_neighbors(
    db: AsyncSession,
    entity_id: str,
    relation_type: str | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict]:
    """Get entities connected to the given entity via edges.

    Returns list of {"entity": Entity, "edge": GraphEdge} dicts.
    """
    # Outgoing edges
    stmt_out = select(GraphEdge, Entity).join(
        Entity, GraphEdge.target_entity_id == Entity.id
    ).where(GraphEdge.source_entity_id == entity_id)

    # Incoming edges
    stmt_in = select(GraphEdge, Entity).join(
        Entity, GraphEdge.source_entity_id == Entity.id
    ).where(GraphEdge.target_entity_id == entity_id)

    if relation_type:
        stmt_out = stmt_out.where(GraphEdge.relation_type == relation_type)
        stmt_in = stmt_in.where(GraphEdge.relation_type == relation_type)

    stmt_out = stmt_out.limit(limit)
    stmt_in = stmt_in.limit(limit)

    out_results = (await db.execute(stmt_out)).all()
    in_results = (await db.execute(stmt_in)).all()

    neighbors: list[dict] = []
    seen_ids: set[str] = set()

    for edge, entity in out_results:
        if entity.id not in seen_ids:
            neighbors.append({"entity": entity, "edge": edge})
            seen_ids.add(entity.id)

    for edge, entity in in_results:
        if entity.id not in seen_ids:
            neighbors.append({"entity": entity, "edge": edge})
            seen_ids.add(entity.id)

    return neighbors[:limit]


async def get_document_entities(
    db: AsyncSession,
    document_id: str,
    limit: int = 1000,
) -> list[Entity]:
    """Get all entities found in a specific document, with mention counts."""
    stmt = (
        select(Entity, func.count(EntityMention.id).label("mention_count"))
        .join(EntityMention, Entity.id == EntityMention.entity_id)
        .where(EntityMention.document_id == document_id)
        .group_by(Entity.id)
        .order_by(Entity.canonical_name)
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    results = []
    for entity, mc in rows:
        entity.mention_count = mc
        results.append(entity)
    return results


async def get_document_edges(
    db: AsyncSession,
    document_id: str,
    scope: str | None = None,
    limit: int = 2000,
) -> list[GraphEdge]:
    """Get all edges involving a specific document."""
    stmt = select(GraphEdge).where(
        or_(
            GraphEdge.source_document_id == document_id,
            GraphEdge.target_document_id == document_id,
        )
    )
    if scope:
        stmt = stmt.where(GraphEdge.scope == scope)

    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_cross_document_links(
    db: AsyncSession,
    document_id: str,
    limit: int = 20,
) -> list[dict]:
    """Find other documents linked to this one via shared entities.

    Returns list of {"document_id", "filename", "shared_entities", "edge_count"}.
    """
    # Get entity IDs mentioned in this document (capped)
    doc_entity_ids = (await db.execute(
        select(EntityMention.entity_id)
        .where(EntityMention.document_id == document_id)
        .distinct()
        .limit(_MAX_IN_CLAUSE)
    )).scalars().all()

    if not doc_entity_ids:
        return []

    doc_entity_ids = list(doc_entity_ids)

    # Find other documents mentioning the same entities, with counts
    stmt = (
        select(
            EntityMention.document_id,
            func.count(EntityMention.entity_id.distinct()).label("shared_count"),
        )
        .where(
            and_(
                EntityMention.entity_id.in_(doc_entity_ids),
                EntityMention.document_id != document_id,
            )
        )
        .group_by(EntityMention.document_id)
        .order_by(func.count(EntityMention.entity_id.distinct()).desc())
        .limit(min(limit, 50))
    )
    linked_docs = (await db.execute(stmt)).all()

    if not linked_docs:
        return []

    # Batch-load linked document filenames
    linked_doc_ids = [row[0] for row in linked_docs]
    docs_result = await db.execute(
        select(Document).where(Document.id.in_(linked_doc_ids))
    )
    doc_map = {d.id: d for d in docs_result.scalars().all()}

    # Count inter-doc edges
    edge_counts: dict[str, int] = {}
    edge_stmt = (
        select(GraphEdge.target_document_id, func.count())
        .where(
            and_(
                GraphEdge.scope == "inter",
                GraphEdge.source_document_id == document_id,
                GraphEdge.target_document_id.in_(linked_doc_ids),
            )
        )
        .group_by(GraphEdge.target_document_id)
    )
    for doc_id, count in (await db.execute(edge_stmt)).all():
        edge_counts[doc_id] = count

    results: list[dict] = []
    for linked_doc_id, shared_count in linked_docs:
        doc = doc_map.get(linked_doc_id)

        # Get shared entity details (capped)
        shared_stmt = (
            select(Entity)
            .join(EntityMention, Entity.id == EntityMention.entity_id)
            .where(
                and_(
                    EntityMention.document_id == linked_doc_id,
                    EntityMention.entity_id.in_(doc_entity_ids),
                )
            )
            .distinct()
            .limit(10)
        )
        shared_entities = list((await db.execute(shared_stmt)).scalars().all())

        results.append({
            "document_id": linked_doc_id,
            "filename": doc.filename if doc else "unknown",
            "shared_entities": shared_entities,
            "edge_count": edge_counts.get(linked_doc_id, 0),
        })

    return results


async def get_related_sections(
    db: AsyncSession,
    document_id: str,
    node_id: str,
    limit: int = 20,
) -> list[dict]:
    """Find sections sharing entities with the given node.

    Returns list of {"document_id", "node_id", "node_title", "shared_entities"}.
    """
    # Get entities mentioned in this node
    stmt = (
        select(EntityMention)
        .where(
            and_(
                EntityMention.document_id == document_id,
                EntityMention.node_id == node_id,
            )
        )
        .limit(100)
    )
    node_mentions = (await db.execute(stmt)).scalars().all()
    entity_ids = [m.entity_id for m in node_mentions]

    if not entity_ids:
        return []

    # Find other mentions of the same entities (excluding this node)
    stmt = (
        select(EntityMention)
        .where(
            and_(
                EntityMention.entity_id.in_(entity_ids[:_MAX_IN_CLAUSE]),
                ~and_(
                    EntityMention.document_id == document_id,
                    EntityMention.node_id == node_id,
                ),
            )
        )
        .limit(200)
    )
    other_mentions = (await db.execute(stmt)).scalars().all()

    # Group by (document_id, node_id)
    section_map: dict[tuple[str, str], dict] = {}
    for mention in other_mentions:
        key = (mention.document_id, mention.node_id)
        if key not in section_map:
            section_map[key] = {
                "document_id": mention.document_id,
                "node_id": mention.node_id,
                "node_title": mention.node_title,
                "shared_entity_ids": set(),
            }
        section_map[key]["shared_entity_ids"].add(mention.entity_id)

    # Convert sets to lists and sort by shared entity count
    results = []
    for section in section_map.values():
        section["shared_entity_count"] = len(section["shared_entity_ids"])
        section["shared_entity_ids"] = list(section["shared_entity_ids"])
        results.append(section)

    results.sort(key=lambda x: x["shared_entity_count"], reverse=True)
    return results[:limit]


async def get_document_graph_summary(
    db: AsyncSession,
    document_id: str,
) -> dict:
    """Get graph statistics for a document."""
    # Count entities
    entity_count = (await db.execute(
        select(func.count(EntityMention.entity_id.distinct()))
        .where(EntityMention.document_id == document_id)
    )).scalar() or 0

    # Count edges
    edge_count = (await db.execute(
        select(func.count())
        .select_from(GraphEdge)
        .where(
            or_(
                GraphEdge.source_document_id == document_id,
                GraphEdge.target_document_id == document_id,
            )
        )
    )).scalar() or 0

    # Entity type breakdown
    stmt = (
        select(Entity.entity_type, func.count())
        .join(EntityMention, Entity.id == EntityMention.entity_id)
        .where(EntityMention.document_id == document_id)
        .group_by(Entity.entity_type)
    )
    type_rows = (await db.execute(stmt)).all()
    entity_types = {row[0]: row[1] for row in type_rows}

    # Top entities by mention count
    stmt = (
        select(Entity, func.count(EntityMention.id).label("mention_count"))
        .join(EntityMention, Entity.id == EntityMention.entity_id)
        .where(EntityMention.document_id == document_id)
        .group_by(Entity.id)
        .order_by(func.count(EntityMention.id).desc())
        .limit(10)
    )
    top_rows = (await db.execute(stmt)).all()
    top_entities = [row[0] for row in top_rows]

    return {
        "document_id": document_id,
        "entity_count": entity_count,
        "edge_count": edge_count,
        "entity_types": entity_types,
        "top_entities": top_entities,
    }


async def get_visualization_data(
    db: AsyncSession,
    document_id: str,
    limit: int = 1000,
) -> dict:
    """Generate nodes+edges JSON for D3/visualization of a document's graph."""
    entities = await get_document_entities(db, document_id, limit=limit)
    edges = await get_document_edges(db, document_id, limit=limit * 3)

    vis_nodes = []
    entity_ids = set()
    for e in entities:
        vis_nodes.append({
            "id": e.id,
            "label": e.canonical_name,
            "type": e.entity_type,
            "document_id": e.first_document_id,
        })
        entity_ids.add(e.id)

    vis_edges = []
    for edge in edges:
        if edge.source_entity_id in entity_ids and edge.target_entity_id in entity_ids:
            vis_edges.append({
                "source": edge.source_entity_id,
                "target": edge.target_entity_id,
                "relation_type": edge.relation_type,
                "weight": edge.weight or 1,
            })

    return {"nodes": vis_nodes, "edges": vis_edges}


async def get_multi_doc_visualization_data(
    db: AsyncSession,
    document_ids: list[str],
    limit: int = 1000,
) -> dict:
    """Generate nodes+edges JSON for D3 visualization across multiple documents."""
    if not document_ids:
        return {"nodes": [], "edges": []}

    distinct_entity_ids_subq = (
        select(EntityMention.entity_id)
        .where(EntityMention.document_id.in_(document_ids))
        .distinct()
        .limit(limit)
        .subquery()
    )
    stmt_entities = (
        select(Entity)
        .where(Entity.id.in_(select(distinct_entity_ids_subq)))
    )
    entities = list((await db.execute(stmt_entities)).scalars().all())

    entity_ids = {e.id for e in entities}
    if not entity_ids:
        return {"nodes": [], "edges": []}

    stmt_edges = (
        select(GraphEdge)
        .where(
            or_(
                GraphEdge.source_document_id.in_(document_ids),
                GraphEdge.target_document_id.in_(document_ids),
            )
        )
        .limit(limit * 3)
    )
    edges = list((await db.execute(stmt_edges)).scalars().all())

    vis_nodes = []
    for e in entities:
        vis_nodes.append({
            "id": e.id,
            "label": e.canonical_name,
            "type": e.entity_type,
            "document_id": e.first_document_id,
        })

    vis_edges = []
    for edge in edges:
        if edge.source_entity_id in entity_ids and edge.target_entity_id in entity_ids:
            vis_edges.append({
                "source": edge.source_entity_id,
                "target": edge.target_entity_id,
                "relation_type": edge.relation_type,
                "weight": edge.weight or 1,
            })

    return {"nodes": vis_nodes, "edges": vis_edges}
