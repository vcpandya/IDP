"""Graph API routes — entity queries, document graphs, visualization."""

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import distinct, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from idpkit.api.deps import get_current_user, get_db, get_llm
from idpkit.core.llm import LLMClient
from idpkit.db.models import Document, Tag, User
from idpkit.graph.models import Entity as EntityModel, EntityMention, GraphEdge
from idpkit.graph.schemas import (
    CrossDocLink,
    DocumentGraphSummary,
    EdgeSchema,
    EntityDetailSchema,
    EntitySchema,
    MentionSchema,
    NeighborSchema,
    VisualizationData,
    VisualizationEdge,
    VisualizationNode,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


async def _verify_doc_ownership(
    db: AsyncSession, doc_id: str, user: User
) -> Document:
    doc = (
        await db.execute(
            select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
        )
    ).scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# -------------------------------------------------------------------------
# Entity endpoints
# -------------------------------------------------------------------------


@router.get("/entity-types", response_model=list[str], summary="List entity types")
async def list_entity_types(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all distinct entity types currently in the database."""
    stmt = select(distinct(EntityModel.entity_type)).order_by(EntityModel.entity_type)
    result = await db.execute(stmt)
    return [row[0] for row in result.all() if row[0]]


@router.get("/entities", response_model=list[EntitySchema], summary="Search entities")
async def search_entities(
    name: str | None = Query(None, description="Filter by name (partial match)"),
    entity_type: str | None = Query(None, description="Filter by entity type"),
    limit: int = Query(100, ge=1, le=5000),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Search entities by name and/or type."""
    from idpkit.graph.queries import search_entities as _search
    entities = await _search(db, name=name, entity_type=entity_type, limit=limit)
    return [EntitySchema.model_validate(e) for e in entities]


@router.get("/entities/{entity_id}", response_model=EntityDetailSchema, summary="Entity details")
async def get_entity(
    entity_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get full entity details including mentions and relationships."""
    from idpkit.graph.queries import get_entity_detail

    detail = await get_entity_detail(db, entity_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Entity not found")

    return EntityDetailSchema(
        entity=EntitySchema.model_validate(detail["entity"]),
        mentions=[MentionSchema.model_validate(m) for m in detail["mentions"]],
        edges=[EdgeSchema.model_validate(e) for e in detail["edges"]],
    )


@router.get(
    "/entities/{entity_id}/mentions",
    response_model=list[MentionSchema],
    summary="Entity mentions",
)
async def get_mentions(
    entity_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all tree nodes where an entity is mentioned."""
    from idpkit.graph.queries import get_entity_mentions

    mentions = await get_entity_mentions(db, entity_id)
    return [MentionSchema.model_validate(m) for m in mentions]


@router.get(
    "/entities/{entity_id}/neighbors",
    response_model=list[NeighborSchema],
    summary="Entity neighbors",
)
async def get_neighbors(
    entity_id: str,
    relation_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=5000),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get entities connected to the given entity via graph edges."""
    from idpkit.graph.queries import get_entity_neighbors

    neighbors = await get_entity_neighbors(
        db, entity_id, relation_type=relation_type, limit=limit
    )
    return [
        NeighborSchema(
            entity=EntitySchema.model_validate(n["entity"]),
            edge=EdgeSchema.model_validate(n["edge"]),
        )
        for n in neighbors
    ]


# -------------------------------------------------------------------------
# Document-level graph endpoints
# -------------------------------------------------------------------------


@router.get(
    "/documents/{doc_id}/entities",
    response_model=list[EntitySchema],
    summary="Document entities",
)
async def get_doc_entities(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all entities found in a document."""
    await _verify_doc_ownership(db, doc_id, user)
    from idpkit.graph.queries import get_document_entities
    entities = await get_document_entities(db, doc_id)
    return [EntitySchema.model_validate(e) for e in entities]


@router.get(
    "/documents/{doc_id}/links",
    response_model=list[CrossDocLink],
    summary="Cross-document links",
)
async def get_doc_links(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Find other documents linked via shared entities."""
    await _verify_doc_ownership(db, doc_id, user)
    from idpkit.graph.queries import get_cross_document_links
    links = await get_cross_document_links(db, doc_id)
    return [
        CrossDocLink(
            linked_document_id=link["document_id"],
            linked_document_filename=link["filename"],
            shared_entities=[EntitySchema.model_validate(e) for e in link["shared_entities"]],
            edge_count=link["edge_count"],
        )
        for link in links
    ]


@router.get(
    "/documents/{doc_id}/summary",
    response_model=DocumentGraphSummary,
    summary="Document graph summary",
)
async def get_doc_graph_summary(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get graph statistics for a document."""
    await _verify_doc_ownership(db, doc_id, user)
    from idpkit.graph.queries import get_document_graph_summary
    summary = await get_document_graph_summary(db, doc_id)
    return DocumentGraphSummary(
        document_id=doc_id,
        entity_count=summary["entity_count"],
        edge_count=summary["edge_count"],
        entity_types=summary["entity_types"],
        top_entities=[EntitySchema.model_validate(e) for e in summary["top_entities"]],
    )


@router.post(
    "/documents/{doc_id}/build",
    summary="Build graph retroactively",
)
async def build_doc_graph(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Trigger graph building on an already-indexed document."""
    doc = await _verify_doc_ownership(db, doc_id, user)
    if not doc.tree_index:
        raise HTTPException(status_code=400, detail="Document has no tree index")

    from idpkit.graph.builder import build_document_graph
    from idpkit.graph.linker import link_entities_across_documents

    ti = doc.tree_index
    if isinstance(ti, dict) and "structure" in ti:
        tree_index = ti
    else:
        tree_index = {"structure": ti}
    build_result = await build_document_graph(doc_id, tree_index, llm, db)
    link_result = await link_entities_across_documents(doc_id, db, llm)

    return {
        "document_id": doc_id,
        "build": build_result,
        "links": link_result,
    }


@router.get(
    "/documents/{doc_id}/visualization",
    response_model=VisualizationData,
    summary="Graph visualization data",
)
async def get_doc_visualization(
    doc_id: str,
    limit: int = Query(1000, ge=1, le=10000, description="Max entities to include"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get nodes+edges JSON for D3 visualization."""
    await _verify_doc_ownership(db, doc_id, user)
    from idpkit.graph.queries import get_visualization_data
    data = await get_visualization_data(db, doc_id, limit=limit)
    return VisualizationData(
        nodes=[VisualizationNode(**n) for n in data["nodes"]],
        edges=[VisualizationEdge(**e) for e in data["edges"]],
    )


@router.get(
    "/visualization",
    response_model=VisualizationData,
    summary="Multi-document graph visualization",
)
async def get_multi_doc_visualization(
    doc_ids: Optional[str] = Query(None, description="Comma-separated document IDs"),
    tag_id: Optional[str] = Query(None, description="Tag ID to load all tagged documents"),
    limit: int = Query(1000, ge=1, le=10000, description="Max entities to include"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get visualization data across multiple documents or a tag group."""
    from idpkit.graph.queries import get_multi_doc_visualization_data

    resolved_ids: list[str] = []

    if tag_id:
        tag = (await db.execute(
            select(Tag)
            .options(selectinload(Tag.documents))
            .where(Tag.id == tag_id, Tag.owner_id == user.id)
        )).scalar_one_or_none()
        if tag:
            resolved_ids = [d.id for d in tag.documents if d.status == "indexed"]

    if doc_ids:
        extra_ids = [did.strip() for did in doc_ids.split(",") if did.strip()]
        if extra_ids:
            owned_docs = (await db.execute(
                select(Document.id).where(
                    Document.id.in_(extra_ids),
                    Document.owner_id == user.id,
                    Document.status == "indexed",
                )
            )).scalars().all()
            owned_set = set(owned_docs)
            for did in extra_ids:
                if did in owned_set and did not in resolved_ids:
                    resolved_ids.append(did)

    if not resolved_ids:
        return VisualizationData(nodes=[], edges=[])

    data = await get_multi_doc_visualization_data(db, resolved_ids, limit=limit)
    return VisualizationData(
        nodes=[VisualizationNode(**n) for n in data["nodes"]],
        edges=[VisualizationEdge(**e) for e in data["edges"]],
    )


@router.get(
    "/export",
    summary="Export full knowledge graph data (no view limit)",
)
async def export_full_graph(
    doc_ids: Optional[str] = Query(None, description="Comma-separated document IDs"),
    tag_id: Optional[str] = Query(None, description="Tag ID"),
    format: str = Query("json", description="Export format: json, csv_entities, csv_relationships"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export the complete knowledge graph for selected documents without view limits."""
    from idpkit.graph.queries import get_multi_doc_visualization_data

    resolved_ids: list[str] = []

    if tag_id:
        tag = (await db.execute(
            select(Tag)
            .options(selectinload(Tag.documents))
            .where(Tag.id == tag_id, Tag.owner_id == user.id)
        )).scalar_one_or_none()
        if tag:
            resolved_ids = [d.id for d in tag.documents if d.status == "indexed"]

    if doc_ids:
        extra_ids = [did.strip() for did in doc_ids.split(",") if did.strip()]
        owned_docs = (await db.execute(
            select(Document.id).where(
                Document.id.in_(extra_ids),
                Document.owner_id == user.id,
                Document.status == "indexed",
            )
        )).scalars().all()
        owned_set = set(owned_docs)
        for did in extra_ids:
            if did in owned_set and did not in resolved_ids:
                resolved_ids.append(did)

    if not resolved_ids:
        if format == "json":
            return Response(
                content=json.dumps({"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": 'attachment; filename="knowledge-graph-full.json"'},
            )
        return Response(content="", media_type="text/csv")

    data = await get_multi_doc_visualization_data(db, resolved_ids, limit=100000)
    nodes = data["nodes"]
    edges = data["edges"]

    if format == "csv_entities":
        rows = ["name,type,mention_count,document_ids"]
        for n in nodes:
            doc_ids_str = ";".join(n.get("document_ids", []) if isinstance(n.get("document_ids"), list) else [])
            name = _csv_escape(n.get("label") or n.get("name", ""))
            ntype = _csv_escape(n.get("type") or n.get("entity_type", ""))
            rows.append(f"{name},{ntype},{n.get('mention_count', 0)},{_csv_escape(doc_ids_str)}")
        return Response(
            content="\n".join(rows),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="knowledge-graph-entities.csv"'},
        )
    elif format == "csv_relationships":
        node_map = {n["id"]: n.get("label") or n.get("name", n["id"]) for n in nodes}
        rows = ["source_name,relation,target_name,weight"]
        for e in edges:
            src = _csv_escape(node_map.get(e.get("source", ""), e.get("source", "")))
            rel = _csv_escape(e.get("relation_type", ""))
            tgt = _csv_escape(node_map.get(e.get("target", ""), e.get("target", "")))
            w = e.get("weight", 1)
            rows.append(f"{src},{rel},{tgt},{w}")
        return Response(
            content="\n".join(rows),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="knowledge-graph-relationships.csv"'},
        )
    else:
        export = {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }
        return Response(
            content=json.dumps(export, indent=2, default=str),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="knowledge-graph-full.json"'},
        )


def _csv_escape(val) -> str:
    s = str(val) if val is not None else ""
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


class InsightsRequest(BaseModel):
    document_ids: List[str] = []


class InsightsFollowupRequest(BaseModel):
    question: str
    document_ids: List[str] = []
    previous_context: str = ""


class SmartFocusRequest(BaseModel):
    prompt: str
    document_ids: List[str] = []
    tag_id: Optional[str] = None


class BulkBuildRequest(BaseModel):
    document_ids: List[str]


@router.post(
    "/build-bulk",
    summary="Index (if needed) and build graphs for multiple documents",
)
async def build_bulk_graphs(
    body: BulkBuildRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    from idpkit.graph.builder import build_document_graph
    from idpkit.graph.linker import link_entities_across_documents

    unique_ids = list(dict.fromkeys(body.document_ids))

    docs = (await db.execute(
        select(Document).where(
            Document.id.in_(unique_ids),
            Document.owner_id == user.id,
        )
    )).scalars().all()

    doc_map = {d.id: d for d in docs}

    mention_counts: dict[str, int] = {}
    if docs:
        rows = (await db.execute(
            select(EntityMention.document_id, func.count(EntityMention.id))
            .where(EntityMention.document_id.in_([d.id for d in docs]))
            .group_by(EntityMention.document_id)
        )).all()
        mention_counts = {r[0]: r[1] for r in rows}

    indexed = 0
    built = 0
    skipped = 0
    failed = 0
    results = []

    for doc_id in unique_ids:
        doc = doc_map.get(doc_id)
        if not doc:
            results.append({"document_id": doc_id, "status": "not_found"})
            failed += 1
            continue
        if not doc.file_path:
            results.append({"document_id": doc_id, "status": "no_file", "filename": doc.filename})
            skipped += 1
            continue

        if not doc.tree_index:
            try:
                logger.info("Bulk build: indexing %s (%s) first", doc_id, doc.filename)
                from idpkit.engine.page_index import build_tree_index
                from idpkit.api.deps import get_storage as _get_storage

                storage = _get_storage()
                tree_result = await build_tree_index(
                    file_key=doc.file_path,
                    storage=storage,
                    llm=llm,
                )

                doc.tree_index = tree_result.get("structure", [])
                doc.description = tree_result.get("doc_description")
                doc.status = "indexed"
                db.add(doc)
                await db.commit()
                await db.refresh(doc)
                indexed += 1
                logger.info("Bulk build: indexed %s (%s)", doc_id, doc.filename)
            except Exception as e:
                logger.error("Bulk build: indexing failed for %s: %s", doc_id, e)
                results.append({"document_id": doc_id, "status": "index_failed", "filename": doc.filename, "error": str(e)})
                failed += 1
                continue

        if mention_counts.get(doc_id, 0) > 0:
            results.append({"document_id": doc_id, "status": "already_built", "filename": doc.filename})
            skipped += 1
            continue

        try:
            ti = doc.tree_index
            if isinstance(ti, dict) and "structure" in ti:
                tree_index = ti
            else:
                tree_index = {"structure": ti}
            build_result = await build_document_graph(doc_id, tree_index, llm, db)
            await link_entities_across_documents(doc_id, db, llm)
            results.append({
                "document_id": doc_id,
                "status": "built",
                "filename": doc.filename,
                "entities": build_result.get("entities_created", 0),
                "edges": build_result.get("edges_created", 0),
            })
            built += 1
            logger.info("Bulk graph build: built graph for %s (%s)", doc_id, doc.filename)
        except Exception as e:
            logger.error("Bulk graph build failed for %s: %s", doc_id, e)
            results.append({"document_id": doc_id, "status": "error", "filename": doc.filename, "error": str(e)})
            failed += 1

    return {
        "total": len(unique_ids),
        "indexed": indexed,
        "built": built,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }


async def _load_graph_data(db: AsyncSession, user: User, doc_ids: list[str]):
    """Shared helper to load entities, edges, and doc info for insights/followup."""
    if doc_ids:
        owned = (await db.execute(
            select(Document.id).where(
                Document.id.in_(doc_ids),
                Document.owner_id == user.id,
            )
        )).scalars().all()
        doc_ids = list(owned)
    else:
        doc_ids = list((await db.execute(
            select(Document.id).where(Document.owner_id == user.id)
        )).scalars().all())

    if not doc_ids:
        return None

    entity_ids_subq = (
        select(EntityMention.entity_id)
        .where(EntityMention.document_id.in_(doc_ids))
        .distinct()
        .subquery()
    )
    entities = list((await db.execute(
        select(EntityModel)
        .where(EntityModel.id.in_(select(entity_ids_subq)))
        .order_by(EntityModel.document_count.desc())
    )).scalars().all())

    edges = list((await db.execute(
        select(GraphEdge)
        .where(or_(
            GraphEdge.source_document_id.in_(doc_ids),
            GraphEdge.target_document_id.in_(doc_ids),
        ))
    )).scalars().all())

    doc_rows = (await db.execute(
        select(Document.id, Document.filename, Document.format, Document.source_url)
        .where(Document.id.in_(doc_ids))
    )).all()

    return {
        "doc_ids": doc_ids,
        "entities": entities,
        "edges": edges,
        "doc_rows": doc_rows,
    }


def _build_graph_context_text(entities, edges, doc_rows):
    """Build the LLM context text from graph data."""
    entity_map = {e.id: e for e in entities}
    doc_names = {r[0]: r[1] for r in doc_rows}
    doc_formats = {r[0]: (r[2] or "pdf") for r in doc_rows}

    type_counts: dict[str, int] = {}
    for e in entities:
        type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1

    relation_type_counts: dict[str, int] = {}
    scope_counts = {"intra": 0, "inter": 0}
    confidence_values: list[int] = []
    relationships = []
    co_occurrences = []

    for edge in edges:
        rt = edge.relation_type
        relation_type_counts[rt] = relation_type_counts.get(rt, 0) + 1
        scope_counts[edge.scope or "intra"] = scope_counts.get(edge.scope or "intra", 0) + 1
        if edge.confidence:
            confidence_values.append(edge.confidence)

        src = entity_map.get(edge.source_entity_id)
        tgt = entity_map.get(edge.target_entity_id)
        if src and tgt:
            if edge.relation_type != "co_occurrence":
                relationships.append({
                    "source": src.canonical_name,
                    "source_type": src.entity_type,
                    "target": tgt.canonical_name,
                    "target_type": tgt.entity_type,
                    "relation": edge.relation_type,
                    "context": edge.context_snippet or "",
                    "scope": edge.scope or "intra",
                })
            else:
                co_occurrences.append(
                    f"{src.canonical_name} ({src.entity_type}) <-> {tgt.canonical_name} ({tgt.entity_type})"
                )

    multi_doc_entities = [e for e in entities if (e.document_count or 1) > 1]
    top_entities = sorted(entities, key=lambda e: e.document_count or 1, reverse=True)[:30]

    source_type_counts: dict[str, int] = {}
    for fmt in doc_formats.values():
        source_type_counts[fmt] = source_type_counts.get(fmt, 0) + 1

    conf_min = min(confidence_values) if confidence_values else 0
    conf_max = max(confidence_values) if confidence_values else 0
    conf_avg = round(sum(confidence_values) / len(confidence_values)) if confidence_values else 0

    parts = [
        "## Graph Overview",
        f"- Total entities: {len(entities)}",
        f"- Total relationships: {len(edges)}",
        f"- Entity types: {', '.join(f'{t} ({c})' for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))}",
        f"- Relationship types: {', '.join(f'{t} ({c})' for t, c in sorted(relation_type_counts.items(), key=lambda x: -x[1])[:15])}",
        f"- Scope: {scope_counts.get('intra', 0)} intra-document, {scope_counts.get('inter', 0)} cross-document edges",
        f"- Confidence: min={conf_min}, avg={conf_avg}, max={conf_max}",
        f"- Source types: {', '.join(f'{t} ({c})' for t, c in source_type_counts.items())}",
    ]

    if doc_names:
        parts.append(f"- Documents analyzed: {', '.join(doc_names.values())}")

    parts.append("\n## Top Entities (by cross-document presence)")
    for e in top_entities[:20]:
        desc = f" — {e.description[:100]}" if e.description else ""
        parts.append(f"- {e.canonical_name} ({e.entity_type}, in {e.document_count} docs){desc}")

    if relationships:
        parts.append(f"\n## Key Relationships (sample of {min(len(relationships), 40)})")
        for r in relationships[:40]:
            ctx = f" [{r['context'][:80]}]" if r['context'] else ""
            scope_tag = " [cross-doc]" if r['scope'] == "inter" else ""
            parts.append(f"- {r['source']} ({r['source_type']}) --[{r['relation']}]--> {r['target']} ({r['target_type']}){ctx}{scope_tag}")

    if co_occurrences:
        parts.append(f"\n## Co-occurrences (sample of {min(len(co_occurrences), 30)})")
        for c in co_occurrences[:30]:
            parts.append(f"- {c}")

    if multi_doc_entities:
        parts.append(f"\n## Cross-Document Entities ({len(multi_doc_entities)} entities span multiple docs)")
        for e in multi_doc_entities[:15]:
            parts.append(f"- {e.canonical_name} ({e.entity_type}) — in {e.document_count} documents")

    analytics = {
        "type_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "relation_distribution": dict(sorted(relation_type_counts.items(), key=lambda x: -x[1])),
        "scope_breakdown": scope_counts,
        "confidence_stats": {"min": conf_min, "max": conf_max, "avg": conf_avg},
        "source_types": source_type_counts,
        "top_entities": [
            {"name": e.canonical_name, "type": e.entity_type, "doc_count": e.document_count or 1}
            for e in top_entities[:10]
        ],
        "cross_doc_count": len(multi_doc_entities),
    }

    return "\n".join(parts), analytics, type_counts, relation_type_counts, multi_doc_entities


@router.post(
    "/insights",
    summary="Generate AI insights from the knowledge graph",
)
async def generate_insights(
    body: InsightsRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Analyze the knowledge graph and generate structured, multi-dimensional insights."""
    data = await _load_graph_data(db, user, body.document_ids)
    if not data:
        return {"insights": {"overview": "No documents found. Upload and index documents, then build their knowledge graphs to generate insights."}, "analytics": {}, "stats": {}}

    entities = data["entities"]
    edges = data["edges"]
    doc_rows = data["doc_rows"]

    if not entities:
        return {"insights": {"overview": "No entities found in the knowledge graph. Build graphs for your documents first."}, "analytics": {}, "stats": {}}

    context_text, analytics, type_counts, relation_type_counts, multi_doc_entities = _build_graph_context_text(
        entities, edges, doc_rows
    )

    prompt = f"""You are an expert knowledge analyst. Analyze the following knowledge graph data and produce deep, structured insights.

{context_text}

## Your Task

Return a JSON object (no markdown fences) with EXACTLY these keys:

{{
  "overview": "A 2-3 sentence executive summary of the most important findings across the entire knowledge graph.",
  "hidden_connections": [
    {{ "finding": "Brief title of the connection", "entities": ["Entity A", "Entity B"], "explanation": "2-3 sentences explaining the non-obvious link and why it matters." }}
  ],
  "cross_document_patterns": [
    {{ "finding": "Brief title", "documents": ["doc name 1", "doc name 2"], "entities": ["Entity X"], "explanation": "What connects these documents and why it matters." }}
  ],
  "temporal_dimensional": [
    {{ "finding": "Brief title", "dimension": "type_distribution|scope|confidence|source_type", "explanation": "Analysis of patterns across this dimension." }}
  ],
  "knowledge_gaps": [
    {{ "gap": "What's missing", "affected_area": "Which part of the graph is affected", "suggestion": "What to add", "search_query": "A specific web search query to fill this gap" }}
  ],
  "cross_source_triangulation": [
    {{ "claim": "A finding confirmed by multiple sources", "sources_confirming": ["source 1", "source 2"], "confidence_note": "How strong is this finding based on multi-source evidence" }}
  ],
  "recommendations": [
    {{ "suggestion": "What to do", "reason": "Why", "chat_prompt": "A ready-to-use prompt the user could ask an AI assistant about their documents" }}
  ],
  "suggested_questions": ["Question 1 the user might want to ask next?", "Question 2?", "Question 3?"]
}}

Rules:
- Each array should have 3-5 items (fewer if insufficient data).
- Reference actual entity names from the data. Be specific, not generic.
- For knowledge_gaps, think about what topics or connections are suspiciously absent.
- For cross_source_triangulation, highlight when the same entity or claim appears across different document types (PDF, YouTube, etc.).
- For temporal_dimensional, analyze patterns by entity type distribution, intra vs inter-document scope, confidence levels, and source types.
- For recommendations, include chat_prompt that starts with "Analyze..." or "Compare..." — something actionable.
- Return ONLY valid JSON. No markdown code fences."""

    try:
        response = await llm.acomplete(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        insights = json.loads(raw)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Insights LLM call failed or returned invalid JSON: %s", exc)
        fallback_text = ""
        if hasattr(exc, "__context__") or isinstance(exc, json.JSONDecodeError):
            try:
                response_text = response.content if 'response' in dir() else ""
                fallback_text = response_text
            except Exception:
                fallback_text = ""
        if not fallback_text:
            try:
                resp = await llm.acomplete(prompt)
                fallback_text = resp.content
            except Exception:
                fallback_text = f"Failed to generate insights: {exc}"
        insights = {"overview": fallback_text}

    if not isinstance(insights, dict):
        insights = {"overview": str(insights)}

    for key in ["hidden_connections", "cross_document_patterns", "temporal_dimensional",
                 "knowledge_gaps", "cross_source_triangulation", "recommendations", "suggested_questions"]:
        if key not in insights:
            insights[key] = []

    if "overview" not in insights:
        insights["overview"] = ""

    return {
        "insights": insights,
        "analytics": analytics,
        "stats": {
            "entities": len(entities),
            "edges": len(edges),
            "entity_types": len(type_counts),
            "relationship_types": len(relation_type_counts),
            "cross_doc_entities": len(multi_doc_entities),
        },
        "suggested_questions": insights.get("suggested_questions", []),
    }


@router.post(
    "/insights/followup",
    summary="Ask a follow-up question about knowledge graph insights",
)
async def insights_followup(
    body: InsightsFollowupRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Answer a follow-up question using graph data context and previous insights."""
    data = await _load_graph_data(db, user, body.document_ids)
    if not data:
        return {"answer": "No graph data available to answer this question.", "suggested_questions": []}

    entities = data["entities"]
    edges = data["edges"]
    doc_rows = data["doc_rows"]

    if not entities:
        return {"answer": "No entities in the graph to analyze.", "suggested_questions": []}

    context_text, _, _, _, _ = _build_graph_context_text(entities, edges, doc_rows)
    context_text_truncated = context_text[:6000]

    prompt = f"""You are a knowledge graph analyst helping a user explore their document knowledge base.

## Graph Data Summary
{context_text_truncated}

## Previous Insights Context
{body.previous_context[:3000] if body.previous_context else "No previous context."}

## User's Question
{body.question}

Answer the user's question thoroughly based on the graph data. Be specific — reference actual entity names and relationships. If the question asks about something not in the data, say so clearly.

After your answer, suggest 2-3 natural follow-up questions the user might ask next.

Return a JSON object (no markdown fences):
{{
  "answer": "Your detailed answer in markdown format...",
  "suggested_questions": ["Follow-up question 1?", "Follow-up question 2?"]
}}"""

    try:
        response = await llm.acomplete(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        return {
            "answer": result.get("answer", response.content),
            "suggested_questions": result.get("suggested_questions", []),
        }
    except (json.JSONDecodeError, Exception):
        try:
            return {"answer": response.content, "suggested_questions": []}
        except Exception as exc:
            logger.error("Insights followup failed: %s", exc)
            return {"answer": f"Failed to process your question: {exc}", "suggested_questions": []}


@router.post(
    "/smart-focus",
    summary="AI-powered Smart Focus for knowledge graph exploration",
)
async def smart_focus(
    body: SmartFocusRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Use AI to identify which entities are most relevant to a natural language prompt."""
    doc_ids = list(body.document_ids)

    if body.tag_id:
        tag = (await db.execute(
            select(Tag)
            .options(selectinload(Tag.documents))
            .where(Tag.id == body.tag_id, Tag.owner_id == user.id)
        )).scalar_one_or_none()
        if tag:
            for d in tag.documents:
                if d.status == "indexed" and d.id not in doc_ids:
                    doc_ids.append(d.id)

    if doc_ids:
        owned = (await db.execute(
            select(Document.id).where(
                Document.id.in_(doc_ids),
                Document.owner_id == user.id,
            )
        )).scalars().all()
        doc_ids = list(owned)

    if doc_ids:
        entity_ids_subq = (
            select(EntityMention.entity_id)
            .where(EntityMention.document_id.in_(doc_ids))
            .distinct()
            .subquery()
        )
        entities = list((await db.execute(
            select(EntityModel)
            .where(EntityModel.id.in_(select(entity_ids_subq)))
            .order_by(EntityModel.document_count.desc())
        )).scalars().all())
    else:
        user_doc_ids = list((await db.execute(
            select(Document.id).where(Document.owner_id == user.id)
        )).scalars().all())
        if not user_doc_ids:
            return {"focus_entity_ids": [], "summary": "No documents found."}
        entity_ids_subq = (
            select(EntityMention.entity_id)
            .where(EntityMention.document_id.in_(user_doc_ids))
            .distinct()
            .subquery()
        )
        entities = list((await db.execute(
            select(EntityModel)
            .where(EntityModel.id.in_(select(entity_ids_subq)))
            .order_by(EntityModel.document_count.desc())
        )).scalars().all())

    if not entities:
        return {"focus_entity_ids": [], "summary": "No entities found in the knowledge graph."}

    name_to_id: dict[str, str] = {}
    entity_list_parts = []
    for idx, e in enumerate(entities[:200]):
        desc = f" — {e.description[:80]}" if e.description else ""
        entity_list_parts.append(f"- #{idx} | {e.canonical_name} ({e.entity_type}){desc}")
        name_to_id[e.canonical_name.lower()] = e.id
    entity_list_text = "\n".join(entity_list_parts)

    prompt = f"""You are a knowledge graph analyst. The user wants to focus on specific aspects of their knowledge graph.

User's focus prompt: "{body.prompt}"

Here are the entities in the graph (each prefixed with a numeric index #N):
{entity_list_text}

Your task:
1. Identify which entities are most relevant to the user's focus prompt.
2. Return ONLY a JSON object with exactly these keys:
   - "indices": an array of the numeric index numbers (integers) of relevant entities (up to 30)
   - "names": an array of the exact entity name strings of relevant entities (up to 30)
   - "summary": a brief 1-2 sentence description of what you found

Return ONLY valid JSON, no markdown fences, no explanation outside the JSON."""

    try:
        response = await llm.acomplete(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        summary = result.get("summary", "")

        focus_ids: list[str] = []
        seen_ids: set[str] = set()
        ent_list = entities[:200]

        for idx in result.get("indices", []):
            try:
                i = int(idx)
                if 0 <= i < len(ent_list) and ent_list[i].id not in seen_ids:
                    focus_ids.append(ent_list[i].id)
                    seen_ids.add(ent_list[i].id)
            except (ValueError, TypeError):
                pass

        for name in result.get("names", []):
            if not isinstance(name, str):
                continue
            nl = name.lower()
            if nl in name_to_id and name_to_id[nl] not in seen_ids:
                focus_ids.append(name_to_id[nl])
                seen_ids.add(name_to_id[nl])
                continue
            for ent_name, ent_id in name_to_id.items():
                if ent_id not in seen_ids and (nl in ent_name or ent_name in nl):
                    focus_ids.append(ent_id)
                    seen_ids.add(ent_id)
                    break

        old_ids = result.get("ids", [])
        valid_ids = {e.id for e in entities}
        for fid in old_ids:
            if fid in valid_ids and fid not in seen_ids:
                focus_ids.append(fid)
                seen_ids.add(fid)

        return {"focus_entity_ids": focus_ids, "summary": summary}
    except Exception as exc:
        logger.error("Smart Focus LLM call failed: %s", exc)
        return {"focus_entity_ids": [], "summary": f"Failed to analyze: {exc}"}
