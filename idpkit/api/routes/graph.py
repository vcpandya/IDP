"""Graph API routes — entity queries, document graphs, visualization."""

import logging

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.api.deps import get_current_user, get_db, get_llm
from idpkit.core.llm import LLMClient
from idpkit.db.models import Document, User
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


# -------------------------------------------------------------------------
# Entity endpoints
# -------------------------------------------------------------------------


@router.get("/entity-types", response_model=list[str], summary="List entity types")
async def list_entity_types(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all distinct entity types currently in the database."""
    from sqlalchemy import distinct, select as sa_select
    from idpkit.graph.models import Entity as EntityModel
    stmt = sa_select(distinct(EntityModel.entity_type)).order_by(EntityModel.entity_type)
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
    from sqlalchemy import select

    doc = (await db.execute(
        select(Document).where(Document.id == doc_id)
    )).scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
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
    from sqlalchemy import select as sa_select
    from idpkit.db.models import Tag
    from idpkit.graph.queries import get_multi_doc_visualization_data

    resolved_ids: list[str] = []

    if tag_id:
        from sqlalchemy.orm import selectinload
        tag = (await db.execute(
            sa_select(Tag)
            .options(selectinload(Tag.documents))
            .where(Tag.id == tag_id, Tag.owner_id == user.id)
        )).scalar_one_or_none()
        if tag:
            resolved_ids = [d.id for d in tag.documents if d.status == "indexed"]

    if doc_ids:
        extra_ids = [did.strip() for did in doc_ids.split(",") if did.strip()]
        if extra_ids:
            owned_docs = (await db.execute(
                sa_select(Document.id).where(
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
    import io
    import json
    from fastapi.responses import Response
    from sqlalchemy import select as sa_select
    from idpkit.db.models import Tag
    from idpkit.graph.queries import get_multi_doc_visualization_data

    resolved_ids: list[str] = []

    if tag_id:
        from sqlalchemy.orm import selectinload
        tag = (await db.execute(
            sa_select(Tag)
            .options(selectinload(Tag.documents))
            .where(Tag.id == tag_id, Tag.owner_id == user.id)
        )).scalar_one_or_none()
        if tag:
            resolved_ids = [d.id for d in tag.documents if d.status == "indexed"]

    if doc_ids:
        extra_ids = [did.strip() for did in doc_ids.split(",") if did.strip()]
        owned_docs = (await db.execute(
            sa_select(Document.id).where(
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
    from sqlalchemy import select, func
    from idpkit.graph.builder import build_document_graph
    from idpkit.graph.linker import link_entities_across_documents
    from idpkit.graph.models import EntityMention

    unique_ids = list(dict.fromkeys(body.document_ids))

    docs = (await db.execute(
        select(Document).where(
            Document.id.in_(unique_ids),
            Document.owner_id == user.id,
        )
    )).scalars().all()

    doc_map = {d.id: d for d in docs}

    mention_counts = {}
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
    """Analyze the knowledge graph and generate hidden insights using LLM."""
    from sqlalchemy import select as sa_select, func, distinct
    from idpkit.graph.models import Entity as EntityModel, EntityMention, GraphEdge

    doc_ids = body.document_ids

    if doc_ids:
        owned = (await db.execute(
            sa_select(Document.id).where(
                Document.id.in_(doc_ids),
                Document.owner_id == user.id,
            )
        )).scalars().all()
        doc_ids = list(owned)

    if doc_ids:
        entity_ids_subq = (
            sa_select(EntityMention.entity_id)
            .where(EntityMention.document_id.in_(doc_ids))
            .distinct()
            .subquery()
        )
        entities = list((await db.execute(
            sa_select(EntityModel)
            .where(EntityModel.id.in_(sa_select(entity_ids_subq)))
            .order_by(EntityModel.document_count.desc())
        )).scalars().all())

        from sqlalchemy import or_
        edges = list((await db.execute(
            sa_select(GraphEdge)
            .where(or_(
                GraphEdge.source_document_id.in_(doc_ids),
                GraphEdge.target_document_id.in_(doc_ids),
            ))
        )).scalars().all())

        doc_rows = (await db.execute(
            sa_select(Document.id, Document.filename)
            .where(Document.id.in_(doc_ids))
        )).all()
        doc_names = {r[0]: r[1] for r in doc_rows}
    else:
        user_doc_ids_q = sa_select(Document.id).where(Document.owner_id == user.id)
        user_doc_ids = list((await db.execute(user_doc_ids_q)).scalars().all())

        if not user_doc_ids:
            return {"insights": "No documents found. Upload and index documents, then build their knowledge graphs to generate insights."}

        entity_ids_subq = (
            sa_select(EntityMention.entity_id)
            .where(EntityMention.document_id.in_(user_doc_ids))
            .distinct()
            .subquery()
        )
        entities = list((await db.execute(
            sa_select(EntityModel)
            .where(EntityModel.id.in_(sa_select(entity_ids_subq)))
            .order_by(EntityModel.document_count.desc())
        )).scalars().all())

        from sqlalchemy import or_
        edges = list((await db.execute(
            sa_select(GraphEdge)
            .where(or_(
                GraphEdge.source_document_id.in_(user_doc_ids),
                GraphEdge.target_document_id.in_(user_doc_ids),
            ))
        )).scalars().all())

        doc_rows = (await db.execute(
            sa_select(Document.id, Document.filename)
            .where(Document.id.in_(user_doc_ids))
        )).all()
        doc_names = {r[0]: r[1] for r in doc_rows}

    if not entities:
        return {"insights": "No entities found in the knowledge graph. Build graphs for your documents first to generate insights."}

    entity_map = {e.id: e for e in entities}
    entity_ids = set(entity_map.keys())

    type_counts: dict[str, int] = {}
    for e in entities:
        type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1

    top_entities = sorted(entities, key=lambda e: e.document_count or 1, reverse=True)[:30]

    relationships = []
    relation_type_counts: dict[str, int] = {}
    for edge in edges:
        src = entity_map.get(edge.source_entity_id)
        tgt = entity_map.get(edge.target_entity_id)
        if src and tgt and edge.relation_type != "co_occurrence":
            relationships.append({
                "source": src.canonical_name,
                "source_type": src.entity_type,
                "target": tgt.canonical_name,
                "target_type": tgt.entity_type,
                "relation": edge.relation_type,
                "context": edge.context_snippet or "",
            })
        rt = edge.relation_type
        relation_type_counts[rt] = relation_type_counts.get(rt, 0) + 1

    co_occurrences = []
    for edge in edges:
        if edge.relation_type == "co_occurrence":
            src = entity_map.get(edge.source_entity_id)
            tgt = entity_map.get(edge.target_entity_id)
            if src and tgt:
                co_occurrences.append(f"{src.canonical_name} ({src.entity_type}) <-> {tgt.canonical_name} ({tgt.entity_type})")

    multi_doc_entities = [e for e in entities if (e.document_count or 1) > 1]

    prompt_parts = [
        "You are an expert knowledge analyst. Analyze the following knowledge graph data and produce deep, actionable insights.",
        "",
        f"## Graph Overview",
        f"- Total entities: {len(entities)}",
        f"- Total relationships: {len(edges)}",
        f"- Entity types: {', '.join(f'{t} ({c})' for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))}",
        f"- Relationship types: {', '.join(f'{t} ({c})' for t, c in sorted(relation_type_counts.items(), key=lambda x: -x[1])[:15])}",
    ]

    if doc_names:
        prompt_parts.append(f"- Documents analyzed: {', '.join(doc_names.values())}")

    prompt_parts.append(f"\n## Top Entities (by cross-document presence)")
    for e in top_entities[:20]:
        desc = f" — {e.description[:100]}" if e.description else ""
        prompt_parts.append(f"- **{e.canonical_name}** ({e.entity_type}, appears in {e.document_count} docs){desc}")

    if relationships:
        prompt_parts.append(f"\n## Key Relationships (sample of {len(relationships)})")
        for r in relationships[:40]:
            ctx = f" [{r['context'][:80]}]" if r['context'] else ""
            prompt_parts.append(f"- {r['source']} ({r['source_type']}) --[{r['relation']}]--> {r['target']} ({r['target_type']}){ctx}")

    if co_occurrences:
        prompt_parts.append(f"\n## Co-occurrences (entities appearing together, sample of {len(co_occurrences)})")
        for c in co_occurrences[:30]:
            prompt_parts.append(f"- {c}")

    if multi_doc_entities:
        prompt_parts.append(f"\n## Cross-Document Entities ({len(multi_doc_entities)} entities span multiple documents)")
        for e in multi_doc_entities[:15]:
            prompt_parts.append(f"- {e.canonical_name} ({e.entity_type}) — in {e.document_count} documents")

    prompt_parts.append("""
## Your Task

Produce a structured analysis in Markdown with EXACTLY these sections:

### 🔍 Hidden Connections
Identify non-obvious relationships between entities that might be missed at first glance. Look for indirect links, shared contexts, and unexpected bridges between different entity types or topics.

### 💡 Key Insights
Highlight the most important patterns, trends, and takeaways from the knowledge graph. What are the central themes? What entities are most influential?

### 🌐 Cross-Document Patterns
If entities span multiple documents, explain what connects them and why this matters. Identify knowledge that only emerges when looking across documents together.

### ⚡ Interesting Facts & Tidbits
Surface surprising, counterintuitive, or particularly noteworthy details from the data. These are the "did you know?" moments.

### 🎯 Recommendations
Based on the graph structure, suggest what the user should explore further, which connections deserve deeper investigation, or what additional documents might enrich the knowledge base.

Be specific — reference actual entity names and relationships from the data. Avoid generic statements. Each section should have 3-5 bullet points.
""")

    prompt = "\n".join(prompt_parts)

    try:
        response = await llm.acomplete(prompt)
        insights_text = response.content
    except Exception as exc:
        logger.error("Insights LLM call failed: %s", exc)
        return {"insights": f"Failed to generate insights: {exc}"}

    return {
        "insights": insights_text,
        "stats": {
            "entities": len(entities),
            "edges": len(edges),
            "entity_types": len(type_counts),
            "relationship_types": len(relation_type_counts),
            "cross_doc_entities": len(multi_doc_entities),
        },
    }
