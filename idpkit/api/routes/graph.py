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
    limit: int = Query(50, ge=1, le=200),
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
    limit: int = Query(50, ge=1, le=200),
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
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get nodes+edges JSON for D3 visualization."""
    from idpkit.graph.queries import get_visualization_data

    data = await get_visualization_data(db, doc_id)
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

    data = await get_multi_doc_visualization_data(db, resolved_ids)
    return VisualizationData(
        nodes=[VisualizationNode(**n) for n in data["nodes"]],
        edges=[VisualizationEdge(**e) for e in data["edges"]],
    )


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
