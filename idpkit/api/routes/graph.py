"""Graph API routes — entity queries, document graphs, visualization."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
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

    tree_index = {"structure": doc.tree_index}
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
