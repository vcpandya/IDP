"""Smart metadata API — categories, facets, filtering, graph, and (re)processing.

Powers the Document Map: browse category-aware facets, combine them to pre-filter
a set of documents, visualise them as a document-centric graph, and reprocess
existing documents to (re)extract their metadata.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.api.deps import get_current_user, get_db, get_llm, limiter
from idpkit.core.llm import LLMClient
from idpkit.db.models import Document, User
from idpkit.metadata import categories as cat_registry
from idpkit.metadata import queries as md_queries
from idpkit.metadata.extractor import profile_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/metadata", tags=["metadata"])

# LLM-backed extraction is expensive; cap how many documents one bulk request
# may process so a single caller cannot monopolise the LLM client / DB pool.
MAX_BULK_DOCS = 200


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class FacetCriterion(BaseModel):
    key: str
    value_norm: str | None = None
    value: str | None = None


class FilterRequest(BaseModel):
    criteria: list[FacetCriterion] = []
    match: Literal["all", "any"] = "all"


class ExtractBulkRequest(BaseModel):
    document_ids: list[str] | None = None
    # Used only when document_ids is omitted.
    scope: Literal["missing", "all"] = "missing"


async def _owned_doc(db: AsyncSession, doc_id: str, user: User) -> Document:
    doc = (
        await db.execute(
            select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
        )
    ).scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ---------------------------------------------------------------------------
# Registry + read endpoints
# ---------------------------------------------------------------------------


@router.get("/categories", summary="List document categories and their schemas")
async def list_categories(user: User = Depends(get_current_user)):
    return cat_registry.list_categories()


@router.get("/stats", summary="Smart-metadata coverage stats")
async def stats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await md_queries.get_stats(db, user.id)


@router.get("/facets", summary="Aggregated facets grouped by field")
async def get_facets(
    category: str | None = Query(None),
    key: str | None = Query(None),
    search: str | None = Query(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await md_queries.get_facets(
        db, user.id, category=category, key=key, search=search
    )


@router.post("/filter", summary="Documents matching combined facet criteria")
async def filter_documents(
    req: FilterRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    criteria = [c.model_dump() for c in req.criteria]
    return await md_queries.filter_documents(
        db, user.id, criteria, match=req.match
    )


@router.post("/graph", summary="Document-centric facet graph for a selection")
async def facet_graph(
    req: FilterRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    criteria = [c.model_dump() for c in req.criteria]
    return await md_queries.build_facet_graph(
        db, user.id, criteria, match=req.match
    )


@router.get("/documents/{doc_id}", summary="A document's category + facets")
async def document_metadata(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc = await _owned_doc(db, doc_id, user)
    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "category": doc.doc_category,
        "confidence": doc.doc_category_confidence,
        "facets": await md_queries.get_document_facets(db, doc_id, owner_id=user.id),
    }


# ---------------------------------------------------------------------------
# (Re)processing endpoints
# ---------------------------------------------------------------------------


@router.post("/documents/{doc_id}/extract", summary="Extract metadata for one document")
@limiter.limit("30/minute")
async def extract_document(
    request: Request,
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    doc = await _owned_doc(db, doc_id, user)
    result = await profile_document(db, llm, doc)
    return result


@router.post("/extract-bulk", summary="Extract metadata for many documents")
@limiter.limit("6/minute")
async def extract_bulk(
    request: Request,
    req: ExtractBulkRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    if req.document_ids and len(req.document_ids) > MAX_BULK_DOCS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many documents; limit is {MAX_BULK_DOCS} per request.",
        )

    # Resolve the working set (capped so an "all" run can't process unbounded docs).
    if req.document_ids:
        stmt = select(Document).where(
            Document.id.in_(req.document_ids), Document.owner_id == user.id
        )
    else:
        stmt = select(Document).where(Document.owner_id == user.id)
        if req.scope == "missing":
            stmt = stmt.where(Document.doc_category.is_(None))
    stmt = stmt.limit(MAX_BULK_DOCS)
    docs = (await db.execute(stmt)).scalars().all()

    processed, failed, skipped = 0, 0, 0
    for doc in docs:
        if not doc.tree_index and not doc.description:
            skipped += 1
            continue
        try:
            await profile_document(db, llm, doc)
            processed += 1
        except Exception as exc:  # noqa: BLE001 - per-doc isolation
            await db.rollback()
            failed += 1
            logger.warning("Bulk metadata extract failed for %s: %s", doc.id, exc)

    return {
        "requested": len(docs),
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
    }
