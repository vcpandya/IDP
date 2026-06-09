"""Smart metadata API — categories, facets, filtering, graph, and (re)processing.

Powers the Document Map: browse category-aware facets, combine them to pre-filter
a set of documents, visualise them as a document-centric graph, and reprocess
existing documents to (re)extract their metadata.
"""

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.api.deps import get_current_user, get_db, get_llm, limiter
from idpkit.core.llm import LLMClient
from idpkit.db.models import Document, Tag, User, document_tags, generate_uuid
from idpkit.metadata import categories as cat_registry
from idpkit.metadata import queries as md_queries
from idpkit.metadata.extractor import profile_document
from idpkit.metadata.job_runner import run_extraction_job
from idpkit.metadata.models import MetadataJob

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
    # Optional scope: restrict to a knowledge base (tag) or an explicit set.
    tag_id: str | None = None
    document_ids: list[str] | None = None


class ExtractBulkRequest(BaseModel):
    document_ids: list[str] | None = None
    tag_ids: list[str] | None = None
    # Used only when document_ids and tag_ids are omitted.
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


async def _tag_doc_ids(db: AsyncSession, owner_id: str, tag_ids: list[str]) -> list[str]:
    """Owner-scoped document ids belonging to any of *tag_ids*."""
    if not tag_ids:
        return []
    rows = (
        await db.execute(
            select(document_tags.c.document_id)
            .join(Document, Document.id == document_tags.c.document_id)
            .join(Tag, Tag.id == document_tags.c.tag_id)
            .where(
                Tag.owner_id == owner_id,
                Document.owner_id == owner_id,
                document_tags.c.tag_id.in_(tag_ids),
            )
            .distinct()
        )
    ).all()
    return [r[0] for r in rows]


async def _resolve_scope_doc_ids(
    db: AsyncSession,
    owner_id: str,
    *,
    tag_id: str | None = None,
    document_ids: list[str] | None = None,
) -> list[str] | None:
    """Resolve a read-scope to an owner-scoped doc-id list, or ``None`` for all.

    ``None`` means "all of the owner's documents". A non-None (possibly empty)
    list restricts the view to a knowledge base / selection; an empty list
    yields an empty result rather than falling back to "all".
    """
    if tag_id:
        return await _tag_doc_ids(db, owner_id, [tag_id])
    if document_ids is not None:
        if not document_ids:
            return []
        rows = (
            await db.execute(
                select(Document.id).where(
                    Document.id.in_(document_ids), Document.owner_id == owner_id
                )
            )
        ).all()
        return [r[0] for r in rows]
    return None


# ---------------------------------------------------------------------------
# Registry + read endpoints
# ---------------------------------------------------------------------------


@router.get("/categories", summary="List document categories and their schemas")
async def list_categories(user: User = Depends(get_current_user)):
    return cat_registry.list_categories()


@router.get("/stats", summary="Smart-metadata coverage stats")
async def stats(
    tag_id: str | None = Query(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc_ids = await _resolve_scope_doc_ids(db, user.id, tag_id=tag_id)
    return await md_queries.get_stats(db, user.id, doc_ids=doc_ids)


@router.get("/facets", summary="Aggregated facets grouped by field")
async def get_facets(
    category: str | None = Query(None),
    key: str | None = Query(None),
    search: str | None = Query(None),
    tag_id: str | None = Query(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc_ids = await _resolve_scope_doc_ids(db, user.id, tag_id=tag_id)
    return await md_queries.get_facets(
        db, user.id, category=category, key=key, search=search, doc_ids=doc_ids
    )


@router.post("/filter", summary="Documents matching combined facet criteria")
async def filter_documents(
    req: FilterRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    criteria = [c.model_dump() for c in req.criteria]
    doc_ids = await _resolve_scope_doc_ids(
        db, user.id, tag_id=req.tag_id, document_ids=req.document_ids
    )
    return await md_queries.filter_documents(
        db, user.id, criteria, match=req.match, doc_ids=doc_ids
    )


@router.post("/graph", summary="Document-centric facet graph for a selection")
async def facet_graph(
    req: FilterRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    criteria = [c.model_dump() for c in req.criteria]
    doc_ids = await _resolve_scope_doc_ids(
        db, user.id, tag_id=req.tag_id, document_ids=req.document_ids
    )
    return await md_queries.build_facet_graph(
        db, user.id, criteria, match=req.match, doc_ids=doc_ids
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


def _job_payload(job: MetadataJob) -> dict:
    return {
        "job_id": job.id,
        "status": job.status,
        "scope": job.scope,
        "label": job.label,
        "total": job.total or 0,
        "processed": job.processed or 0,
        "failed": job.failed or 0,
        "skipped": job.skipped or 0,
        "current": job.current,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


@router.post("/extract-bulk", summary="Start a background metadata (re)extraction job")
@limiter.limit("12/minute")
async def extract_bulk(
    request: Request,
    req: ExtractBulkRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Resolve a working set and launch a background extraction job.

    Returns immediately with a ``job_id`` the client polls via
    ``GET /jobs/{job_id}`` for live progress. The work set is capped at
    ``MAX_BULK_DOCS`` so a single run can never process unbounded documents.
    """
    if req.document_ids and len(req.document_ids) > MAX_BULK_DOCS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many documents; limit is {MAX_BULK_DOCS} per request.",
        )

    # Resolve the working set (owner-scoped) and a human label for the job.
    if req.document_ids:
        rows = (
            await db.execute(
                select(Document.id).where(
                    Document.id.in_(req.document_ids), Document.owner_id == user.id
                )
            )
        ).all()
        doc_ids = [r[0] for r in rows]
        scope_kind, label = "selection", f"{len(doc_ids)} selected document(s)"
    elif req.tag_ids:
        doc_ids = await _tag_doc_ids(db, user.id, req.tag_ids)
        if req.scope == "missing" and doc_ids:
            # Restrict the KB working set to documents not yet profiled.
            rows = (
                await db.execute(
                    select(Document.id).where(
                        Document.id.in_(doc_ids),
                        Document.doc_category.is_(None),
                    )
                )
            ).all()
            doc_ids = [r[0] for r in rows]
        tag_names = (
            await db.execute(
                select(Tag.name).where(
                    Tag.id.in_(req.tag_ids), Tag.owner_id == user.id
                )
            )
        ).all()
        names = ", ".join(n[0] for n in tag_names) or "knowledge base"
        suffix = " (unprocessed)" if req.scope == "missing" else ""
        scope_kind, label = "tag", f"Knowledge base: {names}{suffix}"
    else:
        stmt = select(Document.id).where(Document.owner_id == user.id)
        if req.scope == "missing":
            stmt = stmt.where(Document.doc_category.is_(None))
        rows = (await db.execute(stmt)).all()
        doc_ids = [r[0] for r in rows]
        scope_kind = req.scope
        label = "All documents" if req.scope == "all" else "Unprocessed documents"

    # Cap the work set so a single job stays bounded.
    doc_ids = doc_ids[:MAX_BULK_DOCS]

    job = MetadataJob(
        id=generate_uuid(),
        owner_id=user.id,
        status="pending",
        scope=scope_kind,
        label=label,
        total=len(doc_ids),
    )
    db.add(job)
    await db.flush()
    job_id = job.id
    # Commit before launching so the background task (and any worker polling it)
    # can see the row immediately.
    await db.commit()

    if doc_ids:
        asyncio.create_task(run_extraction_job(job_id, user.id, doc_ids))
    else:
        # Nothing to do — mark completed so the UI doesn't spin forever.
        from sqlalchemy import update as _update

        await db.execute(
            _update(MetadataJob).where(MetadataJob.id == job_id).values(
                status="completed"
            )
        )
        await db.commit()

    return {"job_id": job_id, "total": len(doc_ids), "label": label, "status": "started"}


@router.get("/jobs/{job_id}", summary="Poll a metadata extraction job")
async def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = (
        await db.execute(
            select(MetadataJob).where(
                MetadataJob.id == job_id, MetadataJob.owner_id == user.id
            )
        )
    ).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_payload(job)


@router.get("/jobs", summary="Recent metadata extraction jobs")
async def list_jobs(
    limit: int = Query(10, ge=1, le=50),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    jobs = (
        await db.execute(
            select(MetadataJob)
            .where(MetadataJob.owner_id == user.id)
            .order_by(MetadataJob.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()
    return [_job_payload(j) for j in jobs]
