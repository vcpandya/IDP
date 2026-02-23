"""IDP Kit Indexing API routes — trigger indexing, check status, get tree."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db, async_session
from idpkit.db.models import Document, Job, User
from idpkit.api.deps import get_current_user, get_storage, get_llm
from idpkit.core.storage import StorageBackend
from idpkit.core.llm import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/indexing", tags=["indexing"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    model: Optional[str] = Field(None, description="LLM model override for indexing")
    toc_check_pages: Optional[int] = Field(
        None, ge=1, le=50, description="Number of leading pages to check for a table of contents"
    )
    max_pages_per_node: Optional[int] = Field(
        None, ge=1, le=200, description="Maximum pages per tree node"
    )


class JobStatusResponse(BaseModel):
    id: str
    job_type: str
    status: str
    progress: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class IndexTriggerResponse(BaseModel):
    job_id: str
    document_id: str
    status: str = "pending"
    detail: str = "Indexing job queued"


class TreeResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    tree_index: Optional[dict] = None


# ---------------------------------------------------------------------------
# Background task: run indexing
# ---------------------------------------------------------------------------

async def _run_indexing_task(
    job_id: str,
    doc_id: str,
    user_id: str,
    storage_path: str,
    params: dict,
) -> None:
    """Execute tree indexing in the background.

    This function runs outside of the request lifecycle, so it creates its
    own database session.  On completion it writes the tree_index JSON back
    to the Document row and marks the Job as completed (or failed).
    """
    async with async_session() as db:
        try:
            # Mark job as running
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                logger.error("Indexing background task: job %s not found", job_id)
                return

            job.status = "running"
            job.progress = 0
            db.add(job)
            await db.commit()

            # Mark document as indexing
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if not doc:
                job.status = "failed"
                job.error = "Document not found"
                job.completed_at = datetime.now(timezone.utc)
                db.add(job)
                await db.commit()
                return

            doc.status = "indexing"
            db.add(doc)
            await db.commit()

            # --- Attempt to call the indexer engine ---
            # Import here to avoid circular imports and allow graceful
            # degradation when indexing engine is not yet implemented.
            tree_result = None
            try:
                from idpkit.engine.page_index import build_tree_index

                llm = get_llm()
                storage = get_storage()

                tree_result = await build_tree_index(
                    file_key=storage_path,
                    storage=storage,
                    llm=llm,
                    model=params.get("model"),
                    toc_check_pages=params.get("toc_check_pages"),
                    max_pages_per_node=params.get("max_pages_per_node"),
                    progress_callback=lambda pct: _update_job_progress(job_id, pct),
                )
            except ImportError:
                logger.warning(
                    "build_tree_index not available; storing placeholder tree for doc %s",
                    doc_id,
                )
                tree_result = {
                    "doc_name": doc.filename,
                    "doc_description": None,
                    "structure": [],
                    "_placeholder": True,
                }
            except Exception as exc:
                logger.error("Indexing failed for document %s: %s", doc_id, exc)
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = datetime.now(timezone.utc)
                doc.status = "failed"
                db.add(job)
                db.add(doc)
                await db.commit()
                return

            # Persist result
            if isinstance(tree_result, dict):
                tree_json = tree_result
            elif hasattr(tree_result, "model_dump"):
                tree_json = tree_result.model_dump()
            else:
                tree_json = {"raw": str(tree_result)}

            doc.tree_index = tree_json
            doc.status = "indexed"
            db.add(doc)

            job.status = "completed"
            job.progress = 100
            job.result = tree_json
            job.completed_at = datetime.now(timezone.utc)
            db.add(job)

            await db.commit()
            logger.info("Indexing completed for document %s (job %s)", doc_id, job_id)

        except Exception as exc:
            logger.exception("Unexpected error in indexing background task: %s", exc)
            try:
                result = await db.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.error = f"Unexpected error: {exc}"
                    job.completed_at = datetime.now(timezone.utc)
                    db.add(job)

                result = await db.execute(select(Document).where(Document.id == doc_id))
                doc = result.scalar_one_or_none()
                if doc:
                    doc.status = "failed"
                    db.add(doc)

                await db.commit()
            except Exception:
                logger.exception("Failed to record error state for job %s", job_id)


async def _update_job_progress(job_id: str, progress: int) -> None:
    """Helper to update job progress from inside the indexer."""
    try:
        async with async_session() as db:
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.progress = min(max(progress, 0), 100)
                db.add(job)
                await db.commit()
    except Exception as exc:
        logger.warning("Failed to update job progress for %s: %s", job_id, exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/documents/{doc_id}/index",
    response_model=IndexTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger tree indexing for a document",
)
async def trigger_indexing(
    doc_id: str,
    body: Optional[IndexRequest] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue a background tree-indexing job for the specified document.

    Returns immediately with job metadata; use the status endpoint to poll
    for completion.
    """
    # Resolve body (allow empty body)
    if body is None:
        body = IndexRequest()

    # Verify ownership & existence
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if not doc.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no stored file",
        )

    # Prevent re-indexing while a job is already running
    running_result = await db.execute(
        select(Job).where(
            Job.document_id == doc_id,
            Job.job_type == "index",
            Job.status.in_(["pending", "running"]),
        )
    )
    running_job = running_result.scalar_one_or_none()
    if running_job:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An indexing job is already in progress (job_id={running_job.id})",
        )

    # Create job record
    params = body.model_dump(exclude_none=True)
    job = Job(
        job_type="index",
        status="pending",
        document_id=doc_id,
        params=params,
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)

    # Enqueue background task
    background_tasks.add_task(
        _run_indexing_task,
        job_id=job.id,
        doc_id=doc_id,
        user_id=user.id,
        storage_path=doc.file_path,
        params=params,
    )

    logger.info("Indexing job queued: job=%s, doc=%s", job.id, doc_id)
    return IndexTriggerResponse(job_id=job.id, document_id=doc_id)


@router.get(
    "/documents/{doc_id}/index/status",
    response_model=JobStatusResponse,
    summary="Get indexing job status for a document",
)
async def get_indexing_status(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the most recent indexing job for the given document."""
    # Verify document ownership
    doc_result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    if not doc_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    result = await db.execute(
        select(Job)
        .where(Job.document_id == doc_id, Job.job_type == "index")
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No indexing job found for this document",
        )

    return job


@router.get(
    "/documents/{doc_id}/tree",
    response_model=TreeResponse,
    summary="Get the document tree index",
)
async def get_tree_index(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the tree-index JSON for an indexed document.

    Returns ``tree_index: null`` if the document has not been indexed yet.
    """
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return TreeResponse(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status,
        tree_index=doc.tree_index,
    )
