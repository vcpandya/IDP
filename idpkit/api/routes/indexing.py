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
    stage: Optional[str] = None
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

    This function runs outside of the request lifecycle.  It uses short-lived
    database sessions for each DB operation to avoid connection timeouts
    during the long-running LLM indexing pipeline.
    """
    try:
        async with async_session() as db:
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                logger.error("Indexing background task: job %s not found", job_id)
                return

            job.status = "running"
            job.progress = 0
            job.stage = "Preparing document"
            db.add(job)
            await db.commit()

        doc_filename = None
        async with async_session() as db:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if not doc:
                await _mark_job_failed(job_id, doc_id, "Document not found")
                return
            doc_filename = doc.filename

            doc.status = "indexing"
            db.add(doc)
            await db.commit()

        await _update_job_progress(job_id, 5, "Reading file from storage")

        tree_result = None
        try:
            from idpkit.engine.page_index import build_tree_index

            llm = get_llm()
            storage = get_storage()

            _STAGE_MAP = {
                5: "Loading document",
                10: "Extracting pages",
                15: "Parsing text content",
                20: "Analyzing document structure",
                70: "Building tree index",
                90: "Generating summaries",
                100: "Finalizing",
            }

            async def _progress_with_stage(pct):
                stage = _STAGE_MAP.get(pct)
                if not stage:
                    if pct <= 10:
                        stage = "Loading document"
                    elif pct <= 20:
                        stage = "Parsing text content"
                    elif pct <= 70:
                        stage = "Analyzing document structure"
                    elif pct <= 90:
                        stage = "Building tree index"
                    else:
                        stage = "Generating summaries"
                await _update_job_progress(job_id, pct, stage)

            tree_result = await build_tree_index(
                file_key=storage_path,
                storage=storage,
                llm=llm,
                model=params.get("model"),
                toc_check_pages=params.get("toc_check_pages"),
                max_pages_per_node=params.get("max_pages_per_node"),
                progress_callback=_progress_with_stage,
            )
        except ImportError:
            logger.warning(
                "build_tree_index not available; storing placeholder tree for doc %s",
                doc_id,
            )
            await _update_job_progress(job_id, 50, "Processing document")
            tree_result = {
                "doc_name": doc_filename,
                "doc_description": None,
                "structure": [],
                "_placeholder": True,
            }
        except Exception as exc:
            logger.error("Indexing failed for document %s: %s", doc_id, exc)
            await _mark_job_failed(job_id, doc_id, str(exc))
            return

        await _update_job_progress(job_id, 95, "Saving results")

        if isinstance(tree_result, dict):
            tree_json = tree_result
        elif hasattr(tree_result, "model_dump"):
            tree_json = tree_result.model_dump()
        else:
            tree_json = {"raw": str(tree_result)}

        async with async_session() as db:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.tree_index = tree_json
                doc.status = "indexed"
                db.add(doc)

            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "completed"
                job.progress = 100
                job.stage = "Complete"
                job.result = tree_json
                job.completed_at = datetime.now(timezone.utc)
                db.add(job)

            await db.commit()
        logger.info("Indexing completed for document %s (job %s)", doc_id, job_id)

    except Exception as exc:
        logger.exception("Unexpected error in indexing background task: %s", exc)
        await _mark_job_failed(job_id, doc_id, f"Unexpected error: {exc}")


async def _mark_job_failed(job_id: str, doc_id: str, error: str) -> None:
    """Mark a job and its document as failed using a fresh DB session."""
    try:
        async with async_session() as db:
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.stage = "Failed"
                job.error = error
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


async def _update_job_progress(job_id: str, progress: int, stage: str | None = None) -> None:
    """Helper to update job progress and stage from inside the indexer."""
    try:
        async with async_session() as db:
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if job:
                job.progress = min(max(progress, 0), 100)
                if stage is not None:
                    job.stage = stage
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

    from datetime import timedelta

    STALE_JOB_MINUTES = 10

    running_result = await db.execute(
        select(Job).where(
            Job.document_id == doc_id,
            Job.job_type == "index",
            Job.status.in_(["pending", "running"]),
        )
    )
    running_job = running_result.scalar_one_or_none()
    if running_job:
        age = datetime.now(timezone.utc) - running_job.created_at.replace(tzinfo=timezone.utc)
        if age > timedelta(minutes=STALE_JOB_MINUTES):
            logger.warning(
                "Auto-resetting stale indexing job %s (age: %s)", running_job.id, age
            )
            running_job.status = "failed"
            running_job.error = f"Auto-reset: job was stale for {age}"
            running_job.completed_at = datetime.now(timezone.utc)
            db.add(running_job)

            doc_result = await db.execute(
                select(Document).where(Document.id == doc_id, Document.status == "indexing")
            )
            stale_doc = doc_result.scalar_one_or_none()
            if stale_doc:
                stale_doc.status = "uploaded"
                db.add(stale_doc)

            await db.commit()
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"An indexing job is already in progress (job_id={running_job.id})",
            )

    # Create job record and commit immediately so the background task can find it
    params = body.model_dump(exclude_none=True)
    job = Job(
        job_type="index",
        status="pending",
        document_id=doc_id,
        params=params,
    )
    db.add(job)
    await db.commit()
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


@router.get(
    "/documents/{doc_id}/jobs",
    response_model=list[JobStatusResponse],
    summary="List all indexing jobs for a document",
)
async def list_indexing_jobs(
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    doc_result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user.id)
    )
    if not doc_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    result = await db.execute(
        select(Job)
        .where(Job.document_id == doc_id, Job.job_type == "index")
        .order_by(Job.created_at.desc())
        .limit(50)
    )
    return result.scalars().all()


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=JobStatusResponse,
    summary="Cancel a pending or running indexing job",
)
async def cancel_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.document_id:
        doc_result = await db.execute(
            select(Document).where(Document.id == job.document_id, Document.owner_id == user.id)
        )
        if not doc_result.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.status not in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is already {job.status}",
        )

    job.status = "failed"
    job.error = "Cancelled by user"
    job.completed_at = datetime.now(timezone.utc)
    db.add(job)

    if job.document_id:
        doc_result = await db.execute(
            select(Document).where(Document.id == job.document_id, Document.status == "indexing")
        )
        doc = doc_result.scalar_one_or_none()
        if doc:
            doc.status = "uploaded"
            db.add(doc)

    await db.commit()
    await db.refresh(job)
    logger.info("Job cancelled: %s", job_id)
    return job


@router.delete(
    "/jobs/{job_id}",
    summary="Delete a completed or failed indexing job",
)
async def delete_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.document_id:
        doc_result = await db.execute(
            select(Document).where(Document.id == job.document_id, Document.owner_id == user.id)
        )
        if not doc_result.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.status in ("pending", "running"):
        job.status = "failed"
        job.error = "Cancelled by user"
        job.completed_at = datetime.now(timezone.utc)
        db.add(job)

        if job.document_id:
            doc_result = await db.execute(
                select(Document).where(Document.id == job.document_id, Document.status == "indexing")
            )
            doc = doc_result.scalar_one_or_none()
            if doc:
                doc.status = "uploaded"
                db.add(doc)

    await db.delete(job)
    await db.commit()
    logger.info("Job deleted: %s", job_id)
    return {"detail": "Job deleted"}
