"""IDP Kit Jobs API routes — list, get details, SSE streaming."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db, async_session
from idpkit.db.models import Job, User
from idpkit.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    id: str
    job_type: str
    status: str
    progress: int = 0
    document_id: Optional[str] = None
    params: Optional[dict] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    items: list[JobResponse]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=JobListResponse,
    summary="List user's jobs",
)
async def list_jobs(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all jobs belonging to the current user.

    Jobs are found via the user's documents — a job belongs to a user if
    its ``document_id`` references one of the user's documents.
    Jobs with no document_id are excluded (they belong to system tasks).
    """
    from idpkit.db.models import Document

    # Sub-query: document IDs owned by this user
    user_doc_ids = select(Document.id).where(Document.owner_id == user.id).scalar_subquery()

    stmt = (
        select(Job)
        .where(Job.document_id.in_(user_doc_ids))
        .order_by(Job.created_at.desc())
    )
    result = await db.execute(stmt)
    jobs = result.scalars().all()

    return JobListResponse(items=jobs)


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job details with progress",
)
async def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a single job by ID, verifying ownership via the linked document."""
    from idpkit.db.models import Document

    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    # Verify ownership: the job's document must belong to this user
    if job.document_id:
        doc_result = await db.execute(
            select(Document).where(
                Document.id == job.document_id,
                Document.owner_id == user.id,
            )
        )
        if not doc_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )

    return job


@router.get(
    "/{job_id}/stream",
    summary="Stream job progress updates via SSE",
    responses={
        200: {
            "description": "Server-Sent Events stream of job progress",
            "content": {"text/event-stream": {}},
        },
    },
)
async def stream_job_progress(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stream real-time progress updates for a job using Server-Sent Events.

    The stream emits ``progress`` events with JSON data containing
    ``job_id``, ``status``, ``progress``, ``error``, and ``result`` fields.
    The stream closes automatically when the job reaches a terminal state
    (``completed`` or ``failed``).

    Example event::

        event: progress
        data: {"job_id": "abc-123", "status": "running", "progress": 42}

    A final event with status ``completed`` or ``failed`` is always sent
    before the stream closes.
    """
    from idpkit.db.models import Document

    # Verify the job exists and the user owns it (via its document)
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    if job.document_id:
        doc_result = await db.execute(
            select(Document).where(
                Document.id == job.document_id,
                Document.owner_id == user.id,
            )
        )
        if not doc_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found",
            )

    async def _event_generator():
        """Poll the database and yield SSE events until the job finishes."""
        POLL_INTERVAL = 1.0  # seconds
        MAX_POLLS = 600  # ~10 minutes maximum
        last_progress = -1
        last_status = None

        for _ in range(MAX_POLLS):
            try:
                async with async_session() as poll_db:
                    poll_result = await poll_db.execute(
                        select(Job).where(Job.id == job_id)
                    )
                    current_job = poll_result.scalar_one_or_none()

                if not current_job:
                    yield _sse_event("error", {"detail": "Job disappeared"})
                    return

                # Emit only when something changed
                if current_job.progress != last_progress or current_job.status != last_status:
                    last_progress = current_job.progress
                    last_status = current_job.status

                    payload = {
                        "job_id": current_job.id,
                        "status": current_job.status,
                        "progress": current_job.progress,
                    }
                    if current_job.error:
                        payload["error"] = current_job.error
                    if current_job.result and current_job.status == "completed":
                        payload["result"] = current_job.result

                    yield _sse_event("progress", payload)

                    # Terminal states — send final event and close
                    if current_job.status in ("completed", "failed"):
                        return

            except Exception as exc:
                logger.warning("SSE poll error for job %s: %s", job_id, exc)
                yield _sse_event("error", {"detail": str(exc)})
                return

            await asyncio.sleep(POLL_INTERVAL)

        # Timeout
        yield _sse_event("error", {"detail": "Stream timeout — job still running"})

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


def _sse_event(event: str, data: dict) -> str:
    """Format a single Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
