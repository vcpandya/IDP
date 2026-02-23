"""IDP Kit YouTube API routes — ingest videos, playlists, and channels."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db, async_session
from idpkit.db.models import Document, Job, Tag, User
from idpkit.api.deps import get_current_user, get_storage, get_llm
from idpkit.core.storage import StorageBackend
from idpkit.core.llm import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/youtube", tags=["youtube"])


class YouTubeIngestRequest(BaseModel):
    url: str = Field(..., description="YouTube video, playlist, or channel URL")
    video_ids: Optional[list[str]] = Field(None, description="Specific video IDs to import (for selective playlist/channel import)")
    auto_index: bool = Field(True, description="Auto-trigger indexing after transcript extraction")
    auto_tag: bool = Field(False, description="Run AI auto-tagging after processing")
    tag_id: Optional[str] = Field(None, description="Assign to existing tag")


class YouTubeIngestResponse(BaseModel):
    job_id: str
    url_type: str
    video_count: int
    status: str = "pending"
    detail: str = "YouTube ingestion job queued"


class YouTubePreviewResponse(BaseModel):
    url_type: str
    video_count: int
    videos: list[dict]


@router.post(
    "/preview",
    response_model=YouTubePreviewResponse,
    summary="Preview what a YouTube URL contains",
)
async def preview_url(
    body: YouTubeIngestRequest,
    user: User = Depends(get_current_user),
):
    try:
        from idpkit.youtube.resolver import resolve_url, detect_url_type, enrich_videos
        url_type, _ = detect_url_type(body.url)
        resolved = resolve_url(body.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("YouTube preview failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to resolve YouTube URL")

    raw_videos = resolved["videos"]
    if resolved["type"] != "video":
        try:
            raw_videos = enrich_videos(raw_videos)
        except Exception as exc:
            logger.warning("YouTube enrichment failed, using basic metadata: %s", exc)

    preview_videos = []
    for v in raw_videos:
        preview_videos.append({
            "video_id": v.get("video_id", ""),
            "title": v.get("title", ""),
            "url": v.get("url", f"https://www.youtube.com/watch?v={v.get('video_id', '')}"),
            "published_at": v.get("published_at", ""),
            "view_count": v.get("view_count", 0),
            "like_count": v.get("like_count", 0),
            "duration_seconds": v.get("duration_seconds", 0),
            "channel_title": v.get("channel_title", ""),
        })

    return YouTubePreviewResponse(
        url_type=resolved["type"],
        video_count=resolved["total"],
        videos=preview_videos,
    )


@router.post(
    "/ingest",
    response_model=YouTubeIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest YouTube video(s) into the knowledge base",
)
async def ingest_youtube(
    body: YouTubeIngestRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        from idpkit.youtube.resolver import resolve_url
        resolved = resolve_url(body.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("YouTube URL resolution failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to resolve YouTube URL")

    if body.video_ids:
        selected = set(body.video_ids)
        resolved["videos"] = [v for v in resolved["videos"] if v.get("video_id") in selected]
        resolved["total"] = len(resolved["videos"])
        if not resolved["videos"]:
            raise HTTPException(status_code=400, detail="None of the selected video IDs were found in the resolved URL")

    if body.tag_id:
        tag_result = await db.execute(
            select(Tag).where(Tag.id == body.tag_id, Tag.owner_id == user.id)
        )
        if not tag_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Tag not found")

    job = Job(
        job_type="youtube_ingest",
        status="pending",
        params={
            "url": body.url,
            "url_type": resolved["type"],
            "video_count": resolved["total"],
            "auto_index": body.auto_index,
            "auto_tag": body.auto_tag,
            "tag_id": body.tag_id,
        },
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    background_tasks.add_task(
        _run_youtube_ingest,
        job_id=job.id,
        user_id=user.id,
        resolved=resolved,
        auto_index=body.auto_index,
        auto_tag=body.auto_tag,
        tag_id=body.tag_id,
    )

    return YouTubeIngestResponse(
        job_id=job.id,
        url_type=resolved["type"],
        video_count=resolved["total"],
    )


@router.get(
    "/jobs/{job_id}",
    summary="Check YouTube ingestion job status",
)
async def get_youtube_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


async def _run_youtube_ingest(
    job_id: str,
    user_id: str,
    resolved: dict,
    auto_index: bool,
    auto_tag: bool,
    tag_id: Optional[str],
) -> None:
    from idpkit.parsing.youtube_parser import fetch_transcript, transcript_to_markdown
    from idpkit.youtube.resolver import get_video_metadata

    async with async_session() as db:
        try:
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                return

            job.status = "running"
            job.progress = 0
            db.add(job)
            await db.commit()

            videos = resolved["videos"]
            total = len(videos)
            created_doc_ids = []
            errors = []

            for i, video_info in enumerate(videos):
                video_id = video_info.get("video_id", "")
                if not video_id:
                    continue

                try:
                    try:
                        metadata = get_video_metadata(video_id)
                    except Exception:
                        metadata = {
                            "video_id": video_id,
                            "title": video_info.get("title", f"YouTube Video {video_id}"),
                            "channel_title": "",
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                        }

                    transcript_result = fetch_transcript(video_id)

                    if not transcript_result.text:
                        error_msg = transcript_result.metadata.get("error", "No transcript")
                        errors.append({"video_id": video_id, "error": error_msg})
                        logger.warning("No transcript for %s: %s", video_id, error_msg)
                        continue

                    markdown = transcript_to_markdown(transcript_result, metadata)
                    content_bytes = markdown.encode("utf-8")

                    title = metadata.get("title", f"YouTube {video_id}")
                    filename = f"{_sanitize_filename(title)}.md"

                    doc = Document(
                        filename=filename,
                        format="youtube",
                        file_size=len(content_bytes),
                        page_count=transcript_result.page_count,
                        status="uploaded",
                        source_url=metadata.get("url", f"https://www.youtube.com/watch?v={video_id}"),
                        source_type="youtube",
                        metadata_json={
                            **metadata,
                            "transcript_metadata": transcript_result.metadata,
                            "temporal_segments": [
                                {
                                    "page": p["page"],
                                    "start_time": p["start_time"],
                                    "end_time": p["end_time"],
                                    "timestamp_label": p["timestamp_label"],
                                }
                                for p in transcript_result.pages
                            ],
                        },
                        owner_id=user_id,
                    )
                    db.add(doc)
                    await db.flush()

                    storage = get_storage()
                    storage_key = f"{user_id}/{doc.id}/original.md"
                    await storage.put(storage_key, content_bytes)
                    doc.file_path = storage_key
                    db.add(doc)
                    await db.commit()

                    created_doc_ids.append(doc.id)

                    if tag_id:
                        tag_result = await db.execute(
                            select(Tag).where(Tag.id == tag_id, Tag.owner_id == user_id)
                        )
                        tag = tag_result.scalar_one_or_none()
                        if tag:
                            tag.documents.append(doc)
                            db.add(tag)
                            await db.commit()

                    if auto_index and doc.file_path:
                        await _trigger_indexing(doc.id, user_id, doc.file_path, db)

                    if auto_tag:
                        try:
                            from idpkit.engine.auto_tagger import suggest_tags, apply_tags
                            llm = get_llm()
                            suggestions = await suggest_tags(doc.id, user_id, db, llm)
                            if suggestions:
                                await apply_tags(doc.id, suggestions, user_id, db)
                        except Exception as tag_exc:
                            logger.warning("Auto-tag failed for doc %s: %s", doc.id, tag_exc)

                except Exception as exc:
                    logger.error("Failed to process video %s: %s", video_id, exc)
                    errors.append({"video_id": video_id, "error": str(exc)})

                progress = int(((i + 1) / total) * 100)
                job.progress = progress
                db.add(job)
                await db.commit()

            job.status = "completed"
            job.progress = 100
            job.result = {
                "documents_created": len(created_doc_ids),
                "document_ids": created_doc_ids,
                "errors": errors,
                "total_videos": total,
            }
            job.completed_at = datetime.now(timezone.utc)
            db.add(job)
            await db.commit()

            logger.info(
                "YouTube ingestion completed: job=%s, docs=%d, errors=%d",
                job_id, len(created_doc_ids), len(errors),
            )

        except Exception as exc:
            logger.exception("YouTube ingestion failed: %s", exc)
            try:
                result = await db.execute(select(Job).where(Job.id == job_id))
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.error = str(exc)
                    job.completed_at = datetime.now(timezone.utc)
                    db.add(job)
                    await db.commit()
            except Exception:
                logger.exception("Failed to record error for job %s", job_id)


async def _trigger_indexing(doc_id: str, user_id: str, storage_path: str, db: AsyncSession):
    try:
        from idpkit.api.routes.indexing import _run_indexing_task

        job = Job(
            job_type="index",
            status="pending",
            document_id=doc_id,
            params={},
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        import asyncio
        asyncio.create_task(
            _run_indexing_task(
                job_id=job.id,
                doc_id=doc_id,
                user_id=user_id,
                storage_path=storage_path,
                params={},
            )
        )
    except Exception as exc:
        logger.warning("Failed to trigger indexing for doc %s: %s", doc_id, exc)


def _sanitize_filename(title: str) -> str:
    import re
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    sanitized = sanitized.strip()[:100]
    return sanitized or "youtube_video"
