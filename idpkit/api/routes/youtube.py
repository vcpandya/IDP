"""IDP Kit YouTube API routes — ingest videos, playlists, and channels."""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, insert
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.session import get_db, async_session
from idpkit.db.models import Document, Job, Tag, User, document_tags
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
    default_tag_name: Optional[str] = Field(None, description="Auto-create or reuse a tag with this name (for channel/playlist grouping)")


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
    source_name: str = ""


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
        source_name=resolved.get("source_name", ""),
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

    tag_id = body.tag_id
    if not tag_id and body.default_tag_name:
        clean_name = re.sub(r'[^\w\s-]', '', body.default_tag_name).strip()
        if clean_name:
            existing = await db.execute(
                select(Tag).where(
                    func.lower(Tag.name) == clean_name.lower(),
                    Tag.owner_id == user.id,
                )
            )
            tag = existing.scalar_one_or_none()
            if tag:
                tag_id = tag.id
            else:
                tag = Tag(name=clean_name, owner_id=user.id, color="#ff0000")
                db.add(tag)
                await db.flush()
                tag_id = tag.id

    if tag_id:
        tag_result = await db.execute(
            select(Tag).where(Tag.id == tag_id, Tag.owner_id == user.id)
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
            "tag_id": tag_id,
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
        tag_id=tag_id,
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


@router.post(
    "/jobs/{job_id}/cancel",
    summary="Cancel a running YouTube ingestion job",
)
async def cancel_youtube_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in ("completed", "failed", "cancelled"):
        return {"id": job.id, "status": job.status, "message": "Job already finished"}
    job.status = "cancelled"
    job.completed_at = datetime.now(timezone.utc)
    db.add(job)
    await db.commit()
    return {"id": job.id, "status": "cancelled", "message": "Job cancellation requested"}


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
            source_counts = {
                "youtube-transcript-api": 0,
                "supadata": 0,
                "none": 0,
            }

            for i, video_info in enumerate(videos):
                async with async_session() as check_db:
                    check_result = await check_db.execute(
                        select(Job.status).where(Job.id == job_id)
                    )
                    current_status = check_result.scalar_one_or_none()
                if current_status == "cancelled":
                    logger.info("YouTube ingestion cancelled: job=%s after %d/%d videos", job_id, i, total)
                    job.result = {
                        "documents_created": len(created_doc_ids),
                        "document_ids": created_doc_ids,
                        "errors": errors,
                        "total_videos": total,
                        "processed_videos": i,
                        "transcript_source_counts": source_counts,
                        "cancelled": True,
                    }
                    job.completed_at = datetime.now(timezone.utc)
                    db.add(job)
                    await db.commit()
                    return

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
                    transcript_source = transcript_result.metadata.get("transcript_source", "none")
                    has_transcript = bool(transcript_result.text)

                    if transcript_source in source_counts:
                        source_counts[transcript_source] += 1
                    else:
                        source_counts[transcript_source] = 1

                    markdown = transcript_to_markdown(transcript_result, metadata)
                    content_bytes = markdown.encode("utf-8")

                    title = metadata.get("title", f"YouTube {video_id}")
                    filename = f"{_sanitize_filename(title)}.md"

                    doc_metadata = {
                        **metadata,
                        "transcript_available": has_transcript,
                        "transcript_source": transcript_source,
                        "transcript_metadata": transcript_result.metadata,
                    }
                    if has_transcript:
                        doc_metadata["temporal_segments"] = [
                            {
                                "page": p["page"],
                                "start_time": p["start_time"],
                                "end_time": p["end_time"],
                                "timestamp_label": p["timestamp_label"],
                            }
                            for p in transcript_result.pages
                        ]

                    doc = Document(
                        filename=filename,
                        format="youtube",
                        file_size=len(content_bytes),
                        page_count=transcript_result.page_count,
                        status="uploaded",
                        source_url=metadata.get("url", f"https://www.youtube.com/watch?v={video_id}"),
                        source_type="youtube",
                        metadata_json=doc_metadata,
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
                        await db.execute(
                            insert(document_tags).values(
                                document_id=doc.id, tag_id=tag_id
                            )
                        )
                        await db.commit()

                    if auto_index and has_transcript:
                        tree_structure = _build_youtube_tree_index(
                            transcript_result, metadata
                        )
                        doc.tree_index = tree_structure
                        doc.description = tree_structure.get("doc_description")
                        doc.status = "indexed"
                        db.add(doc)
                        await db.commit()
                        logger.info("Indexed YouTube doc %s inline", doc.id)

                    if auto_tag and has_transcript:
                        try:
                            from idpkit.engine.auto_tagger import suggest_tags, apply_tags
                            llm = get_llm()
                            suggestions = await suggest_tags(doc.id, user_id, db, llm)
                            if suggestions:
                                await apply_tags(doc.id, suggestions, user_id, db)
                        except Exception as tag_exc:
                            logger.warning("Auto-tag failed for doc %s: %s", doc.id, tag_exc)

                    if not has_transcript:
                        logger.info("Created metadata-only document for video %s (no transcript)", video_id)

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
                "transcript_source_counts": source_counts,
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


def _build_youtube_tree_index(transcript_result, metadata: dict) -> dict:
    title = metadata.get("title", "YouTube Video")
    channel = metadata.get("channel_title", "")
    url = metadata.get("url", "")
    description_parts = [f"YouTube video: {title}"]
    if channel:
        description_parts.append(f"by {channel}")
    doc_description = " ".join(description_parts)

    children = []
    for page in transcript_result.pages:
        label = page.get("timestamp_label", "")
        text = page.get("text", "")
        if not text.strip():
            continue
        children.append({
            "title": f"[{label}]" if label else f"Segment {len(children) + 1}",
            "text": text,
            "summary": text[:200].replace("\n", " ").strip(),
            "node_id": f"seg_{len(children)}",
            "page_start": int(page.get("start_time") or page.get("page", 0)),
            "page_end": int(page.get("end_time") or page.get("page", 0)),
        })

    full_text = "\n\n".join(p.get("text", "") for p in transcript_result.pages if p.get("text", "").strip())

    structure = [{
        "title": title,
        "text": full_text,
        "summary": doc_description,
        "node_id": "root",
        "page_start": 0,
        "page_end": max((int(p.get("end_time") or p.get("page", 0)) for p in transcript_result.pages), default=0),
        "children": children,
    }]

    return {
        "doc_name": title,
        "doc_description": doc_description,
        "structure": structure,
        "source_url": url,
    }


def _sanitize_filename(title: str) -> str:
    import re
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    sanitized = sanitized.strip()[:100]
    return sanitized or "youtube_video"
