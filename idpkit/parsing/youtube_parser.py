"""YouTube transcript parser — extracts timestamped transcripts as temporal documents."""

import logging
import os
from typing import Optional

import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
)

from .base import ParseResult

logger = logging.getLogger(__name__)

_SEGMENT_DURATION = 120


def _fetch_via_supadata(video_id: str) -> Optional[list[dict]]:
    api_key = os.environ.get("SUPADATA_API_KEY")
    if not api_key:
        logger.debug("SUPADATA_API_KEY not set, skipping Supadata fallback")
        return None

    url = "https://api.supadata.ai/v1/transcript"
    params = {"url": f"https://youtu.be/{video_id}"}
    headers = {"x-api-key": api_key}

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()

        data = resp.json()
        content = data.get("content")
        if not content or not isinstance(content, list):
            logger.warning("Supadata returned empty or invalid content for %s", video_id)
            return None

        entries = []
        for item in content:
            entries.append({
                "text": item.get("text", ""),
                "start": item.get("offset", 0) / 1000.0,
                "duration": item.get("duration", 0) / 1000.0,
            })

        logger.info("Supadata returned %d transcript entries for %s", len(entries), video_id)
        return entries

    except httpx.HTTPStatusError as exc:
        logger.warning("Supadata API HTTP error for %s: %s", video_id, exc.response.status_code)
        return None
    except Exception as exc:
        logger.warning("Supadata API request failed for %s: %s", video_id, exc)
        return None


def _format_timestamp(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fetch_via_yt_api(video_id: str, languages: list[str]) -> Optional[tuple[list[dict], dict]]:
    api = YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)

        transcript_obj = None
        try:
            transcript_obj = transcript_list.find_transcript(languages)
        except NoTranscriptFound:
            try:
                transcript_obj = transcript_list.find_generated_transcript(languages)
            except NoTranscriptFound:
                for t in transcript_list:
                    transcript_obj = t
                    break

        if transcript_obj is None:
            logger.info("youtube-transcript-api found no transcript for %s", video_id)
            return None

        fetched = transcript_obj.fetch()

        entries = []
        for snippet in fetched:
            entries.append({
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration,
            })

        language_info = {
            "language": getattr(fetched, "language", "unknown"),
            "language_code": getattr(fetched, "language_code", ""),
            "is_generated": getattr(fetched, "is_generated", None),
        }

        logger.info("youtube-transcript-api returned %d entries for %s", len(entries), video_id)
        return (entries, language_info)

    except TranscriptsDisabled:
        logger.warning("youtube-transcript-api: transcripts disabled for %s", video_id)
        return None
    except Exception as exc:
        logger.warning("youtube-transcript-api failed for %s: %s", video_id, exc)
        return None


def fetch_transcript(
    video_id: str,
    languages: Optional[list[str]] = None,
) -> ParseResult:
    if languages is None:
        languages = ["en"]

    entries = None
    language_info = {}
    transcript_source = "none"

    logger.info("Fetching transcript for %s: trying youtube-transcript-api", video_id)
    yt_result = _fetch_via_yt_api(video_id, languages)
    if yt_result is not None:
        entries, language_info = yt_result
        transcript_source = "youtube-transcript-api"
        logger.info("Transcript for %s obtained via youtube-transcript-api", video_id)

    if entries is None:
        logger.info("Falling back to Supadata API for %s", video_id)
        supadata_entries = _fetch_via_supadata(video_id)
        if supadata_entries is not None:
            entries = supadata_entries
            transcript_source = "supadata"
            logger.info("Transcript for %s obtained via Supadata API", video_id)

    if entries is None:
        logger.warning("All transcript sources failed for %s, returning empty result", video_id)
        return ParseResult(
            text="",
            pages=[],
            metadata={"video_id": video_id, "transcript_source": "none"},
            page_count=0,
        )

    segments = _build_temporal_segments(entries)

    full_text_parts = []
    pages = []
    for i, seg in enumerate(segments, 1):
        label = f"{_format_timestamp(seg['start_time'])} - {_format_timestamp(seg['end_time'])}"
        page_text = seg["text"]
        full_text_parts.append(f"[{label}]\n{page_text}")
        pages.append({
            "page": i,
            "text": page_text,
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "timestamp_label": label,
        })

    full_text = "\n\n".join(full_text_parts)

    total_duration = entries[-1]["start"] + entries[-1].get("duration", 0) if entries else 0

    metadata = {
        "video_id": video_id,
        "transcript_source": transcript_source,
        "transcript_language": language_info.get("language", "unknown"),
        "transcript_language_code": language_info.get("language_code", ""),
        "is_generated": language_info.get("is_generated"),
        "total_duration_seconds": total_duration,
        "segment_count": len(segments),
        "entry_count": len(entries),
    }

    return ParseResult(
        text=full_text,
        pages=pages,
        metadata=metadata,
        page_count=len(pages),
    )


def _build_temporal_segments(entries: list[dict]) -> list[dict]:
    if not entries:
        return []

    segments = []
    current_texts = []
    segment_start = 0.0
    segment_end = _SEGMENT_DURATION

    for entry in entries:
        start = entry.get("start", 0)
        text = entry.get("text", "").strip()

        if not text:
            continue

        if start >= segment_end and current_texts:
            segments.append({
                "start_time": segment_start,
                "end_time": segment_end,
                "text": " ".join(current_texts),
            })
            segment_start = segment_end
            segment_end = segment_start + _SEGMENT_DURATION
            current_texts = []

        current_texts.append(text)

    if current_texts:
        last_entry = entries[-1]
        actual_end = last_entry.get("start", 0) + last_entry.get("duration", 0)
        segments.append({
            "start_time": segment_start,
            "end_time": max(segment_end, actual_end),
            "text": " ".join(current_texts),
        })

    return segments


def transcript_to_markdown(parse_result: ParseResult, video_metadata: Optional[dict] = None) -> str:
    parts = []

    if video_metadata:
        parts.append(f"# {video_metadata.get('title', 'YouTube Video')}")
        parts.append("")
        if video_metadata.get("channel_title"):
            parts.append(f"**Channel:** {video_metadata['channel_title']}")
        if video_metadata.get("published_at"):
            parts.append(f"**Published:** {video_metadata['published_at']}")
        if video_metadata.get("url"):
            parts.append(f"**URL:** {video_metadata['url']}")
        if video_metadata.get("duration_seconds"):
            dur = video_metadata["duration_seconds"]
            parts.append(f"**Duration:** {_format_timestamp(dur)}")
        if video_metadata.get("description"):
            desc = video_metadata["description"][:500]
            parts.append(f"\n> {desc}")
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append("## Transcript")
        parts.append("")

    for page in parse_result.pages:
        label = page.get("timestamp_label", "")
        parts.append(f"### [{label}]")
        parts.append("")
        parts.append(page["text"])
        parts.append("")

    return "\n".join(parts)
