"""YouTube URL parsing and metadata resolution via YouTube Data API v3."""

import enum
import logging
import os
import re
from typing import Optional

from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

_MAX_PLAYLIST_VIDEOS = 200
_MAX_CHANNEL_VIDEOS = 200


class YouTubeURLType(str, enum.Enum):
    VIDEO = "video"
    PLAYLIST = "playlist"
    CHANNEL = "channel"
    UNKNOWN = "unknown"


_VIDEO_ID_PATTERNS = [
    re.compile(r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})"),
]

_PLAYLIST_ID_PATTERN = re.compile(r"[?&]list=([a-zA-Z0-9_-]+)")
_CHANNEL_ID_PATTERN = re.compile(r"youtube\.com/channel/([a-zA-Z0-9_-]+)")
_CHANNEL_HANDLE_PATTERN = re.compile(r"youtube\.com/@([a-zA-Z0-9_.-]+)")
_CHANNEL_USER_PATTERN = re.compile(r"youtube\.com/user/([a-zA-Z0-9_-]+)")


def _get_api_key() -> str:
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        raise ValueError("YOUTUBE_API_KEY environment variable is not set")
    return key


def _build_youtube():
    return build("youtube", "v3", developerKey=_get_api_key())


def extract_video_id(url: str) -> Optional[str]:
    for pat in _VIDEO_ID_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


def detect_url_type(url: str) -> tuple[YouTubeURLType, Optional[str]]:
    pm = _PLAYLIST_ID_PATTERN.search(url)
    if pm:
        return YouTubeURLType.PLAYLIST, pm.group(1)

    cm = _CHANNEL_ID_PATTERN.search(url)
    if cm:
        return YouTubeURLType.CHANNEL, cm.group(1)

    hm = _CHANNEL_HANDLE_PATTERN.search(url)
    if hm:
        return YouTubeURLType.CHANNEL, f"@{hm.group(1)}"

    um = _CHANNEL_USER_PATTERN.search(url)
    if um:
        return YouTubeURLType.CHANNEL, f"user:{um.group(1)}"

    vid = extract_video_id(url)
    if vid:
        return YouTubeURLType.VIDEO, vid

    return YouTubeURLType.UNKNOWN, None


def _parse_duration(iso_duration: str) -> int:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration or "")
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mi * 60 + s


def get_video_metadata(video_id: str) -> dict:
    yt = _build_youtube()
    resp = yt.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id,
    ).execute()

    items = resp.get("items", [])
    if not items:
        raise ValueError(f"Video not found: {video_id}")

    item = items[0]
    snippet = item["snippet"]
    details = item.get("contentDetails", {})
    stats = item.get("statistics", {})

    return {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "channel_id": snippet.get("channelId", ""),
        "description": (snippet.get("description") or "")[:2000],
        "published_at": snippet.get("publishedAt", ""),
        "duration_seconds": _parse_duration(details.get("duration", "")),
        "thumbnail_url": (snippet.get("thumbnails", {}).get("high") or snippet.get("thumbnails", {}).get("default", {})).get("url", ""),
        "view_count": int(stats.get("viewCount", 0)),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "tags": snippet.get("tags", [])[:20],
    }


def get_playlist_videos(playlist_id: str) -> list[dict]:
    yt = _build_youtube()
    videos = []
    page_token = None

    while len(videos) < _MAX_PLAYLIST_VIDEOS:
        resp = yt.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token,
        ).execute()

        for item in resp.get("items", []):
            vid = item["snippet"]["resourceId"].get("videoId")
            if vid:
                videos.append({
                    "video_id": vid,
                    "title": item["snippet"].get("title", ""),
                    "position": item["snippet"].get("position", 0),
                })

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return videos[:_MAX_PLAYLIST_VIDEOS]


def _resolve_channel_id(identifier: str) -> str:
    yt = _build_youtube()

    if identifier.startswith("@"):
        resp = yt.channels().list(
            part="contentDetails",
            forHandle=identifier[1:],
        ).execute()
    elif identifier.startswith("user:"):
        resp = yt.channels().list(
            part="contentDetails",
            forUsername=identifier[5:],
        ).execute()
    else:
        resp = yt.channels().list(
            part="contentDetails",
            id=identifier,
        ).execute()

    items = resp.get("items", [])
    if not items:
        raise ValueError(f"Channel not found: {identifier}")

    uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    return uploads_playlist


def get_channel_videos(channel_identifier: str) -> list[dict]:
    uploads_playlist = _resolve_channel_id(channel_identifier)
    return get_playlist_videos(uploads_playlist)


def enrich_videos(videos: list[dict]) -> list[dict]:
    if not videos:
        return videos
    yt = _build_youtube()
    ids = [v["video_id"] for v in videos if v.get("video_id")]
    meta_map: dict[str, dict] = {}

    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        resp = yt.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch),
        ).execute()
        for item in resp.get("items", []):
            vid = item["id"]
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            details = item.get("contentDetails", {})
            meta_map[vid] = {
                "published_at": snippet.get("publishedAt", ""),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "duration_seconds": _parse_duration(details.get("duration", "")),
                "channel_title": snippet.get("channelTitle", ""),
            }

    for v in videos:
        meta = meta_map.get(v["video_id"], {})
        v["published_at"] = meta.get("published_at", "")
        v["view_count"] = meta.get("view_count", 0)
        v["like_count"] = meta.get("like_count", 0)
        v["duration_seconds"] = meta.get("duration_seconds", 0)
        v["channel_title"] = meta.get("channel_title", "")

    return videos


def resolve_url(url: str) -> dict:
    url_type, identifier = detect_url_type(url)

    if url_type == YouTubeURLType.VIDEO:
        metadata = get_video_metadata(identifier)
        return {
            "type": url_type.value,
            "videos": [metadata],
            "total": 1,
        }

    if url_type == YouTubeURLType.PLAYLIST:
        videos = get_playlist_videos(identifier)
        return {
            "type": url_type.value,
            "playlist_id": identifier,
            "videos": videos,
            "total": len(videos),
        }

    if url_type == YouTubeURLType.CHANNEL:
        videos = get_channel_videos(identifier)
        return {
            "type": url_type.value,
            "channel": identifier,
            "videos": videos,
            "total": len(videos),
        }

    raise ValueError(f"Could not parse YouTube URL: {url}")
