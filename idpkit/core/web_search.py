"""Shared Jina AI web search utility.

Used by both the IDA agent tools and the Knowledge Graph insights module.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

JINA_SEARCH_URL = "https://s.jina.ai/"
JINA_READER_URL = "https://r.jina.ai/"
JINA_TIMEOUT = 30
JINA_MAX_CONTENT_LEN = 8000


def get_jina_api_key() -> str | None:
    return os.getenv("JINA_API_KEY")


async def web_search(query: str, max_results: int = 5, site: str | None = None) -> dict[str, Any]:
    import httpx

    query = query.strip()
    if not query:
        return {"error": "Search query is required."}

    api_key = get_jina_api_key()
    if not api_key:
        return {"error": "Web search is not configured. JINA_API_KEY is not set."}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-No-Cache": "true",
    }

    if site:
        headers["X-Site"] = site

    try:
        async with httpx.AsyncClient(timeout=JINA_TIMEOUT) as client:
            resp = await client.post(
                JINA_SEARCH_URL,
                headers=headers,
                json={"q": query},
            )

        if resp.status_code != 200:
            logger.warning("Jina search returned %d: %s", resp.status_code, resp.text[:200])
            return {"error": f"Web search failed (status {resp.status_code})."}

        data = resp.json()
        results = []
        for item in (data.get("data") or [])[:max_results]:
            content = (item.get("content") or "")[:JINA_MAX_CONTENT_LEN]
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
                "content": content,
            })

        return {
            "query": query,
            "result_count": len(results),
            "results": results,
        }

    except Exception as exc:
        logger.error("web_search failed: %s", exc)
        return {"error": "Web search request failed. Please try again."}


async def fetch_url(url: str) -> dict[str, Any]:
    import httpx

    url = url.strip()
    if not url:
        return {"error": "URL is required."}

    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    api_key = get_jina_api_key()
    if not api_key:
        return {"error": "URL fetching is not configured. JINA_API_KEY is not set."}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "X-No-Cache": "true",
    }

    try:
        reader_url = f"{JINA_READER_URL}{url}"
        async with httpx.AsyncClient(timeout=JINA_TIMEOUT) as client:
            resp = await client.get(reader_url, headers=headers)

        if resp.status_code != 200:
            logger.warning("Jina reader returned %d for %s", resp.status_code, url)
            return {"error": f"Failed to fetch URL (status {resp.status_code})."}

        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            data = resp.json()
            page_data = data.get("data", {})
            content = (page_data.get("content") or "")[:JINA_MAX_CONTENT_LEN * 2]
            title = page_data.get("title", "")
        else:
            content = resp.text[:JINA_MAX_CONTENT_LEN * 2]
            title = ""

        return {
            "url": url,
            "title": title,
            "content": content,
            "content_length": len(content),
        }

    except Exception as exc:
        logger.error("fetch_url failed for %s: %s", url, exc)
        return {"error": "Failed to fetch URL. Please try again."}
