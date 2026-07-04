"""Gemini Deep Research tool wrapper.

Wraps the ``google-genai`` SDK's Interactions API (Deep Research agent) so
IDA can dispatch a long-running research task and get a cited report back.

Notes
-----
* Deep Research is exclusively available through the Interactions API and
  is asynchronous — tasks can take many minutes. We poll with a generous
  timeout cap and surface progress to the caller via an optional
  ``progress_cb`` so the chat UI can show live updates instead of staring
  at a spinner for 10+ minutes.
* The user's preferred env var is ``GOOGLE_API_KEY``. The SDK also accepts
  ``GEMINI_API_KEY``; we honor either, preferring ``GOOGLE_API_KEY``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Default model — speed-optimized variant per the developer guide.
_DEFAULT_AGENT = "deep-research-preview-04-2026"
_MAX_AGENT = "deep-research-max-preview-04-2026"

# Generous polling cap. Deep Research can legitimately take 10+ minutes;
# 5 minutes was too tight. We keep it bounded so an indefinitely-stuck
# task can never wedge the agent loop forever.
_POLL_TIMEOUT_SECONDS = 1800           # 30 minutes
_POLL_INTERVAL_SECONDS = 6

# Type alias for the optional progress sink. Receives a short, human-readable
# message describing what the research agent is currently doing.
ProgressCallback = Callable[[str], Awaitable[None]]


def _resolve_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def _get_client():
    """Lazy-import google-genai so missing dep doesn't break server boot."""
    try:
        from google import genai  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "google-genai is not installed. Run `pip install google-genai`."
        ) from exc
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. "
            "Create a key at https://aistudio.google.com/apikey."
        )
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Step / output extraction helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, n: int = 140) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def _summarize_step(step: Any) -> str | None:
    """Convert a Deep Research step into a short progress line.

    Returns ``None`` for step types we don't want to surface (e.g. the
    initial ``user_input`` echo).
    """
    stype = getattr(step, "type", None)
    if not stype or stype == "user_input":
        return None

    if stype == "thought":
        # Summary is a list of {text: ...} entries
        summaries = getattr(step, "summary", None) or []
        for s in summaries:
            text = getattr(s, "text", None) or (
                s.get("text") if isinstance(s, dict) else None
            )
            if text:
                return f"💭 {_truncate(text)}"
        return "💭 Thinking…"

    if stype == "google_search_call":
        args = getattr(step, "arguments", None)
        queries = getattr(args, "queries", None) if args else None
        if queries:
            joined = " · ".join(q for q in queries if q)[:160]
            return f"🔎 Searching: {joined}"
        return "🔎 Searching the web…"

    if stype == "google_search_result":
        return "🔎 Got search results"

    if stype == "url_context_call":
        args = getattr(step, "arguments", None)
        urls = getattr(args, "urls", None) if args else None
        if urls:
            count = len(urls)
            first = urls[0]
            extra = f" (+{count - 1} more)" if count > 1 else ""
            return f"📄 Reading {_truncate(first, 100)}{extra}"
        return "📄 Reading sources…"

    if stype == "url_context_result":
        return "📄 Read sources"

    if stype == "code_execution_call":
        return "🧮 Running analysis code…"

    if stype == "code_execution_result":
        return "🧮 Code analysis done"

    if stype == "function_call":
        name = getattr(step, "name", None) or "tool"
        return f"🛠 Calling {name}…"

    if stype == "function_result":
        return "🛠 Tool returned"

    if stype == "model_output":
        return "✍️ Synthesizing the report…"

    # Unknown step type — surface its discriminator so we can learn what
    # else exists in the wild without spamming the UI.
    return f"… {stype}"


def _outputs_to_text(steps: Any) -> str:
    """Flatten the model_output steps into a single readable string."""
    if not steps:
        return ""
    parts: list[str] = []
    for s in steps:
        if getattr(s, "type", None) != "model_output":
            continue
        content_list = getattr(s, "content", None) or []
        for c in content_list:
            ctype = getattr(c, "type", None)
            text = getattr(c, "text", None)
            if text and (ctype is None or ctype == "text"):
                parts.append(text)
    return "\n\n".join(p for p in parts if p)


def _outputs_to_artifacts(steps: Any) -> list[dict]:
    """Pull out non-text outputs (charts/infographics) as a small summary."""
    if not steps:
        return []
    artifacts: list[dict] = []
    for s in steps:
        if getattr(s, "type", None) != "model_output":
            continue
        content_list = getattr(s, "content", None) or []
        for c in content_list:
            ctype = getattr(c, "type", None)
            if ctype and ctype != "text":
                entry: dict[str, Any] = {"type": ctype}
                for attr in ("title", "caption", "mime_type", "format"):
                    v = getattr(c, attr, None)
                    if v:
                        entry[attr] = v
                artifacts.append(entry)
    return artifacts


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

async def deep_research(
    prompt: str,
    use_max: bool = False,
    visualization: str | None = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Run a Gemini Deep Research task and return the final report.

    Args:
        prompt: The natural-language research question.
        use_max: If True, use the higher-comprehensiveness Max variant.
        visualization: Pass ``"auto"`` to let the agent generate charts.
        progress_cb: Optional async callable invoked with a short status
            string each time the underlying interaction reports a new step
            (e.g. a thought, a search query, a URL fetch). Failures inside
            the callback are swallowed so they can never break research.

    Returns:
        ``{"success": True, "report": "...", "artifacts": [...], ...}`` on
        success, otherwise an ``{"error": "...", "success": False}``-shaped
        dict that IDA can surface to the user.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return {"error": "deep_research requires a non-empty prompt.", "success": False}

    agent = _MAX_AGENT if use_max else _DEFAULT_AGENT

    try:
        client = _get_client()
    except RuntimeError as exc:
        return {"error": str(exc), "success": False}

    agent_config: dict[str, Any] = {"type": "deep-research"}
    if visualization in ("auto", "on"):
        agent_config["visualization"] = "auto"

    async def _emit(msg: str) -> None:
        if not progress_cb or not msg:
            return
        try:
            await progress_cb(msg)
        except Exception:
            logger.debug("deep_research progress_cb raised", exc_info=True)

    try:
        await _emit(f"Starting Deep Research ({agent})…")
        interaction = await asyncio.to_thread(
            client.interactions.create,
            agent=agent,
            input=prompt,
            agent_config=agent_config,
            background=True,
        )
    except Exception as exc:
        logger.error("Deep Research create failed: %s", exc)
        return {"error": f"Deep Research call failed: {exc}", "success": False}

    interaction_id = getattr(interaction, "id", None)
    if not interaction_id:
        return {
            "error": "Deep Research did not return an interaction id.",
            "success": False,
        }

    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    seen_steps = 0
    last_status = "in_progress"
    started = time.monotonic()

    while time.monotonic() < deadline:
        try:
            current = await asyncio.to_thread(client.interactions.get, interaction_id)
        except Exception as exc:
            logger.warning("Deep Research poll failed: %s", exc)
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            continue

        last_status = getattr(current, "status", "in_progress") or "in_progress"
        steps = getattr(current, "steps", None) or []

        # Emit progress for any newly-arrived steps
        if len(steps) > seen_steps:
            for s in steps[seen_steps:]:
                msg = _summarize_step(s)
                if msg:
                    await _emit(msg)
            seen_steps = len(steps)

        if last_status == "completed":
            await _emit("✅ Research complete — finalizing report…")
            return {
                "success": True,
                "interaction_id": interaction_id,
                "agent": agent,
                "status": "completed",
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "report": _outputs_to_text(steps),
                "artifacts": _outputs_to_artifacts(steps),
            }
        if last_status in ("failed", "cancelled", "incomplete"):
            return {
                "success": False,
                "interaction_id": interaction_id,
                "agent": agent,
                "status": last_status,
                "error": f"Deep Research ended with status '{last_status}'.",
            }

        await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    return {
        "success": False,
        "interaction_id": interaction_id,
        "agent": agent,
        "status": last_status,
        "error": (
            f"Deep Research did not finish within {_POLL_TIMEOUT_SECONDS}s "
            f"(last status: {last_status}). The task is still running on "
            f"Google's side; the user can retry the question later."
        ),
    }
