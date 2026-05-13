"""Gemini Deep Research tool wrapper.

Wraps the ``google-genai`` SDK's Interactions API (Deep Research agent) so
IDA can dispatch a long-running research task and get a cited report back.

Notes
-----
* Deep Research is exclusively available through the Interactions API and
  is asynchronous — tasks can take several minutes. We poll with a hard
  timeout cap so the agent loop can never hang indefinitely.
* The user's preferred env var is ``GOOGLE_API_KEY``. The SDK also accepts
  ``GEMINI_API_KEY``; we honor either, preferring ``GOOGLE_API_KEY``.
* This tool intentionally runs a single-shot research (no collaborative
  planning) — IDA can re-invoke it with a refined prompt if needed.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Default model — speed-optimized variant per the developer guide.
_DEFAULT_AGENT = "deep-research-preview-04-2026"
_MAX_AGENT = "deep-research-max-preview-04-2026"

# Hard cap on total polling time. Deep Research can take several minutes;
# we keep this conservative to avoid wedging the agent loop. If the task
# isn't done in time we return a structured "still_running" result so IDA
# can tell the user to retry rather than crash.
_POLL_TIMEOUT_SECONDS = 300
_POLL_INTERVAL_SECONDS = 8


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
    # Pass key explicitly so we don't rely on env-var auto-pickup ordering.
    return genai.Client(api_key=api_key)


def _outputs_to_text(outputs: Any) -> str:
    """Flatten the SDK's outputs list into a single readable string."""
    if not outputs:
        return ""
    parts: list[str] = []
    for o in outputs:
        otype = getattr(o, "type", None)
        text = getattr(o, "text", None)
        if otype == "text" and text:
            parts.append(text)
        elif text:
            parts.append(text)
    return "\n\n".join(p for p in parts if p)


def _outputs_to_artifacts(outputs: Any) -> list[dict]:
    """Pull out non-text outputs (charts/infographics) as a small summary."""
    if not outputs:
        return []
    artifacts: list[dict] = []
    for o in outputs:
        otype = getattr(o, "type", None)
        if otype and otype != "text":
            entry: dict[str, Any] = {"type": otype}
            for attr in ("title", "caption", "mime_type", "format"):
                v = getattr(o, attr, None)
                if v:
                    entry[attr] = v
            artifacts.append(entry)
    return artifacts


def _run_deep_research_blocking(
    prompt: str,
    agent: str,
    visualization: str | None,
) -> dict[str, Any]:
    """Synchronous polling helper, executed in a worker thread."""
    import time

    client = _get_client()

    agent_config: dict[str, Any] = {"type": "deep-research"}
    if visualization in ("auto", "on"):
        agent_config["visualization"] = "auto"

    interaction = client.interactions.create(
        agent=agent,
        input=prompt,
        agent_config=agent_config,
        background=True,
    )
    interaction_id = getattr(interaction, "id", None)
    if not interaction_id:
        return {
            "error": "Deep Research did not return an interaction id.",
            "success": False,
        }

    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    last_status = "pending"
    while time.monotonic() < deadline:
        try:
            current = client.interactions.get(interaction_id)
        except Exception as exc:
            logger.warning("Deep Research poll failed: %s", exc)
            time.sleep(_POLL_INTERVAL_SECONDS)
            continue

        last_status = getattr(current, "status", "unknown") or "unknown"

        if last_status == "completed":
            outputs = getattr(current, "outputs", None) or []
            return {
                "success": True,
                "interaction_id": interaction_id,
                "agent": agent,
                "status": "completed",
                "report": _outputs_to_text(outputs),
                "artifacts": _outputs_to_artifacts(outputs),
            }
        if last_status == "failed":
            err = getattr(current, "error", None)
            return {
                "success": False,
                "interaction_id": interaction_id,
                "agent": agent,
                "status": "failed",
                "error": f"Deep Research failed: {err}",
            }
        time.sleep(_POLL_INTERVAL_SECONDS)

    return {
        "success": False,
        "interaction_id": interaction_id,
        "agent": agent,
        "status": last_status,
        "error": (
            f"Deep Research did not finish within {_POLL_TIMEOUT_SECONDS}s "
            f"(last status: {last_status}). Tell the user the task is still "
            f"running on Google's side; they can retry the question later."
        ),
    }


async def deep_research(
    prompt: str,
    use_max: bool = False,
    visualization: str | None = None,
) -> dict[str, Any]:
    """Run a Gemini Deep Research task and return the final report.

    Args:
        prompt: The natural-language research question.
        use_max: If True, use the higher-comprehensiveness Max variant.
        visualization: Pass ``"auto"`` to let the agent generate charts.

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
        return await asyncio.to_thread(
            _run_deep_research_blocking, prompt, agent, visualization,
        )
    except RuntimeError as exc:
        return {"error": str(exc), "success": False}
    except Exception as exc:
        logger.error("Deep Research failed: %s", exc)
        return {"error": f"Deep Research call failed: {exc}", "success": False}
