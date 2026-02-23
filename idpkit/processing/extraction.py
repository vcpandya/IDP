"""Entity extraction — LLM-based named entity recognition."""

import json
import logging
from typing import Optional

from idpkit.core.llm import LLMClient

logger = logging.getLogger(__name__)

# Default entity types to extract when none are specified.
DEFAULT_ENTITY_TYPES = ["person", "organization", "date", "amount", "location"]


async def extract_entities(
    text: str,
    llm: LLMClient,
    entity_types: Optional[list[str]] = None,
) -> list[dict]:
    """Extract named entities from text using an LLM.

    Parameters
    ----------
    text:
        The source text to analyse.
    llm:
        An :class:`LLMClient` instance.
    entity_types:
        List of entity types to extract.  Defaults to
        ``["person", "organization", "date", "amount", "location"]``.

    Returns
    -------
    list[dict]
        Each dict contains ``"entity"`` (str), ``"type"`` (str), and
        ``"context"`` (str, optional snippet around the entity).
    """
    types = entity_types or DEFAULT_ENTITY_TYPES
    types_str = ", ".join(types)

    prompt = (
        "Extract all named entities from the following text. "
        f"Focus on these entity types: {types_str}.\n\n"
        "Return a JSON array of objects, each with:\n"
        '  "entity": the entity text,\n'
        '  "type": one of the requested types,\n'
        '  "context": a short surrounding snippet (max 100 chars).\n\n'
        "If no entities are found, return an empty array [].\n\n"
        "Text:\n"
        "---\n"
        f"{_truncate(text, 12000)}\n"
        "---\n\n"
        "Respond with valid JSON only."
    )

    response = await llm.acomplete(prompt)
    raw = response.content.strip()

    entities = _parse_json_array(raw)

    logger.info("Extracted %d entities from %d chars of text", len(entities), len(text))
    return entities


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to *max_chars*, appending an ellipsis if trimmed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def _parse_json_array(raw: str) -> list[dict]:
    """Attempt to parse a JSON array from the LLM response."""
    # Strip markdown code fences if present.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "entities" in result:
            return result["entities"]
        return []
    except json.JSONDecodeError:
        logger.warning("Failed to parse entity extraction response as JSON")
        return []
