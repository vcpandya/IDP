"""Summarization — LLM-based text and tree summarization."""

import logging
from typing import Optional

from idpkit.core.llm import LLMClient

logger = logging.getLogger(__name__)

# Length presets for summarization.
_LENGTH_PROMPTS = {
    "brief": "Provide a very brief summary in 1-2 sentences.",
    "standard": "Provide a clear and concise summary in one paragraph (3-5 sentences).",
    "detailed": "Provide a detailed summary covering all key points, organized in multiple paragraphs.",
}


async def summarize_text(
    text: str,
    llm: LLMClient,
    length: str = "standard",
) -> str:
    """Summarize a block of text using the LLM.

    Parameters
    ----------
    text:
        The text content to summarize.
    llm:
        An :class:`LLMClient` instance.
    length:
        One of ``"brief"``, ``"standard"``, or ``"detailed"``.  Controls
        the expected summary length.

    Returns
    -------
    str
        The generated summary.
    """
    length_instruction = _LENGTH_PROMPTS.get(length, _LENGTH_PROMPTS["standard"])

    prompt = (
        "Summarize the following text. "
        f"{length_instruction}\n\n"
        "Text:\n"
        "---\n"
        f"{_truncate(text, 14000)}\n"
        "---\n\n"
        "Summary:"
    )

    response = await llm.acomplete(prompt)
    summary = response.content.strip()

    logger.info(
        "Summarized %d chars of text (length=%s) -> %d chars",
        len(text), length, len(summary),
    )
    return summary


async def summarize_tree(
    tree_index: dict,
    llm: LLMClient,
    depth: int = 2,
) -> dict:
    """Summarize a tree index, generating summaries for nodes up to *depth*.

    Parameters
    ----------
    tree_index:
        A dict with ``"doc_name"`` and ``"structure"`` (list of tree nodes).
    llm:
        An :class:`LLMClient` instance.
    depth:
        Maximum depth of nodes to summarize (1 = top-level only).

    Returns
    -------
    dict
        A new dict mirroring the tree structure with added ``"summary"``
        fields.  The original tree_index is not mutated.
    """
    import copy
    result = copy.deepcopy(tree_index)

    structure = result.get("structure", [])
    for node in structure:
        await _summarize_node(node, llm, current_depth=1, max_depth=depth)

    # Generate an overall document summary.
    section_titles = [n.get("title", "") for n in structure]
    section_summaries = [n.get("summary", "") for n in structure if n.get("summary")]

    if section_summaries:
        overview_prompt = (
            f"This document is titled '{result.get('doc_name', 'Document')}'.\n"
            "It has the following sections and summaries:\n\n"
        )
        for title, summary in zip(section_titles, section_summaries):
            overview_prompt += f"- **{title}**: {summary}\n"
        overview_prompt += (
            "\nProvide a concise overall summary of this document in 2-3 sentences."
        )

        response = await llm.acomplete(overview_prompt)
        result["doc_description"] = response.content.strip()

    logger.info(
        "Tree summarization complete for '%s' (depth=%d)",
        result.get("doc_name", "?"), depth,
    )
    return result


async def _summarize_node(
    node: dict,
    llm: LLMClient,
    current_depth: int,
    max_depth: int,
) -> None:
    """Recursively summarize a single tree node in-place."""
    if current_depth > max_depth:
        return

    # Summarize this node if it has text content.
    text = node.get("text", "")
    children = node.get("nodes", [])

    # Collect child summaries first (bottom-up).
    for child in children:
        await _summarize_node(child, llm, current_depth + 1, max_depth)

    # Build content for this node's summary.
    content_parts = []
    if text:
        content_parts.append(text)

    # Include child titles and summaries.
    for child in children:
        child_summary = child.get("summary", "")
        child_title = child.get("title", "")
        if child_summary:
            content_parts.append(f"{child_title}: {child_summary}")

    if not content_parts:
        return

    combined = "\n".join(content_parts)
    if len(combined) < 50:
        # Too short to summarize meaningfully.
        node["summary"] = combined
        return

    prompt = (
        f"Summarize the following section titled '{node.get('title', 'Section')}'.\n"
        "Provide a concise summary in 1-2 sentences.\n\n"
        f"{_truncate(combined, 6000)}\n\n"
        "Summary:"
    )

    response = await llm.acomplete(prompt)
    node["summary"] = response.content.strip()


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to *max_chars*, appending an ellipsis if trimmed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"
