"""Build LLM-ready context strings from tree search results.

Takes the list of relevant nodes returned by ``tree_search`` and assembles
them into a single context string that respects a token budget.  The output
is formatted with section headers and page references so the LLM can cite
sources when generating an answer.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Rough characters-per-token estimate (conservative for English text).
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Cheap token count estimate based on character length."""
    return len(text) // _CHARS_PER_TOKEN


def _format_node_block(node: dict, index: int) -> str:
    """Format a single node into a readable context block."""
    title = node.get("title", "(untitled)")
    start = node.get("start_index")
    end = node.get("end_index")
    summary = node.get("summary") or node.get("prefix_summary") or ""
    text = node.get("text") or ""
    node_id = node.get("node_id", "")

    # Build page reference
    if start is not None and end is not None:
        page_ref = f"pages {start}-{end}"
    elif start is not None:
        page_ref = f"page {start}"
    else:
        page_ref = "page unknown"

    header = f"--- Section {index}: {title} [{page_ref}] ---"
    if node_id:
        header = f"--- Section {index} (id={node_id}): {title} [{page_ref}] ---"

    # Annotate graph-augmented nodes
    graph_annotation = ""
    if node.get("_graph_augmented"):
        related_entities = node.get("_related_entities", [])
        if related_entities:
            entity_names = ", ".join(f'"{e}"' for e in related_entities)
            graph_annotation = f"[Related via entities: {entity_names}]\n"
        else:
            graph_annotation = "[Found via knowledge graph]\n"

    # Prefer full text, fall back to summary
    body = text if text else summary

    return f"{header}\n{graph_annotation}{body}\n"


def build_context(
    nodes: list[dict],
    max_tokens: int = 4000,
    include_header: bool = True,
) -> str:
    """Build a context string from search result nodes.

    Concatenates relevant node summaries (or full text where available)
    with section headers and page references.  Stops adding content once
    the estimated token count reaches ``max_tokens``.

    Args:
        nodes: List of node dicts from ``tree_search``.
        max_tokens: Approximate upper bound on context size in tokens.
        include_header: Whether to prepend a preamble line.

    Returns:
        Formatted context string ready to be inserted into an LLM prompt.
    """
    if not nodes:
        return ""

    parts: list[str] = []
    if include_header:
        parts.append(
            "The following are relevant sections extracted from the document:\n"
        )

    current_tokens = _estimate_tokens("\n".join(parts)) if parts else 0

    for i, node in enumerate(nodes, start=1):
        block = _format_node_block(node, i)
        block_tokens = _estimate_tokens(block)

        if current_tokens + block_tokens > max_tokens:
            # Try to fit a truncated version
            remaining_chars = (max_tokens - current_tokens) * _CHARS_PER_TOKEN
            if remaining_chars > 100:
                truncated = block[:remaining_chars] + "\n[...truncated]\n"
                parts.append(truncated)
            break

        parts.append(block)
        current_tokens += block_tokens

    context = "\n".join(parts)
    logger.info(
        "Built context with %d sections, ~%d estimated tokens",
        min(len(nodes), len(parts) - (1 if include_header else 0)),
        _estimate_tokens(context),
    )
    return context
