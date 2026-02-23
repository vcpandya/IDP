"""Markdown generator — converts tree indices to structured markdown."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def generate_markdown(tree_index: dict, options: dict = None) -> str:
    """Convert a tree index into a well-structured markdown document.

    Parameters
    ----------
    tree_index:
        A dict with ``"doc_name"``, ``"doc_description"`` (optional), and
        ``"structure"`` (list of tree nodes).
    options:
        Optional configuration dict.  Recognised keys:

        - ``include_summaries`` (bool, default True): Include node summaries.
        - ``include_text`` (bool, default False): Include full node text.
        - ``max_depth`` (int, default 0 = unlimited): Max heading depth.
        - ``heading_offset`` (int, default 1): Starting heading level.

    Returns
    -------
    str
        The generated markdown string.
    """
    opts = options or {}
    include_summaries = opts.get("include_summaries", True)
    include_text = opts.get("include_text", False)
    max_depth = opts.get("max_depth", 0)
    heading_offset = opts.get("heading_offset", 1)

    parts: list[str] = []

    # Document title
    doc_name = tree_index.get("doc_name", "Document")
    parts.append(f"{'#' * heading_offset} {doc_name}")
    parts.append("")

    # Document description
    doc_desc = tree_index.get("doc_description")
    if doc_desc:
        parts.append(doc_desc)
        parts.append("")

    # Render structure
    structure = tree_index.get("structure", [])
    for node in structure:
        _render_node(
            node,
            depth=1,
            heading_offset=heading_offset,
            include_summaries=include_summaries,
            include_text=include_text,
            max_depth=max_depth,
            parts=parts,
        )

    result = "\n".join(parts)
    logger.info("Markdown generated: %d characters", len(result))
    return result


def _render_node(
    node: dict,
    depth: int,
    heading_offset: int,
    include_summaries: bool,
    include_text: bool,
    max_depth: int,
    parts: list[str],
) -> None:
    """Recursively render a tree node as markdown."""
    if max_depth and depth > max_depth:
        return

    level = heading_offset + depth
    # Cap at h6
    level = min(level, 6)

    title = node.get("title", "Untitled")
    parts.append(f"{'#' * level} {title}")
    parts.append("")

    # Page range
    start = node.get("start_index")
    end = node.get("end_index")
    if start is not None and end is not None:
        if start == end:
            parts.append(f"*Page {start}*")
        else:
            parts.append(f"*Pages {start}-{end}*")
        parts.append("")

    # Summary
    if include_summaries:
        summary = node.get("summary") or node.get("prefix_summary")
        if summary:
            parts.append(summary)
            parts.append("")

    # Full text
    if include_text:
        text = node.get("text")
        if text:
            parts.append(text)
            parts.append("")

    # Children
    children = node.get("nodes", [])
    for child in children:
        _render_node(
            child,
            depth=depth + 1,
            heading_offset=heading_offset,
            include_summaries=include_summaries,
            include_text=include_text,
            max_depth=max_depth,
            parts=parts,
        )
