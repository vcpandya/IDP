"""Document comparison — structural and content comparison via LLM."""

import logging

from idpkit.core.llm import LLMClient

logger = logging.getLogger(__name__)


async def compare_documents(
    doc1_tree: dict,
    doc2_tree: dict,
    llm: LLMClient,
) -> dict:
    """Compare two document tree indices structurally and by content.

    Parameters
    ----------
    doc1_tree:
        Tree index of the first document (must have ``"doc_name"`` and
        ``"structure"``).
    doc2_tree:
        Tree index of the second document.
    llm:
        An :class:`LLMClient` instance.

    Returns
    -------
    dict
        A comparison result with keys:

        - ``"structural_comparison"`` — differences in section hierarchy.
        - ``"content_comparison"`` — high-level content differences.
        - ``"summary"`` — overall comparison summary.
        - ``"doc1_name"`` / ``"doc2_name"`` — the document names.
    """
    doc1_name = doc1_tree.get("doc_name", "Document 1")
    doc2_name = doc2_tree.get("doc_name", "Document 2")

    # Build structural outlines for comparison.
    outline1 = _build_outline(doc1_tree.get("structure", []))
    outline2 = _build_outline(doc2_tree.get("structure", []))

    # Build content summaries for comparison.
    content1 = _build_content_summary(doc1_tree.get("structure", []))
    content2 = _build_content_summary(doc2_tree.get("structure", []))

    # --- Structural comparison ---
    structural_prompt = (
        "Compare the structure (section hierarchy) of two documents.\n\n"
        f"**{doc1_name}** structure:\n{outline1}\n\n"
        f"**{doc2_name}** structure:\n{outline2}\n\n"
        "Describe the structural differences and similarities in a few bullet points. "
        "Note any sections present in one but not the other."
    )

    structural_response = await llm.acomplete(structural_prompt)

    # --- Content comparison ---
    content_prompt = (
        "Compare the content of two documents based on their section summaries.\n\n"
        f"**{doc1_name}** content:\n{_truncate(content1, 5000)}\n\n"
        f"**{doc2_name}** content:\n{_truncate(content2, 5000)}\n\n"
        "Describe the key content differences and similarities in a few bullet points. "
        "Highlight any contradictions, additions, or removals."
    )

    content_response = await llm.acomplete(content_prompt)

    # --- Overall summary ---
    summary_prompt = (
        f"Given the following analyses of two documents ('{doc1_name}' and '{doc2_name}'), "
        "provide a brief overall comparison summary in 2-3 sentences.\n\n"
        f"Structural comparison:\n{structural_response.content}\n\n"
        f"Content comparison:\n{content_response.content}\n\n"
        "Overall summary:"
    )

    summary_response = await llm.acomplete(summary_prompt)

    result = {
        "doc1_name": doc1_name,
        "doc2_name": doc2_name,
        "structural_comparison": structural_response.content.strip(),
        "content_comparison": content_response.content.strip(),
        "summary": summary_response.content.strip(),
    }

    logger.info(
        "Document comparison complete: '%s' vs '%s'",
        doc1_name, doc2_name,
    )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_outline(structure: list[dict], indent: int = 0) -> str:
    """Build a plain-text outline from tree structure nodes."""
    lines: list[str] = []
    prefix = "  " * indent
    for node in structure:
        title = node.get("title", "Untitled")
        pages = ""
        start = node.get("start_index")
        end = node.get("end_index")
        if start is not None:
            pages = f" (p.{start}" + (f"-{end}" if end and end != start else "") + ")"
        lines.append(f"{prefix}- {title}{pages}")

        children = node.get("nodes", [])
        if children:
            lines.append(_build_outline(children, indent + 1))

    return "\n".join(lines) if lines else "(no structure)"


def _build_content_summary(structure: list[dict]) -> str:
    """Build a content summary from tree nodes using summaries and text."""
    parts: list[str] = []
    for node in structure:
        title = node.get("title", "Untitled")
        summary = node.get("summary") or node.get("prefix_summary") or ""
        text = node.get("text", "")

        content = summary or text[:300]
        if content:
            parts.append(f"**{title}**: {content}")

        children = node.get("nodes", [])
        if children:
            child_summary = _build_content_summary(children)
            if child_summary:
                parts.append(child_summary)

    return "\n".join(parts) if parts else "(no content)"


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to *max_chars*, appending an ellipsis if trimmed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"
