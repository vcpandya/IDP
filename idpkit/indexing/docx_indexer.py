"""DOCX indexer — builds a tree index from Word documents."""

import logging
import re
from typing import Any

from .base import BaseIndexer

logger = logging.getLogger(__name__)


class DOCXIndexer(BaseIndexer):
    """Indexer for DOCX documents.

    Uses :class:`~idpkit.parsing.docx_parser.DOCXParser` to extract text,
    then builds a hierarchical tree from the heading structure embedded
    in the extracted markdown-style text.
    """

    def supported_formats(self) -> list[str]:
        return [".docx", ".doc"]

    async def build_index(self, source: Any, **options) -> dict:
        """Build a tree index from a DOCX file path.

        Parameters
        ----------
        source:
            A ``str`` file path pointing at a ``.docx`` file.
        **options:
            Reserved for future use (e.g. ``model`` for LLM-based summaries).
        """
        if not isinstance(source, str):
            raise TypeError(
                f"DOCXIndexer expects a file path (str), got {type(source).__name__}"
            )

        from idpkit.parsing.docx_parser import DOCXParser

        logger.info("Starting DOCX indexing for source=%s", source)

        parser = DOCXParser()
        result = parser.parse(source)

        # Build tree from the heading hierarchy in the parsed text.
        structure = _build_tree_from_headings(result.text)

        import os
        doc_name = os.path.splitext(os.path.basename(source))[0]

        index = {
            "doc_name": doc_name,
            "doc_description": result.metadata.get("title", ""),
            "structure": structure,
        }

        logger.info(
            "DOCX indexing complete — doc_name=%s, sections=%d",
            doc_name,
            len(structure),
        )
        return index


# ---------------------------------------------------------------------------
# Heading-based tree builder
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _build_tree_from_headings(text: str) -> list[dict]:
    """Parse markdown-style headings and build a nested tree structure."""
    lines = text.split("\n")
    root_nodes: list[dict] = []
    stack: list[tuple[int, dict]] = []  # (level, node)

    current_text_parts: list[str] = []

    def _flush_text():
        """Attach accumulated text to the most recent node."""
        if current_text_parts and stack:
            node = stack[-1][1]
            existing = node.get("text", "")
            new_text = "\n".join(current_text_parts).strip()
            node["text"] = f"{existing}\n{new_text}".strip() if existing else new_text
        current_text_parts.clear()

    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if m:
            _flush_text()
            level = len(m.group(1))
            title = m.group(2).strip()
            node = {"title": title, "nodes": []}

            # Pop stack until we find a parent at a lower level.
            while stack and stack[-1][0] >= level:
                stack.pop()

            if stack:
                stack[-1][1]["nodes"].append(node)
            else:
                root_nodes.append(node)

            stack.append((level, node))
        else:
            stripped = line.strip()
            if stripped:
                current_text_parts.append(stripped)

    _flush_text()

    # If no headings were found, create a single root node.
    if not root_nodes:
        root_nodes = [{"title": "Document", "text": text.strip(), "nodes": []}]

    return root_nodes
