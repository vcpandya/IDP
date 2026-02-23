"""IDP Kit Indexing — Document indexer adapters.

Provides a unified interface for building tree-structured indices from
different document formats (PDF, Markdown, DOCX, HTML, spreadsheets,
presentations, images, etc.).

Usage::

    from idpkit.indexing import get_indexer

    indexer = get_indexer(".pdf")
    result = await indexer.build_index("/path/to/doc.pdf", model="gpt-4o")
"""

from .base import BaseIndexer
from .markdown_indexer import MarkdownIndexer
from .pdf_indexer import PDFIndexer
from .docx_indexer import DOCXIndexer
from .html_indexer import HTMLIndexer
from .generic_indexer import GenericIndexer

__all__ = [
    "BaseIndexer",
    "PDFIndexer",
    "MarkdownIndexer",
    "DOCXIndexer",
    "HTMLIndexer",
    "GenericIndexer",
    "get_indexer",
]

# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

# Build a lookup table from file extension to indexer instance.  Each indexer
# is instantiated once (they are stateless).
_INDEXERS: list[BaseIndexer] = [
    PDFIndexer(),
    MarkdownIndexer(),
    DOCXIndexer(),
    HTMLIndexer(),
    GenericIndexer(),
]

_REGISTRY: dict[str, BaseIndexer] = {}
for _indexer in _INDEXERS:
    for _ext in _indexer.supported_formats():
        _REGISTRY[_ext.lower()] = _indexer


def get_indexer(fmt: str) -> BaseIndexer:
    """Return the appropriate indexer for a file extension.

    Parameters
    ----------
    fmt:
        A file extension **including** the leading dot, e.g. ``".pdf"`` or
        ``".md"``.  The lookup is case-insensitive.

    Raises
    ------
    ValueError
        If no indexer is registered for the given format.
    """
    fmt_lower = fmt.lower()
    indexer = _REGISTRY.get(fmt_lower)
    if indexer is None:
        supported = sorted(_REGISTRY.keys())
        raise ValueError(
            f"No indexer registered for format {fmt!r}. "
            f"Supported formats: {supported}"
        )
    return indexer
