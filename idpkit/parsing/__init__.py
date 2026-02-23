"""IDP Kit Parsing — Document text and content extraction.

Provides parsers for multiple document formats.  Each parser implements
:class:`BaseParser` and returns a :class:`ParseResult`.

Usage::

    from idpkit.parsing import get_parser

    parser = get_parser("pdf")
    result = parser.parse("/path/to/doc.pdf")
"""

from .base import BaseParser, ParseResult
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .html_parser import HTMLParser
from .spreadsheet_parser import SpreadsheetParser
from .pptx_parser import PPTXParser
from .image_parser import ImageParser

__all__ = [
    "BaseParser",
    "ParseResult",
    "PDFParser",
    "DOCXParser",
    "HTMLParser",
    "SpreadsheetParser",
    "PPTXParser",
    "ImageParser",
    "get_parser",
]

# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

# Build a lookup table from file extension to parser instance.  Each parser
# is instantiated once (they are stateless).
_PARSERS: list[BaseParser] = [
    PDFParser(),
    DOCXParser(),
    HTMLParser(),
    SpreadsheetParser(),
    PPTXParser(),
    ImageParser(),
]

_REGISTRY: dict[str, BaseParser] = {}
for _parser in _PARSERS:
    for _ext in _parser.supported_extensions():
        _REGISTRY[_ext.lower()] = _parser


def get_parser(ext: str) -> BaseParser:
    """Return the appropriate parser for a file extension.

    Parameters
    ----------
    ext:
        A file extension **without** the leading dot, e.g. ``"pdf"`` or
        ``"docx"``.  The lookup is case-insensitive.

    Raises
    ------
    ValueError
        If no parser is registered for the given extension.
    """
    ext_lower = ext.lower().lstrip(".")
    parser = _REGISTRY.get(ext_lower)
    if parser is None:
        supported = sorted(_REGISTRY.keys())
        raise ValueError(
            f"No parser registered for extension {ext!r}. "
            f"Supported extensions: {supported}"
        )
    return parser
