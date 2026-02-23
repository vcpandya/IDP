"""Generic indexer for spreadsheets, presentations, and images."""

import logging
import os
from typing import Any

from .base import BaseIndexer

logger = logging.getLogger(__name__)


class GenericIndexer(BaseIndexer):
    """Generic indexer for document formats that lack heading-based hierarchy.

    Builds a simple flat tree where each "page" (sheet, slide, or image)
    becomes a child node under a single root.  Supports spreadsheets (XLSX,
    XLS, CSV), presentations (PPTX, PPT), and images (PNG, JPG, etc.).
    """

    def supported_formats(self) -> list[str]:
        return [
            ".xlsx", ".xls", ".csv",
            ".pptx", ".ppt",
            ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif",
        ]

    async def build_index(self, source: Any, **options) -> dict:
        """Build a simple tree index from the given file.

        Parameters
        ----------
        source:
            A ``str`` file path.
        **options:
            Reserved for future use.
        """
        if not isinstance(source, str):
            raise TypeError(
                f"GenericIndexer expects a file path (str), got {type(source).__name__}"
            )

        ext = os.path.splitext(source)[1].lower()
        parser = _get_parser_for_ext(ext)

        logger.info("Starting generic indexing for source=%s (ext=%s)", source, ext)

        result = parser.parse(source)

        # Build one node per page/sheet/slide.
        structure: list[dict] = []
        for page_info in result.pages:
            page_num = page_info.get("page", 1)
            page_text = page_info.get("text", "")
            node = {
                "title": _page_title(ext, page_num),
                "text": page_text,
                "start_index": page_num,
                "end_index": page_num,
                "nodes": [],
            }
            structure.append(node)

        # If no pages, create a single node from the full text.
        if not structure:
            structure = [
                {
                    "title": "Content",
                    "text": result.text,
                    "nodes": [],
                }
            ]

        doc_name = os.path.splitext(os.path.basename(source))[0]

        index = {
            "doc_name": doc_name,
            "doc_description": result.metadata.get("title", ""),
            "structure": structure,
        }

        logger.info(
            "Generic indexing complete — doc_name=%s, nodes=%d",
            doc_name,
            len(structure),
        )
        return index


def _get_parser_for_ext(ext: str):
    """Return the appropriate parser instance for the given extension."""
    ext_lower = ext.lower().lstrip(".")

    # Spreadsheet formats
    if ext_lower in ("xlsx", "xls", "csv"):
        from idpkit.parsing.spreadsheet_parser import SpreadsheetParser
        return SpreadsheetParser()

    # Presentation formats
    if ext_lower in ("pptx", "ppt"):
        from idpkit.parsing.pptx_parser import PPTXParser
        return PPTXParser()

    # Image formats
    if ext_lower in ("png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp", "gif"):
        from idpkit.parsing.image_parser import ImageParser
        return ImageParser()

    raise ValueError(f"GenericIndexer has no parser for extension {ext!r}")


def _page_title(ext: str, page_num: int) -> str:
    """Generate a human-readable title for a page/sheet/slide node."""
    ext_lower = ext.lower().lstrip(".")
    if ext_lower in ("xlsx", "xls", "csv"):
        return f"Sheet {page_num}"
    if ext_lower in ("pptx", "ppt"):
        return f"Slide {page_num}"
    return f"Page {page_num}"
