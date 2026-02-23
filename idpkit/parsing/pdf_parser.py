"""PDF parser using PyMuPDF (fitz) for text extraction."""

import logging

from .base import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """Extract text and metadata from PDF documents.

    Uses PyMuPDF (``fitz``), which is already a project dependency.  Text is
    extracted page-by-page and aggregated into a single :class:`ParseResult`.
    """

    def supported_extensions(self) -> list[str]:
        return ["pdf"]

    def parse(self, file_path: str) -> ParseResult:
        """Parse a PDF file and return its text content.

        Parameters
        ----------
        file_path:
            Path to the ``.pdf`` file.

        Raises
        ------
        ImportError
            If PyMuPDF is not installed.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF parsing. "
                "Install it with: pip install PyMuPDF"
            )

        logger.info("Parsing PDF: %s", file_path)

        doc = fitz.open(file_path)
        pages: list[dict] = []
        all_text_parts: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            pages.append({"page": page_num + 1, "text": text})
            all_text_parts.append(text)

        # Extract metadata from the PDF info dict.
        raw_meta = doc.metadata or {}
        metadata = {
            "title": raw_meta.get("title", ""),
            "author": raw_meta.get("author", ""),
            "subject": raw_meta.get("subject", ""),
            "creator": raw_meta.get("creator", ""),
            "producer": raw_meta.get("producer", ""),
            "creation_date": raw_meta.get("creationDate", ""),
            "modification_date": raw_meta.get("modDate", ""),
            "format": raw_meta.get("format", ""),
        }
        # Remove empty metadata fields.
        metadata = {k: v for k, v in metadata.items() if v}

        page_count = len(doc)
        doc.close()

        full_text = "\n\n".join(all_text_parts)

        logger.info("PDF parsing complete: %d pages extracted", page_count)

        return ParseResult(
            text=full_text,
            pages=pages,
            metadata=metadata,
            page_count=page_count,
        )
