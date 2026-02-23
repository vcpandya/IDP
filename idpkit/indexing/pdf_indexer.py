"""PDF indexer adapter wrapping the PageIndex engine."""

import asyncio
import logging
from io import BytesIO
from typing import Any

from .base import BaseIndexer

logger = logging.getLogger(__name__)


class PDFIndexer(BaseIndexer):
    """Indexer for PDF documents.

    Delegates to :func:`idpkit.engine.page_index.page_index_main`, which is
    synchronous and internally calls ``asyncio.run``.  We therefore execute it
    inside a thread pool so the caller's event loop stays free.
    """

    def supported_formats(self) -> list[str]:
        return [".pdf"]

    async def build_index(self, source: Any, **options) -> dict:
        """Build a tree index from a PDF file path or BytesIO object.

        Parameters
        ----------
        source:
            Either a ``str`` file path pointing at a ``.pdf`` file or a
            ``BytesIO`` object containing the PDF bytes.
        **options:
            Keys matching the engine's ``config.yaml`` (e.g. ``model``,
            ``if_add_node_summary``, ``toc_check_page_num``).  They are
            merged with the defaults by :class:`ConfigLoader`.
        """
        # Validate source type early so we get a clear error.
        if not isinstance(source, (str, BytesIO)):
            raise TypeError(
                f"PDFIndexer expects a file path (str) or BytesIO, got {type(source).__name__}"
            )

        # Lazy import to avoid circular-import issues at module level and to
        # keep the import cost out of application startup.
        from idpkit.engine.page_index import page_index_main
        from idpkit.engine.utils import ConfigLoader

        # Build the config namespace the engine expects.
        opt = ConfigLoader().load(options if options else None)

        logger.info("Starting PDF indexing for source=%s", _source_label(source))

        # page_index_main is synchronous (it calls asyncio.run internally),
        # so we must run it in a thread to avoid blocking the event loop and
        # to avoid "cannot call asyncio.run() while another loop is running".
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, page_index_main, source, opt)

        logger.info(
            "PDF indexing complete — doc_name=%s, sections=%d",
            result.get("doc_name", "?"),
            len(result.get("structure", [])),
        )
        return result


def _source_label(source: Any) -> str:
    """Return a short human-readable label for logging."""
    if isinstance(source, str):
        return source
    if isinstance(source, BytesIO):
        return f"<BytesIO {len(source.getvalue())} bytes>"
    return repr(source)
