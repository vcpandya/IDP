"""Markdown indexer adapter wrapping the md_to_tree engine."""

import logging
from typing import Any

from .base import BaseIndexer

logger = logging.getLogger(__name__)


class MarkdownIndexer(BaseIndexer):
    """Indexer for Markdown documents.

    Delegates to :func:`idpkit.engine.page_index_md.md_to_tree`, which is
    already an ``async`` function, so we simply ``await`` it directly.
    """

    def supported_formats(self) -> list[str]:
        return [".md", ".markdown"]

    async def build_index(self, source: Any, **options) -> dict:
        """Build a tree index from a Markdown file path.

        Parameters
        ----------
        source:
            A ``str`` file path pointing at a ``.md`` or ``.markdown`` file.
        **options:
            Keyword arguments forwarded to :func:`md_to_tree`.  Recognised
            keys include ``if_thinning``, ``min_token_threshold``,
            ``if_add_node_summary``, ``summary_token_threshold``, ``model``,
            ``if_add_doc_description``, ``if_add_node_text``,
            ``if_add_node_id``.
        """
        if not isinstance(source, str):
            raise TypeError(
                f"MarkdownIndexer expects a file path (str), got {type(source).__name__}"
            )

        # Lazy import to keep startup fast and avoid circular imports.
        from idpkit.engine.page_index_md import md_to_tree

        # Map engine config-style option names to md_to_tree parameter names.
        # md_to_tree accepts these keyword arguments directly:
        #   md_path, if_thinning, min_token_threshold,
        #   if_add_node_summary, summary_token_threshold, model,
        #   if_add_doc_description, if_add_node_text, if_add_node_id
        md_options = _extract_md_options(options)

        logger.info("Starting Markdown indexing for source=%s", source)

        result = await md_to_tree(source, **md_options)

        logger.info(
            "Markdown indexing complete — doc_name=%s, sections=%d",
            result.get("doc_name", "?"),
            len(result.get("structure", [])),
        )
        return result


# Keys that md_to_tree accepts (excluding the positional md_path).
_MD_TO_TREE_PARAMS = frozenset({
    "if_thinning",
    "min_token_threshold",
    "if_add_node_summary",
    "summary_token_threshold",
    "model",
    "if_add_doc_description",
    "if_add_node_text",
    "if_add_node_id",
})


def _extract_md_options(options: dict) -> dict:
    """Filter *options* down to the keys that md_to_tree understands."""
    return {k: v for k, v in options.items() if k in _MD_TO_TREE_PARAMS}
