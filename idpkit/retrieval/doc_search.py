"""Multi-document search — search across several document tree indices.

Given a list of documents (each carrying its own ``tree_index``), this
module runs ``tree_search`` on each document in parallel, then ranks and
merges the results into a unified list.
"""

import asyncio
import logging
from typing import Optional

from idpkit.core.llm import LLMClient
from .tree_search import tree_search

logger = logging.getLogger(__name__)


def _score_node(node: dict) -> float:
    """Heuristic relevance score for ranking merged results.

    Nodes found at shallower depths (closer to the root) that were still
    selected as relevant are assumed to cover broader and potentially more
    important content.  Nodes with summaries or text are preferred over
    empty ones.
    """
    score = 1.0

    # Prefer shallower nodes (depth 0 is most general)
    depth = node.get("_depth", 0)
    score += max(0, 5 - depth) * 0.2  # bonus for shallow nodes

    # Prefer nodes with actual content
    if node.get("text"):
        score += 0.5
    if node.get("summary"):
        score += 0.3

    # Prefer nodes with wider page ranges (more substantial sections)
    start = node.get("start_index")
    end = node.get("end_index")
    if start is not None and end is not None:
        page_span = max(end - start, 0)
        score += min(page_span * 0.05, 0.5)

    return score


async def multi_doc_search(
    documents: list,
    query: str,
    llm: LLMClient,
    model: Optional[str] = None,
    max_results: int = 20,
    graph_augment: bool = False,
    db=None,
) -> list[dict]:
    """Search across multiple document tree indices.

    Each item in *documents* should be a dict (or object) with at least:
    - ``id``: document identifier
    - ``filename``: human-readable name
    - ``tree_index``: the tree index dict (with ``structure`` key)

    Args:
        documents: List of document dicts/objects with ``tree_index``.
        query: The user's natural language query.
        llm: ``LLMClient`` instance.
        model: Optional model override.
        max_results: Maximum number of merged results to return.
        graph_augment: If True, expand results with graph-linked nodes.
        db: Async DB session, required when ``graph_augment=True``.

    Returns:
        A list of result dicts, each augmented with ``_doc_id`` and
        ``_doc_filename`` so the caller knows which document the node
        came from.  Results are sorted by descending relevance score.
    """
    if not documents:
        return []

    async def _search_one(doc) -> list[dict]:
        """Run tree_search on a single document and tag results."""
        # Support both dict-like and ORM-object access
        if isinstance(doc, dict):
            doc_id = doc.get("id", "")
            filename = doc.get("filename", "")
            tree_idx = doc.get("tree_index")
        else:
            doc_id = getattr(doc, "id", "")
            filename = getattr(doc, "filename", "")
            tree_idx = getattr(doc, "tree_index", None)

        if not tree_idx or not tree_idx.get("structure"):
            logger.debug("Skipping document %s — no tree index", doc_id)
            return []

        try:
            nodes = await tree_search(
                tree_idx,
                query,
                llm,
                model=model,
                graph_augment=graph_augment,
                document_id=doc_id,
                db=db,
            )
        except Exception as exc:
            logger.error("tree_search failed for doc %s: %s", doc_id, exc)
            return []

        # Tag each result with its source document
        for node in nodes:
            node["_doc_id"] = doc_id
            node["_doc_filename"] = filename

        return nodes

    # Run searches concurrently across all documents
    tasks = [_search_one(doc) for doc in documents]
    all_results_nested = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten and filter out exceptions
    merged: list[dict] = []
    for result in all_results_nested:
        if isinstance(result, Exception):
            logger.error("multi_doc_search task failed: %s", result)
            continue
        merged.extend(result)

    # Sort by heuristic relevance score (descending)
    merged.sort(key=_score_node, reverse=True)

    # Trim to max_results
    merged = merged[:max_results]

    logger.info(
        "multi_doc_search returned %d results across %d documents for query: %s",
        len(merged),
        len(documents),
        query[:80],
    )
    return merged
