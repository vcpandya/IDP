"""LLM-guided tree search over hierarchical document indices.

The tree index produced by `idpkit.engine.page_index` has the shape::

    {
        "doc_name": "...",
        "doc_description": "...",
        "structure": [
            {
                "node_id": "1",
                "title": "Introduction",
                "summary": "...",
                "start_index": 1,
                "end_index": 5,
                "nodes": [ ... ]   # recursive children
            },
            ...
        ]
    }

The search strategy is:
1. Present the top-level node titles/summaries to the LLM and ask which
   nodes are relevant to the query.
2. For each relevant node that has children, recurse into its ``nodes``
   subtree and repeat.
3. Collect all leaf-level (or deepest-relevant) nodes and return them
   with their metadata (title, summary, page range).
"""

import json
import logging
from typing import Optional

from idpkit.core.llm import LLMClient

logger = logging.getLogger(__name__)


def _format_node_list(nodes: list[dict]) -> str:
    """Format a list of tree nodes into a concise numbered list for the LLM."""
    lines: list[str] = []
    for i, node in enumerate(nodes):
        node_id = node.get("node_id", str(i))
        title = node.get("title", "(untitled)")
        summary = node.get("summary") or node.get("prefix_summary") or ""
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        summary_snippet = summary[:300] + "..." if len(summary) > 300 else summary
        lines.append(
            f"[{node_id}] {title} (pages {start}-{end})"
            + (f"\n    Summary: {summary_snippet}" if summary_snippet else "")
        )
    return "\n".join(lines)


def _build_selection_prompt(query: str, node_list_text: str) -> str:
    """Build the prompt that asks the LLM to select relevant nodes."""
    return f"""You are a document retrieval assistant. Given a user query and a list of document sections, identify which sections are likely to contain information relevant to the query.

User query: {query}

Document sections:
{node_list_text}

Return a JSON object with a single key "relevant_ids" whose value is a list of the node IDs (the values in square brackets) that are relevant to the query.
If none are relevant, return {{"relevant_ids": []}}.

Return ONLY the JSON object, no other text."""


def _extract_relevant_ids(response_text: str) -> list[str]:
    """Parse the LLM response to extract relevant node IDs."""
    # Try to parse as JSON directly
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        ids = data.get("relevant_ids", [])
        return [str(rid) for rid in ids]
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse LLM node selection response: %s", text[:200])
        return []


async def _search_level(
    nodes: list[dict],
    query: str,
    llm: LLMClient,
    model: Optional[str],
    depth: int = 0,
    max_depth: int = 5,
) -> list[dict]:
    """Recursively search one level of the tree.

    Returns a flat list of relevant node dicts (augmented with ``_depth``).
    """
    if not nodes or depth > max_depth:
        return []

    node_list_text = _format_node_list(nodes)
    prompt = _build_selection_prompt(query, node_list_text)

    response = await llm.acomplete(prompt, model=model)
    relevant_ids = _extract_relevant_ids(response.content)

    if not relevant_ids:
        return []

    # Build a lookup by node_id
    id_to_node: dict[str, dict] = {}
    for node in nodes:
        nid = node.get("node_id", "")
        if nid:
            id_to_node[nid] = node

    results: list[dict] = []
    for rid in relevant_ids:
        node = id_to_node.get(rid)
        if node is None:
            continue

        children = node.get("nodes", [])
        if children and depth < max_depth:
            # Drill deeper into this subtree
            child_results = await _search_level(
                children, query, llm, model, depth + 1, max_depth
            )
            if child_results:
                results.extend(child_results)
            else:
                # Children weren't relevant; use this node itself
                results.append({**node, "_depth": depth})
        else:
            # Leaf node or max depth reached
            results.append({**node, "_depth": depth})

    return results


async def tree_search(
    tree_index: dict,
    query: str,
    llm: LLMClient,
    model: Optional[str] = None,
    max_depth: int = 5,
    graph_augment: bool = False,
    document_id: Optional[str] = None,
    db=None,
) -> list[dict]:
    """Search a document tree index for sections relevant to a query.

    Args:
        tree_index: The full tree index dict (with ``structure`` key).
        query: The user's natural language query.
        llm: An ``LLMClient`` instance for making LLM calls.
        model: Optional model override.
        max_depth: Maximum recursion depth for drilling into subtrees.
        graph_augment: If True, expand results with graph-linked nodes.
        document_id: Required when ``graph_augment=True``.
        db: Async DB session, required when ``graph_augment=True``.

    Returns:
        A list of relevant node dicts, each containing at least:
        ``node_id``, ``title``, ``summary``, ``start_index``, ``end_index``.
    """
    structure = tree_index.get("structure", [])
    if not structure:
        logger.warning("tree_search called with empty structure")
        return []

    results = await _search_level(structure, query, llm, model, depth=0, max_depth=max_depth)

    # Deduplicate by node_id, preserving order
    seen: set[str] = set()
    deduped: list[dict] = []
    for node in results:
        nid = node.get("node_id", "")
        if nid and nid in seen:
            continue
        seen.add(nid)
        deduped.append(node)

    # --- Graph augmentation ---
    if graph_augment and document_id and db:
        import asyncio

        try:
            augmented = await asyncio.wait_for(
                _augment_with_graph(deduped, document_id, structure, db),
                timeout=10.0,
            )
            deduped.extend(augmented)
        except asyncio.TimeoutError:
            logger.warning("Graph augmentation timed out (non-fatal)")
        except Exception as exc:
            logger.warning("Graph augmentation failed (non-fatal): %s", exc)

    logger.info(
        "tree_search found %d relevant nodes for query: %s",
        len(deduped),
        query[:80],
    )
    return deduped


async def _augment_with_graph(
    nodes: list[dict],
    document_id: str,
    structure: list[dict],
    db,
) -> list[dict]:
    """Expand search results with graph-linked nodes from the same document.

    For each relevant node, look up its entities via EntityMention,
    then find other nodes in this document that also mention those entities.
    """
    from idpkit.graph.queries import get_related_sections

    existing_ids = {n.get("node_id") for n in nodes}
    augmented: list[dict] = []

    # Build a flat lookup of all nodes in the tree
    node_lookup = _build_node_lookup(structure)

    for node in nodes:
        nid = node.get("node_id", "")
        if not nid:
            continue

        related = await get_related_sections(db, document_id, nid)
        for section in related:
            related_nid = section["node_id"]
            if related_nid in existing_ids:
                continue

            # Look up the full node data from the tree
            related_node = node_lookup.get(related_nid)
            if related_node:
                aug_node = {**related_node, "_graph_augmented": True}
                augmented.append(aug_node)
                existing_ids.add(related_nid)

    return augmented


def _build_node_lookup(structure: list[dict]) -> dict[str, dict]:
    """Build a flat dict mapping node_id -> node dict from a tree structure."""
    lookup: dict[str, dict] = {}
    visited: set[str] = set()

    def _walk(nodes: list):
        for node in nodes:
            if isinstance(node, dict):
                nid = node.get("node_id", "")
                if nid:
                    if nid in visited:
                        continue  # Prevent infinite loops on circular trees
                    visited.add(nid)
                    lookup[nid] = {k: v for k, v in node.items() if k != "nodes"}
                for child in node.get("nodes", []):
                    _walk([child])

    _walk(structure)
    return lookup
