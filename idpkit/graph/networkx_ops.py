"""Optional NetworkX utilities — centrality, communities, path finding.

These require the ``networkx`` package (optional dependency).
Import errors are caught gracefully so the rest of the graph module
works without NetworkX installed.
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from .queries import get_document_entities, get_document_edges

logger = logging.getLogger(__name__)

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    nx = None  # type: ignore[assignment]
    HAS_NETWORKX = False


def _require_networkx() -> None:
    if not HAS_NETWORKX:
        raise ImportError(
            "NetworkX is required for graph analytics. "
            "Install it with: pip install idpkit[graph]"
        )


async def build_networkx_graph(
    db: AsyncSession,
    document_id: str,
) -> Any:
    """Build a NetworkX graph from a document's entities and edges.

    Returns:
        A ``networkx.Graph`` instance.
    """
    _require_networkx()

    entities = await get_document_entities(db, document_id)
    edges = await get_document_edges(db, document_id)

    G = nx.Graph()

    for entity in entities:
        G.add_node(
            entity.id,
            label=entity.canonical_name,
            entity_type=entity.entity_type,
        )

    for edge in edges:
        if G.has_node(edge.source_entity_id) and G.has_node(edge.target_entity_id):
            G.add_edge(
                edge.source_entity_id,
                edge.target_entity_id,
                relation_type=edge.relation_type,
                weight=edge.weight or 1,
                confidence=edge.confidence or 80,
            )

    return G


async def compute_centrality(
    db: AsyncSession,
    document_id: str,
) -> dict[str, float]:
    """Compute betweenness centrality for entities in a document's graph.

    Returns:
        Dict mapping entity_id → centrality score (0.0 to 1.0).
    """
    G = await build_networkx_graph(db, document_id)
    if len(G.nodes) == 0:
        return {}
    return nx.betweenness_centrality(G)


async def detect_communities(
    db: AsyncSession,
    document_id: str,
) -> list[set[str]]:
    """Detect communities (clusters) of related entities using greedy modularity.

    Returns:
        List of sets, each set containing entity IDs in a community.
    """
    _require_networkx()
    G = await build_networkx_graph(db, document_id)
    if len(G.nodes) == 0:
        return []

    from networkx.algorithms.community import greedy_modularity_communities

    communities = greedy_modularity_communities(G)
    return [set(c) for c in communities]


async def find_shortest_path(
    db: AsyncSession,
    document_id: str,
    source_entity_id: str,
    target_entity_id: str,
) -> list[str] | None:
    """Find the shortest path between two entities in a document's graph.

    Returns:
        List of entity IDs forming the path, or None if no path exists.
    """
    G = await build_networkx_graph(db, document_id)
    try:
        return nx.shortest_path(G, source_entity_id, target_entity_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
