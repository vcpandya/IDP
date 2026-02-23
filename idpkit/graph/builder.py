"""Build document knowledge graph — extract entities and intra-doc edges at index time."""

import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.core.llm import LLMClient
from .models import Entity, EntityMention, GraphEdge
from .prompts import (
    ENTITY_EXTRACTION_PROMPT,
    VALID_ENTITY_TYPES,
    VALID_RELATION_TYPES,
    _sanitize_for_prompt,
)
from .schemas import ExtractedEntity, ExtractedRelation, NodeExtractionResult

logger = logging.getLogger(__name__)

# Process tree nodes in batches of this size per LLM call.
_BATCH_SIZE = 5

# Safety caps to prevent memory exhaustion on large documents.
_MAX_ENTITIES_PER_BATCH = 50
_MAX_RELATIONS_PER_BATCH = 100
_MAX_CONTENT_NODES = 200


def _flatten_tree(tree: Any) -> list[dict]:
    """Recursively flatten a tree index into a list of node dicts."""
    nodes: list[dict] = []
    if isinstance(tree, dict):
        node_copy = {k: v for k, v in tree.items() if k != "nodes"}
        nodes.append(node_copy)
        for child in tree.get("nodes", []):
            nodes.extend(_flatten_tree(child))
    elif isinstance(tree, list):
        for item in tree:
            nodes.extend(_flatten_tree(item))
    return nodes


def _format_sections_for_extraction(nodes: list[dict]) -> str:
    """Format a batch of tree nodes into text for the LLM extraction prompt."""
    parts: list[str] = []
    for node in nodes:
        node_id = node.get("node_id", "")
        title = _sanitize_for_prompt(node.get("title", "(untitled)"), max_length=500)
        text = node.get("text") or node.get("summary") or node.get("prefix_summary") or ""
        text = _sanitize_for_prompt(text, max_length=3000)
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        parts.append(f"[Section {node_id}: {title}, pages {start}-{end}]\n{text}")
    return "\n\n---\n\n".join(parts)


def _parse_extraction_response(content: str) -> NodeExtractionResult:
    """Parse the LLM extraction response into structured data."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse entity extraction response: %s", text[:200])
        return NodeExtractionResult()

    entities = []
    for e in data.get("entities", [])[:_MAX_ENTITIES_PER_BATCH]:
        try:
            extracted = ExtractedEntity(**e)
            # Validate entity type — normalize or skip invalid ones
            normalized_type = extracted.entity_type.upper().strip()
            if normalized_type not in VALID_ENTITY_TYPES:
                logger.debug("Skipping entity with invalid type: %s", normalized_type)
                continue
            extracted.entity_type = normalized_type
            # Truncate fields to prevent oversized storage
            extracted.name = extracted.name[:500]
            extracted.description = extracted.description[:2000]
            extracted.aliases = [a[:200] for a in extracted.aliases[:20]]
            entities.append(extracted)
        except Exception:
            continue

    relations = []
    for r in data.get("relations", [])[:_MAX_RELATIONS_PER_BATCH]:
        try:
            rel = ExtractedRelation(**r)
            if rel.relation_type not in VALID_RELATION_TYPES:
                continue
            rel.confidence = max(0, min(100, rel.confidence))
            rel.context = rel.context[:500]
            relations.append(rel)
        except Exception:
            continue

    return NodeExtractionResult(entities=entities, relations=relations)


async def _extract_from_batch(
    nodes: list[dict],
    llm: LLMClient,
    model: str | None = None,
) -> NodeExtractionResult:
    """Run entity extraction on a batch of tree nodes via LLM."""
    sections_text = _format_sections_for_extraction(nodes)
    if not sections_text.strip():
        return NodeExtractionResult()

    prompt = ENTITY_EXTRACTION_PROMPT.format(sections_text=sections_text)

    try:
        response = await llm.acomplete(prompt, model=model)
        return _parse_extraction_response(response.content)
    except Exception as exc:
        logger.error("Entity extraction LLM call failed: %s", exc)
        return NodeExtractionResult()


def _deduplicate_entities(
    entities: list[tuple[ExtractedEntity, str, dict]],
) -> dict[str, tuple[ExtractedEntity, list[tuple[str, dict]]]]:
    """Deduplicate entities within a document by (canonical_name, entity_type).

    Args:
        entities: list of (ExtractedEntity, node_id, node_dict) tuples.

    Returns:
        Dict mapping "name|type" key to (merged_entity, [(node_id, node_dict), ...]).
    """
    merged: dict[str, tuple[ExtractedEntity, list[tuple[str, dict]]]] = {}

    for entity, node_id, node in entities:
        key = f"{entity.name.lower().strip()}|{entity.entity_type.upper().strip()}"
        if key in merged:
            existing_entity, mentions = merged[key]
            # Merge aliases (guard against None)
            existing_aliases = set(existing_entity.aliases or [])
            new_aliases = set(entity.aliases or [])
            existing_entity.aliases = list(existing_aliases | new_aliases)
            # Keep longer description
            if len(entity.description) > len(existing_entity.description):
                existing_entity.description = entity.description
            mentions.append((node_id, node))
        else:
            merged[key] = (entity, [(node_id, node)])

    return merged


async def build_document_graph(
    document_id: str,
    tree_index: dict,
    llm: LLMClient,
    db: AsyncSession,
    model: str | None = None,
) -> dict:
    """Extract entities and build intra-document graph edges.

    Args:
        document_id: The document's database ID.
        tree_index: The full tree index dict (with ``structure`` key).
        llm: LLMClient for making extraction calls.
        db: Async database session.
        model: Optional model override.

    Returns:
        Summary dict with entity_count, edge_count, mention_count.
    """
    structure = tree_index.get("structure", [])
    if not structure:
        logger.warning("build_document_graph: empty structure for doc %s", document_id)
        return {"entity_count": 0, "edge_count": 0, "mention_count": 0}

    # Flatten tree to get all nodes
    all_nodes = _flatten_tree(structure)
    # Filter to nodes with actual content
    content_nodes = [
        n for n in all_nodes
        if (n.get("text") or n.get("summary") or n.get("prefix_summary"))
    ]

    if not content_nodes:
        return {"entity_count": 0, "edge_count": 0, "mention_count": 0}

    # Cap content nodes to prevent excessive LLM calls on very large documents
    if len(content_nodes) > _MAX_CONTENT_NODES:
        logger.warning(
            "Document %s has %d content nodes, capping at %d for graph building",
            document_id, len(content_nodes), _MAX_CONTENT_NODES,
        )
        content_nodes = content_nodes[:_MAX_CONTENT_NODES]

    # --- Step 1: Extract entities in batches ---
    all_extracted: list[tuple[ExtractedEntity, str, dict]] = []
    all_relations: list[ExtractedRelation] = []

    for i in range(0, len(content_nodes), _BATCH_SIZE):
        batch = content_nodes[i : i + _BATCH_SIZE]
        result = await _extract_from_batch(batch, llm, model)

        # Associate each entity with the first node in the batch
        # (a simplification — the LLM saw all batch nodes together)
        first_node = batch[0]
        for entity in result.entities:
            all_extracted.append((entity, first_node.get("node_id", ""), first_node))

        all_relations.extend(result.relations)

    if not all_extracted:
        return {"entity_count": 0, "edge_count": 0, "mention_count": 0}

    # --- Step 2: Deduplicate entities within document ---
    deduped = _deduplicate_entities(all_extracted)

    # --- Step 3: Persist entities, mentions, and edges ---
    entity_name_to_id: dict[str, str] = {}
    entity_count = 0
    mention_count = 0

    for key, (extracted, node_mentions) in deduped.items():
        canonical = extracted.name.strip()[:500]
        etype = extracted.entity_type.upper().strip()

        # Check if entity already exists globally (exact match)
        stmt = select(Entity).where(
            Entity.canonical_name == canonical,
            Entity.entity_type == etype,
        )
        existing = (await db.execute(stmt)).scalar_one_or_none()

        if existing:
            entity_row = existing
            entity_row.document_count = (entity_row.document_count or 1) + 1
            if extracted.description and len(extracted.description) > len(entity_row.description or ""):
                entity_row.description = extracted.description[:2000]
            # Merge aliases (guard against None)
            old_aliases = entity_row.aliases or []
            new_aliases = list(set(old_aliases + (extracted.aliases or [])))[:50]
            entity_row.aliases = new_aliases
        else:
            entity_row = Entity(
                canonical_name=canonical,
                entity_type=etype,
                description=extracted.description[:2000] if extracted.description else None,
                aliases=(extracted.aliases or [])[:50],
                first_document_id=document_id,
                document_count=1,
            )
            db.add(entity_row)
            await db.flush()  # Get the generated id
            entity_count += 1

        entity_name_to_id[canonical.lower()] = entity_row.id

        # Create mentions
        for node_id, node in node_mentions:
            mention = EntityMention(
                entity_id=entity_row.id,
                document_id=document_id,
                node_id=node_id,
                node_title=(node.get("title") or "")[:500],
                mention_text=canonical[:500],
                start_page=node.get("start_index"),
                end_page=node.get("end_index"),
            )
            db.add(mention)
            mention_count += 1

    await db.flush()

    # --- Step 4: Create intra-doc edges from extracted relations ---
    edge_count = 0
    for rel in all_relations:
        src_id = entity_name_to_id.get(rel.source.lower().strip())
        tgt_id = entity_name_to_id.get(rel.target.lower().strip())
        if not src_id or not tgt_id or src_id == tgt_id:
            continue

        edge = GraphEdge(
            source_entity_id=src_id,
            source_document_id=document_id,
            target_entity_id=tgt_id,
            target_document_id=document_id,
            relation_type=rel.relation_type,
            scope="intra",
            weight=1,
            confidence=max(0, min(100, rel.confidence)),
            context_snippet=rel.context[:500] if rel.context else None,
        )
        db.add(edge)
        edge_count += 1

    # --- Step 5: Create co-occurrence edges for entities sharing the same node ---
    _create_cooccurrence_edges(deduped, entity_name_to_id, document_id, db)

    await db.commit()

    logger.info(
        "build_document_graph for doc %s: %d entities, %d edges, %d mentions",
        document_id,
        entity_count,
        edge_count,
        mention_count,
    )
    return {
        "entity_count": entity_count,
        "edge_count": edge_count,
        "mention_count": mention_count,
    }


def _create_cooccurrence_edges(
    deduped: dict[str, tuple[ExtractedEntity, list[tuple[str, dict]]]],
    entity_name_to_id: dict[str, str],
    document_id: str,
    db: AsyncSession,
) -> None:
    """Create co_occurrence edges between entities that share the same tree node."""
    # Build node_id → entity_ids mapping
    node_entities: dict[str, list[str]] = {}
    for key, (extracted, mentions) in deduped.items():
        eid = entity_name_to_id.get(extracted.name.lower().strip())
        if not eid:
            continue
        for node_id, _node in mentions:
            node_entities.setdefault(node_id, []).append(eid)

    # Create edges between entities that share nodes
    seen_pairs: set[tuple[str, str]] = set()
    for node_id, eids in node_entities.items():
        for i, eid_a in enumerate(eids):
            for eid_b in eids[i + 1 :]:
                pair = (min(eid_a, eid_b), max(eid_a, eid_b))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edge = GraphEdge(
                    source_entity_id=pair[0],
                    source_document_id=document_id,
                    source_node_id=node_id,
                    target_entity_id=pair[1],
                    target_document_id=document_id,
                    target_node_id=node_id,
                    relation_type="co_occurrence",
                    scope="intra",
                    weight=1,
                    confidence=90,
                )
                db.add(edge)
