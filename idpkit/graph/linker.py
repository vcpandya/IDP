"""Cross-document entity linking — match entities across documents."""

import json
import logging

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.core.llm import LLMClient
from .models import Entity, EntityMention, GraphEdge
from .prompts import ENTITY_RESOLUTION_PROMPT, _sanitize_for_prompt

logger = logging.getLogger(__name__)

# Cap the number of fuzzy alias comparisons to prevent runaway LLM calls.
_MAX_FUZZY_CANDIDATES = 20


async def link_entities_across_documents(
    document_id: str,
    db: AsyncSession,
    llm: LLMClient | None = None,
    model: str | None = None,
) -> dict:
    """Link entities from a newly indexed document to existing corpus entities.

    Performs two passes:
    1. Exact match: Same canonical_name + entity_type -> auto-link (no LLM).
    2. Fuzzy match: Overlapping aliases or similar names -> LLM confirmation.

    Args:
        document_id: The document whose entities to link.
        db: Async database session.
        llm: Optional LLMClient for fuzzy resolution.
        model: Optional model override.

    Returns:
        Summary dict with exact_links and fuzzy_links counts.
    """
    # Get entities mentioned in this document (capped)
    stmt = (
        select(EntityMention.entity_id)
        .where(EntityMention.document_id == document_id)
        .distinct()
        .limit(500)
    )
    result = await db.execute(stmt)
    doc_entity_ids = [row[0] for row in result.all()]

    if not doc_entity_ids:
        return {"exact_links": 0, "fuzzy_links": 0}

    # Load the full entity objects for this document
    stmt = select(Entity).where(Entity.id.in_(doc_entity_ids))
    doc_entities = (await db.execute(stmt)).scalars().all()

    exact_links = 0
    fuzzy_links = 0

    for entity in doc_entities:
        # Find entities in OTHER documents with the same canonical name + type
        # that are not already the same entity row
        stmt = (
            select(Entity)
            .where(
                and_(
                    Entity.canonical_name == entity.canonical_name,
                    Entity.entity_type == entity.entity_type,
                    Entity.id != entity.id,
                )
            )
            .limit(50)
        )
        exact_matches = (await db.execute(stmt)).scalars().all()

        for match in exact_matches:
            # Check if edge already exists
            if await _edge_exists(db, entity.id, match.id):
                continue

            # Create same_entity edge
            edge = GraphEdge(
                source_entity_id=entity.id,
                source_document_id=document_id,
                target_entity_id=match.id,
                target_document_id=match.first_document_id,
                relation_type="same_entity",
                scope="inter",
                weight=1,
                confidence=100,
                context_snippet=f"Exact name match: {entity.canonical_name[:200]}",
            )
            db.add(edge)
            exact_links += 1

        # Fuzzy matching: check for alias overlaps
        if llm and entity.aliases:
            fuzzy_links += await _fuzzy_link(
                entity, document_id, db, llm, model
            )

    await db.flush()
    await db.commit()

    logger.info(
        "link_entities for doc %s: %d exact, %d fuzzy links",
        document_id,
        exact_links,
        fuzzy_links,
    )
    return {"exact_links": exact_links, "fuzzy_links": fuzzy_links}


async def _edge_exists(
    db: AsyncSession,
    entity_a_id: str,
    entity_b_id: str,
) -> bool:
    """Check if any edge (in either direction) exists between two entities."""
    stmt = select(func.count()).select_from(GraphEdge).where(
        (
            (GraphEdge.source_entity_id == entity_a_id)
            & (GraphEdge.target_entity_id == entity_b_id)
        )
        | (
            (GraphEdge.source_entity_id == entity_b_id)
            & (GraphEdge.target_entity_id == entity_a_id)
        )
    )
    count = (await db.execute(stmt)).scalar()
    return count > 0


async def _fuzzy_link(
    entity: Entity,
    document_id: str,
    db: AsyncSession,
    llm: LLMClient,
    model: str | None,
) -> int:
    """Attempt fuzzy entity linking by alias overlap + LLM confirmation."""
    aliases = entity.aliases or []
    if not aliases:
        return 0

    links_created = 0
    candidates_checked = 0

    for alias in aliases[:20]:  # Cap aliases to check
        alias_clean = alias.strip().lower()
        if not alias_clean or len(alias_clean) < 3:
            continue

        # Search for entities with this alias as canonical name
        stmt = (
            select(Entity)
            .where(
                and_(
                    func.lower(Entity.canonical_name) == alias_clean,
                    Entity.id != entity.id,
                )
            )
            .limit(10)
        )
        candidates = (await db.execute(stmt)).scalars().all()

        for candidate in candidates:
            if candidates_checked >= _MAX_FUZZY_CANDIDATES:
                return links_created
            candidates_checked += 1

            if await _edge_exists(db, entity.id, candidate.id):
                continue

            # Use LLM to confirm the match
            confirmed = await _llm_confirm_match(entity, candidate, llm, model)
            if confirmed:
                edge = GraphEdge(
                    source_entity_id=entity.id,
                    source_document_id=document_id,
                    target_entity_id=candidate.id,
                    target_document_id=candidate.first_document_id,
                    relation_type="same_entity",
                    scope="inter",
                    weight=1,
                    confidence=85,
                    context_snippet=f"Fuzzy match via alias: {alias[:200]}",
                )
                db.add(edge)
                links_created += 1

    return links_created


async def _llm_confirm_match(
    entity_a: Entity,
    entity_b: Entity,
    llm: LLMClient,
    model: str | None,
) -> bool:
    """Ask the LLM whether two entities refer to the same thing."""
    prompt = ENTITY_RESOLUTION_PROMPT.format(
        name_a=_sanitize_for_prompt(entity_a.canonical_name, 500),
        type_a=_sanitize_for_prompt(entity_a.entity_type, 50),
        desc_a=_sanitize_for_prompt(entity_a.description or "(no description)", 500),
        aliases_a=_sanitize_for_prompt(
            ", ".join(entity_a.aliases or []) or "(none)", 500
        ),
        doc_a=entity_a.first_document_id or "unknown",
        name_b=_sanitize_for_prompt(entity_b.canonical_name, 500),
        type_b=_sanitize_for_prompt(entity_b.entity_type, 50),
        desc_b=_sanitize_for_prompt(entity_b.description or "(no description)", 500),
        aliases_b=_sanitize_for_prompt(
            ", ".join(entity_b.aliases or []) or "(none)", 500
        ),
        doc_b=entity_b.first_document_id or "unknown",
    )

    try:
        response = await llm.acomplete(prompt, model=model)
        text = response.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        return data.get("same_entity", False) and data.get("confidence", 0) >= 70
    except Exception as exc:
        logger.warning("LLM entity resolution failed: %s", exc)
        return False
