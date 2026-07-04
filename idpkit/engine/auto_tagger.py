"""AI-powered auto-tagging for documents using LLM analysis."""

import json
import logging
from typing import Optional

from sqlalchemy import select, insert, exists, func
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.core.llm import LLMClient
from idpkit.db.models import Document, Tag, document_tags
from idpkit.db.session import lock_tag_name

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 4000
_MAX_SUGGESTIONS = 3


async def suggest_tags(
    doc_id: str,
    user_id: str,
    db: AsyncSession,
    llm: LLMClient,
    content_override: Optional[str] = None,
) -> list[dict]:
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise ValueError(f"Document {doc_id} not found")

    if content_override:
        content = content_override[:_MAX_CONTENT_CHARS]
    else:
        content = _extract_document_content(doc)

    if not content.strip():
        return []

    tag_result = await db.execute(
        select(Tag).where(Tag.owner_id == user_id).order_by(Tag.name)
    )
    existing_tags = tag_result.scalars().all()
    existing_tag_info = [{"id": t.id, "name": t.name} for t in existing_tags]

    prompt = _build_prompt(content, existing_tag_info, doc.filename)

    try:
        response = await llm.acomplete(prompt, temperature=0.3)
        suggestions = _parse_response(response.content, existing_tag_info)
    except Exception as exc:
        logger.error("Auto-tag LLM call failed for doc %s: %s", doc_id, exc)
        return []

    return suggestions


def _extract_document_content(doc: Document) -> str:
    parts = []

    if doc.filename:
        parts.append(f"Filename: {doc.filename}")

    if doc.description:
        parts.append(f"Description: {doc.description}")

    meta = doc.metadata_json or {}
    if meta.get("title"):
        parts.append(f"Title: {meta['title']}")
    if meta.get("channel_title"):
        parts.append(f"Channel: {meta['channel_title']}")
    if meta.get("description"):
        parts.append(f"Content Description: {meta['description'][:500]}")
    if meta.get("tags"):
        parts.append(f"Source Tags: {', '.join(meta['tags'][:10])}")

    if doc.tree_index:
        tree_text = _extract_tree_text(doc.tree_index)
        if tree_text:
            parts.append(f"Document Structure:\n{tree_text}")

    return "\n".join(parts)[:_MAX_CONTENT_CHARS]


def _extract_tree_text(tree: dict) -> str:
    parts = []
    if tree.get("doc_name"):
        parts.append(tree["doc_name"])
    if tree.get("doc_description"):
        parts.append(tree["doc_description"])

    def walk(nodes, depth=0):
        if not isinstance(nodes, list):
            return
        for node in nodes[:20]:
            if isinstance(node, dict):
                title = node.get("title", "")
                if title:
                    parts.append("  " * depth + title)
                walk(node.get("children", []), depth + 1)

    walk(tree.get("structure", []))
    return "\n".join(parts[:30])


def _build_prompt(content: str, existing_tags: list[dict], filename: str) -> str:
    tag_list = ""
    if existing_tags:
        tag_list = "\n".join(f"  - \"{t['name']}\" (id: {t['id']})" for t in existing_tags)
    else:
        tag_list = "  (no existing tags)"

    return f"""Analyze the following document and suggest 1-{_MAX_SUGGESTIONS} descriptive tags for categorization.

DOCUMENT:
Filename: {filename}
{content}

EXISTING TAGS (prefer reusing these if they match):
{tag_list}

RULES:
1. Suggest 1-{_MAX_SUGGESTIONS} tags total
2. Prefer existing tags when the document fits their category
3. Only suggest new tags if no existing tag is a good match
4. Tag names should be short (1-3 words), descriptive, and in Title Case
5. Each suggestion needs a confidence score from 0.0 to 1.0

Return ONLY valid JSON in this format:
[
  {{"name": "Tag Name", "existing_id": "id-if-existing-or-null", "confidence": 0.85}},
  ...
]"""


def _parse_response(content: str, existing_tags: list[dict]) -> list[dict]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:])
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        suggestions = json.loads(content)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                suggestions = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Could not parse auto-tag response: %s", content[:200])
                return []
        else:
            return []

    if not isinstance(suggestions, list):
        return []

    existing_ids = {t["id"] for t in existing_tags}
    valid = []
    for s in suggestions[:_MAX_SUGGESTIONS]:
        if not isinstance(s, dict) or "name" not in s:
            continue
        eid = s.get("existing_id")
        if eid and eid not in existing_ids:
            eid = None
        valid.append({
            "name": str(s["name"]),
            "existing_id": eid,
            "confidence": min(1.0, max(0.0, float(s.get("confidence", 0.5)))),
        })

    return valid


async def apply_tags(
    doc_id: str,
    suggestions: list[dict],
    user_id: str,
    db: AsyncSession,
) -> list[dict]:
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.owner_id == user_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise ValueError(f"Document {doc_id} not found")

    applied = []
    # Acquire the per-name advisory locks up front in a deterministic (sorted)
    # order. Two concurrent auto-tag passes that touch overlapping name sets
    # would otherwise be able to grab the same locks in opposite order and
    # deadlock; sorting guarantees a consistent global acquisition order.
    lock_names = sorted({
        str(s["name"]).strip().lower()
        for s in suggestions
        if not s.get("existing_id") and str(s.get("name", "")).strip()
    })
    for _name in lock_names:
        await lock_tag_name(db, user_id, _name)

    for s in suggestions:
        if s.get("existing_id"):
            tag_result = await db.execute(
                select(Tag).where(Tag.id == s["existing_id"], Tag.owner_id == user_id)
            )
            tag = tag_result.scalar_one_or_none()
            if tag:
                already_linked = await db.execute(
                    select(document_tags).where(
                        document_tags.c.document_id == doc_id,
                        document_tags.c.tag_id == tag.id,
                    )
                )
                if not already_linked.first():
                    await db.execute(
                        insert(document_tags).values(
                            document_id=doc_id, tag_id=tag.id
                        )
                    )
                    applied.append({"tag_id": tag.id, "name": tag.name, "action": "assigned"})
        else:
            # No existing_id from the LLM — but a same-name tag may still exist
            # for this owner (the model can omit the id, or normalise casing
            # differently). Reuse it case-insensitively instead of creating a
            # duplicate "folder".
            name = str(s["name"]).strip()
            if not name:
                continue
            # The (owner, lower(name)) advisory lock was already acquired up
            # front, so this existence check + insert is serialized against
            # concurrent creates without risking a lock-ordering deadlock.
            existing = (
                await db.execute(
                    select(Tag).where(
                        Tag.owner_id == user_id,
                        func.lower(Tag.name) == name.lower(),
                    )
                )
            ).scalars().first()

            if existing is not None:
                tag = existing
                action = "assigned"
            else:
                tag = Tag(name=name, owner_id=user_id)
                db.add(tag)
                await db.flush()
                action = "created"

            already_linked = await db.execute(
                select(document_tags).where(
                    document_tags.c.document_id == doc_id,
                    document_tags.c.tag_id == tag.id,
                )
            )
            if not already_linked.first():
                await db.execute(
                    insert(document_tags).values(
                        document_id=doc_id, tag_id=tag.id
                    )
                )
                applied.append({"tag_id": tag.id, "name": tag.name, "action": action})

    await db.commit()
    return applied
