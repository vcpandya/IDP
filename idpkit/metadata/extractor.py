"""Category-aware smart-metadata extraction.

Two LLM passes per document:
  1. classify the document into one of the known categories;
  2. extract the category's standard fields plus a few contextual extras.

Results are persisted as ``DocumentFacet`` rows (one per value) and mirrored onto
``Document.doc_category`` / ``doc_category_confidence`` / ``smart_metadata``.

All failures are non-fatal — profiling enriches a document but never blocks
indexing or the rest of the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy import delete, select, text

from idpkit.core.llm import LLMClient
from idpkit.db.models import Document
from idpkit.metadata.categories import (
    DEFAULT_CATEGORY,
    field_label,
    get_category,
    get_category_keys,
)
from idpkit.metadata.models import DocumentFacet

logger = logging.getLogger(__name__)

# Cap how much document context we feed the LLM to keep cost/latency bounded.
_MAX_TITLES = 60
_MAX_DESCRIPTION_CHARS = 1500
_MAX_TITLE_CHARS = 4000
_VALUE_MAX_CHARS = 1000

# Values that should never become a facet.
_EMPTY_VALUES = {"", "n/a", "na", "none", "unknown", "not specified", "not stated",
                 "not available", "null", "-", "—"}


def _extract_json(text: str) -> dict | None:
    """Best-effort parse of a JSON object from an LLM response."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip ```json ... ``` fences.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} block.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _collect_titles(nodes: Any, acc: list[str]) -> None:
    """Recursively collect section titles from a tree-index structure."""
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if len(acc) >= _MAX_TITLES:
            return
        title = node.get("title") or node.get("name")
        if title and isinstance(title, str):
            acc.append(title.strip())
        children = node.get("structure") or node.get("children") or node.get("nodes")
        if children:
            _collect_titles(children, acc)


def _build_doc_context(doc: Document) -> str:
    """Assemble a compact textual snapshot of a document for the LLM."""
    parts: list[str] = [f"Filename: {doc.filename}", f"Format: {doc.format}"]

    meta = doc.metadata_json or {}
    if isinstance(meta, dict):
        for mk in ("title", "author", "subject", "channel_title"):
            mv = meta.get(mk)
            if mv:
                parts.append(f"{mk.title()}: {str(mv)[:200]}")

    if doc.description:
        parts.append(f"Summary: {doc.description[:_MAX_DESCRIPTION_CHARS]}")

    titles: list[str] = []
    tree = doc.tree_index
    if isinstance(tree, dict):
        _collect_titles(tree.get("structure") or [], titles)
    elif isinstance(tree, list):
        _collect_titles(tree, titles)
    if titles:
        joined = " | ".join(titles)[:_MAX_TITLE_CHARS]
        parts.append(f"Section titles: {joined}")

    return "\n".join(parts)


def _classify_prompt(context: str) -> str:
    keys = get_category_keys()
    options = "\n".join(
        f"- {key}: {get_category(key)['label']} — {get_category(key)['description']}"
        for key in keys
    )
    return (
        "You are a document classifier. Given the document context below, choose "
        "the single most appropriate category from the list.\n\n"
        f"Categories:\n{options}\n\n"
        f"Document context:\n{context}\n\n"
        'Respond ONLY with JSON: {"category": "<one_key_from_the_list>", '
        '"confidence": <0-100 integer>}. If none fit well, use "general".'
    )


def _extract_prompt(category_key: str, context: str) -> str:
    spec = get_category(category_key)
    field_lines = "\n".join(
        f'- "{f["key"]}" ({f["type"]}): {f["label"]} — {f["description"]}'
        for f in spec["fields"]
    )
    return (
        f"You are extracting structured metadata from a '{spec['label']}'.\n"
        "Fill in the standard fields below from the document context. You may also "
        "add a few extra fields (snake_case keys) if they are clearly important for "
        "this document.\n\n"
        f"Standard fields:\n{field_lines}\n\n"
        f"Document context:\n{context}\n\n"
        "Rules:\n"
        "- Only include fields you can determine from the context. Omit unknown fields entirely.\n"
        "- For 'list' type fields, return a JSON array of strings.\n"
        "- For all other types, return a single string.\n"
        "- Keep values concise (names, dates, short phrases). Do not invent data.\n\n"
        'Respond ONLY with a JSON object mapping field keys to values, e.g. '
        '{"judge": ["Justice A", "Justice B"], "court": "Supreme Court"}.'
    )


def _norm(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _coerce_values(raw: Any) -> list[str]:
    """Normalise an extracted field value into a list of clean string values."""
    out: list[str] = []
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        if item is None:
            continue
        if isinstance(item, (dict, list)):
            text = json.dumps(item, ensure_ascii=False)
        else:
            text = str(item)
        text = text.strip()[:_VALUE_MAX_CHARS]
        if not text or text.lower() in _EMPTY_VALUES:
            continue
        out.append(text)
    return out


async def _classify(llm: LLMClient, context: str) -> tuple[str, int]:
    prompt = _classify_prompt(context)
    try:
        resp = await llm.acomplete(prompt)
        data = _extract_json(resp.content) or {}
    except Exception as exc:  # noqa: BLE001 - non-fatal
        logger.warning("Metadata classification failed: %s", exc)
        return DEFAULT_CATEGORY, 0
    category = str(data.get("category") or DEFAULT_CATEGORY).strip()
    if category not in get_category_keys():
        category = DEFAULT_CATEGORY
    try:
        confidence = int(data.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0
    return category, max(0, min(100, confidence))


async def _extract_fields(llm: LLMClient, category: str, context: str) -> dict[str, Any]:
    prompt = _extract_prompt(category, context)
    try:
        resp = await llm.acomplete(prompt)
        data = _extract_json(resp.content)
    except Exception as exc:  # noqa: BLE001 - non-fatal
        logger.warning("Metadata field extraction failed: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _build_facets(
    category: str, fields: dict[str, Any]
) -> tuple[list[dict], dict[str, Any]]:
    """Turn raw extracted fields into facet dicts + a clean smart_metadata blob."""
    facets: list[dict] = []
    clean: dict[str, Any] = {}
    seen: set[tuple[str, str]] = set()

    for raw_key, raw_val in fields.items():
        if not isinstance(raw_key, str):
            continue
        key = _norm(raw_key).replace(" ", "_")
        if not key:
            continue
        values = _coerce_values(raw_val)
        if not values:
            continue
        clean[key] = values if len(values) > 1 else values[0]
        label = field_label(category, key)
        for value in values:
            vnorm = _norm(value)
            dedup = (key, vnorm)
            if dedup in seen:
                continue
            seen.add(dedup)
            facets.append({
                "key": key,
                "label": label,
                "value": value,
                "value_norm": vnorm,
            })
    return facets, clean


async def profile_document(
    session,
    llm: LLMClient,
    document: Document,
) -> dict:
    """Classify *document*, extract its facets, and persist the results.

    Returns a summary dict. Replaces any existing facets for the document so the
    operation is idempotent and safe to re-run ("reprocess").
    """
    context = _build_doc_context(document)
    category, confidence = await _classify(llm, context)
    raw_fields = await _extract_fields(llm, category, context)
    facets, clean = _build_facets(category, raw_fields)

    # Serialize the replace-facets critical section so the background post-index
    # hook and a manual "reprocess" cannot interleave their delete+insert and
    # leave duplicate facets behind. On PostgreSQL a transaction-scoped advisory
    # lock (keyed on the document id) makes the second writer wait for the first
    # to commit; the unique constraint on (document_id, key, value_norm) is the
    # portable safety net for other engines.
    if getattr(session.bind, "dialect", None) is not None \
            and session.bind.dialect.name == "postgresql":
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
            {"k": f"idpkit_facet:{document.id}"},
        )

    # Replace existing facets for this document.
    await session.execute(
        delete(DocumentFacet).where(DocumentFacet.document_id == document.id)
    )
    for f in facets:
        session.add(DocumentFacet(
            document_id=document.id,
            category=category,
            key=f["key"],
            label=f["label"],
            value=f["value"],
            value_norm=f["value_norm"],
            confidence=confidence or 80,
        ))

    document.doc_category = category
    document.doc_category_confidence = confidence
    document.smart_metadata = {
        "category": category,
        "confidence": confidence,
        "fields": clean,
    }
    await session.commit()

    return {
        "document_id": document.id,
        "category": category,
        "confidence": confidence,
        "facet_count": len(facets),
    }


async def profile_document_by_id(session, llm: LLMClient, doc_id: str) -> dict | None:
    """Convenience wrapper: load a document by id then profile it."""
    doc = (
        await session.execute(select(Document).where(Document.id == doc_id))
    ).scalar_one_or_none()
    if not doc:
        return None
    if not doc.tree_index and not doc.description:
        logger.debug("Skipping metadata profiling for %s: nothing to profile", doc_id)
        return None
    return await profile_document(session, llm, doc)
