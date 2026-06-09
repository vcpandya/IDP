"""Read queries over smart-metadata facets.

All queries are scoped to a single owner via a join on ``documents.owner_id`` so
users only ever see facets for their own documents.
"""

from __future__ import annotations

from sqlalchemy import distinct, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.models import Document
from idpkit.metadata.categories import field_label, normalize_value
from idpkit.metadata.models import DocumentFacet

# Bound payloads/memory so a tenant with very many documents can never load the
# whole facet table into the worker or ship a multi-megabyte JSON response.
MAX_FACET_ROWS = 2000
MAX_FILTER_DOCS = 1000
MAX_GRAPH_FACET_ROWS = 5000


async def get_facets(
    db: AsyncSession,
    owner_id: str,
    *,
    category: str | None = None,
    key: str | None = None,
    search: str | None = None,
    min_count: int = 1,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """Aggregate facets into groups keyed by field.

    Returns a list of ``{key, label, values: [{value, value_norm, document_count}]}``
    sorted by total document count desc.

    When *doc_ids* is provided the aggregation is restricted to that set
    (a knowledge base or an explicit selection). ``None`` means all of the
    owner's documents; an empty list means "no documents" (empty result).
    """
    if doc_ids is not None and not doc_ids:
        return []
    stmt = (
        select(
            DocumentFacet.key,
            DocumentFacet.label,
            DocumentFacet.value,
            DocumentFacet.value_norm,
            func.count(distinct(DocumentFacet.document_id)).label("doc_count"),
        )
        .join(Document, Document.id == DocumentFacet.document_id)
        .where(Document.owner_id == owner_id)
        .group_by(
            DocumentFacet.key,
            DocumentFacet.label,
            DocumentFacet.value,
            DocumentFacet.value_norm,
        )
    )
    if doc_ids is not None:
        stmt = stmt.where(DocumentFacet.document_id.in_(doc_ids))
    if category:
        stmt = stmt.where(DocumentFacet.category == category)
    if key:
        stmt = stmt.where(DocumentFacet.key == key)
    if search:
        stmt = stmt.where(
            DocumentFacet.value_norm.like(f"%{normalize_value(search)}%")
        )
    # Keep the most common facet values and cap the row set so the response
    # stays bounded for large libraries.
    stmt = stmt.order_by(func.count(distinct(DocumentFacet.document_id)).desc()).limit(
        MAX_FACET_ROWS
    )

    rows = (await db.execute(stmt)).all()

    groups: dict[str, dict] = {}
    for row in rows:
        if row.doc_count < min_count:
            continue
        grp = groups.setdefault(
            row.key,
            {"key": row.key, "label": row.label or field_label(None, row.key),
             "values": [], "total": 0},
        )
        grp["values"].append({
            "value": row.value,
            "value_norm": row.value_norm,
            "document_count": int(row.doc_count),
        })
        grp["total"] += int(row.doc_count)

    for grp in groups.values():
        grp["values"].sort(key=lambda v: (-v["document_count"], v["value"].lower()))

    return sorted(groups.values(), key=lambda g: (-g["total"], g["label"].lower()))


async def filter_documents(
    db: AsyncSession,
    owner_id: str,
    criteria: list[dict],
    *,
    match: str = "all",
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """Return documents matching the facet *criteria*.

    Each criterion is ``{"key": str, "value_norm": str}``. With ``match="all"``
    a document must satisfy every criterion; with ``match="any"`` at least one.
    Returns ``{id, filename, format, category, matched: [{key, value}]}``.

    *doc_ids* optionally restricts the search to a knowledge base / selection.
    """
    if doc_ids is not None and not doc_ids:
        return []
    if not criteria:
        return []

    # Normalise + de-duplicate (key, value_norm) pairs once.
    pairs: list[tuple[str, str]] = []
    for crit in criteria:
        ckey = crit.get("key")
        cval = normalize_value(crit.get("value_norm") or crit.get("value") or "")
        if ckey and cval:
            pairs.append((ckey, cval))
    pairs = list(dict.fromkeys(pairs))
    if not pairs:
        return []

    pair_filter = tuple_(DocumentFacet.key, DocumentFacet.value_norm).in_(pairs)

    # Single aggregation to find candidate documents. The unique
    # (document_id, key, value_norm) constraint guarantees a document can match
    # each pair at most once, so the matched-row count equals the number of
    # distinct criteria satisfied: HAVING count == len(pairs) implements AND
    # ("all"); for OR ("any") any matching row qualifies.
    cand = (
        select(DocumentFacet.document_id)
        .join(Document, Document.id == DocumentFacet.document_id)
        .where(Document.owner_id == owner_id, pair_filter)
        .group_by(DocumentFacet.document_id)
    )
    if doc_ids is not None:
        cand = cand.where(DocumentFacet.document_id.in_(doc_ids))
    if match != "any":
        cand = cand.having(func.count() == len(pairs))
    # Deterministic ordering so that, when the result set exceeds the cap, the
    # truncated subset is stable across repeated calls (most-matching first).
    cand = cand.order_by(
        func.count().desc(), DocumentFacet.document_id
    ).limit(MAX_FILTER_DOCS)
    doc_ids = [row[0] for row in (await db.execute(cand)).all()]
    if not doc_ids:
        return []

    # One pass to fetch the documents and the facets that matched (for display),
    # owner-scoped again as defense-in-depth.
    rows = (
        await db.execute(
            select(Document, DocumentFacet)
            .join(DocumentFacet, DocumentFacet.document_id == Document.id)
            .where(
                Document.owner_id == owner_id,
                Document.id.in_(doc_ids),
                pair_filter,
            )
        )
    ).all()

    docs_by_id: dict[str, Document] = {}
    matched_by_doc: dict[str, list[dict]] = {}
    for doc, facet in rows:
        docs_by_id[doc.id] = doc
        matched_by_doc.setdefault(doc.id, []).append(
            {"key": facet.key, "label": facet.label, "value": facet.value}
        )

    result = [
        {
            "id": d.id,
            "filename": d.filename,
            "format": d.format,
            "category": d.doc_category,
            "status": d.status,
            "matched": matched_by_doc.get(d.id, []),
        }
        for d in docs_by_id.values()
    ]
    result.sort(key=lambda d: (-len(d["matched"]), d["filename"].lower()))
    return result


async def build_facet_graph(
    db: AsyncSession,
    owner_id: str,
    criteria: list[dict],
    *,
    match: str = "all",
    doc_ids: list[str] | None = None,
) -> dict:
    """Build a document-centric graph for the documents matching *criteria*.

    Nodes are documents plus facet-value "hub" nodes; edges connect a document to
    each facet value it carries (restricted to the criteria field keys so the
    graph stays focused on the dimensions the user is exploring).
    """
    docs = await filter_documents(db, owner_id, criteria, match=match, doc_ids=doc_ids)
    if not docs:
        return {"nodes": [], "edges": []}

    doc_ids = [d["id"] for d in docs]
    focus_keys = {c.get("key") for c in criteria if c.get("key")}

    # Only load the facet rows we actually graph (restricted to the focus keys)
    # and cap the row set so a broad selection cannot pull the whole table.
    facet_stmt = select(DocumentFacet).where(DocumentFacet.document_id.in_(doc_ids))
    if focus_keys:
        facet_stmt = facet_stmt.where(DocumentFacet.key.in_(focus_keys))
    # Deterministic order so a capped graph renders a stable subset.
    facet_stmt = facet_stmt.order_by(
        DocumentFacet.key, DocumentFacet.value_norm, DocumentFacet.document_id
    ).limit(MAX_GRAPH_FACET_ROWS)
    facet_rows = (await db.execute(facet_stmt)).scalars().all()

    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    for d in docs:
        nodes[f"doc:{d['id']}"] = {
            "id": f"doc:{d['id']}",
            "type": "document",
            "label": d["filename"],
            "document_id": d["id"],
            "category": d["category"],
        }

    for f in facet_rows:
        hub_id = f"facet:{f.key}:{f.value_norm}"
        if hub_id not in nodes:
            nodes[hub_id] = {
                "id": hub_id,
                "type": "facet",
                "facet_key": f.key,
                "label": f.value,
                "facet_label": f.label,
            }
        edges.append({
            "source": f"doc:{f.document_id}",
            "target": hub_id,
            "relation": f.key,
        })

    return {"nodes": list(nodes.values()), "edges": edges}


async def get_document_facets(
    db: AsyncSession, doc_id: str, *, owner_id: str | None = None
) -> list[dict]:
    """Return all facets for a single document, ordered by field.

    When *owner_id* is supplied the query is scoped via a join on the owning
    document, so a facet can never be read for another tenant's document even if
    the caller forgets an upstream ownership check (defense-in-depth).
    """
    stmt = (
        select(DocumentFacet)
        .where(DocumentFacet.document_id == doc_id)
        .order_by(DocumentFacet.key, DocumentFacet.value)
    )
    if owner_id is not None:
        stmt = stmt.join(Document, Document.id == DocumentFacet.document_id).where(
            Document.owner_id == owner_id
        )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {"key": f.key, "label": f.label, "value": f.value, "confidence": f.confidence}
        for f in rows
    ]


async def get_stats(
    db: AsyncSession, owner_id: str, *, doc_ids: list[str] | None = None
) -> dict:
    """Summary counts for the Document Map header / reprocess affordance.

    *doc_ids* optionally scopes the counts to a knowledge base / selection.
    """
    if doc_ids is not None and not doc_ids:
        return {"total_documents": 0, "profiled_documents": 0, "by_category": {}}

    total_stmt = select(func.count(Document.id)).where(Document.owner_id == owner_id)
    profiled_stmt = (
        select(func.count(distinct(DocumentFacet.document_id)))
        .join(Document, Document.id == DocumentFacet.document_id)
        .where(Document.owner_id == owner_id)
    )
    by_cat_stmt = (
        select(Document.doc_category, func.count(Document.id))
        .where(Document.owner_id == owner_id, Document.doc_category.isnot(None))
        .group_by(Document.doc_category)
    )
    if doc_ids is not None:
        total_stmt = total_stmt.where(Document.id.in_(doc_ids))
        profiled_stmt = profiled_stmt.where(DocumentFacet.document_id.in_(doc_ids))
        by_cat_stmt = by_cat_stmt.where(Document.id.in_(doc_ids))

    total = (await db.execute(total_stmt)).scalar_one()
    profiled = (await db.execute(profiled_stmt)).scalar_one()
    by_cat_rows = (await db.execute(by_cat_stmt)).all()
    return {
        "total_documents": int(total or 0),
        "profiled_documents": int(profiled or 0),
        "by_category": {row[0]: int(row[1]) for row in by_cat_rows},
    }
