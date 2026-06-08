"""Read queries over smart-metadata facets.

All queries are scoped to a single owner via a join on ``documents.owner_id`` so
users only ever see facets for their own documents.
"""

from __future__ import annotations

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from idpkit.db.models import Document
from idpkit.metadata.categories import field_label
from idpkit.metadata.models import DocumentFacet


async def get_facets(
    db: AsyncSession,
    owner_id: str,
    *,
    category: str | None = None,
    key: str | None = None,
    search: str | None = None,
    min_count: int = 1,
) -> list[dict]:
    """Aggregate facets into groups keyed by field.

    Returns a list of ``{key, label, values: [{value, value_norm, document_count}]}``
    sorted by total document count desc.
    """
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
    if category:
        stmt = stmt.where(DocumentFacet.category == category)
    if key:
        stmt = stmt.where(DocumentFacet.key == key)
    if search:
        stmt = stmt.where(DocumentFacet.value_norm.like(f"%{search.lower().strip()}%"))

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
) -> list[dict]:
    """Return documents matching the facet *criteria*.

    Each criterion is ``{"key": str, "value_norm": str}``. With ``match="all"``
    a document must satisfy every criterion; with ``match="any"`` at least one.
    Returns ``{id, filename, format, category, matched: [{key, value}]}``.
    """
    if not criteria:
        return []

    per_criterion: list[set[str]] = []
    for crit in criteria:
        ckey = crit.get("key")
        cval = (crit.get("value_norm") or crit.get("value") or "").lower().strip()
        if not ckey or not cval:
            continue
        stmt = (
            select(distinct(DocumentFacet.document_id))
            .join(Document, Document.id == DocumentFacet.document_id)
            .where(
                Document.owner_id == owner_id,
                DocumentFacet.key == ckey,
                DocumentFacet.value_norm == cval,
            )
        )
        ids = {row[0] for row in (await db.execute(stmt)).all()}
        per_criterion.append(ids)

    if not per_criterion:
        return []

    if match == "any":
        doc_ids: set[str] = set().union(*per_criterion)
    else:
        doc_ids = set(per_criterion[0])
        for s in per_criterion[1:]:
            doc_ids &= s

    if not doc_ids:
        return []

    docs = (
        await db.execute(
            select(Document).where(Document.id.in_(doc_ids))
        )
    ).scalars().all()

    # Fetch matched facets per doc for display.
    crit_pairs = {
        (c.get("key"), (c.get("value_norm") or c.get("value") or "").lower().strip())
        for c in criteria
    }
    facet_rows = (
        await db.execute(
            select(DocumentFacet).where(DocumentFacet.document_id.in_(doc_ids))
        )
    ).scalars().all()
    matched_by_doc: dict[str, list[dict]] = {}
    for f in facet_rows:
        if (f.key, f.value_norm) in crit_pairs:
            matched_by_doc.setdefault(f.document_id, []).append(
                {"key": f.key, "label": f.label, "value": f.value}
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
        for d in docs
    ]
    result.sort(key=lambda d: (-len(d["matched"]), d["filename"].lower()))
    return result


async def build_facet_graph(
    db: AsyncSession,
    owner_id: str,
    criteria: list[dict],
    *,
    match: str = "all",
) -> dict:
    """Build a document-centric graph for the documents matching *criteria*.

    Nodes are documents plus facet-value "hub" nodes; edges connect a document to
    each facet value it carries (restricted to the criteria field keys so the
    graph stays focused on the dimensions the user is exploring).
    """
    docs = await filter_documents(db, owner_id, criteria, match=match)
    if not docs:
        return {"nodes": [], "edges": []}

    doc_ids = [d["id"] for d in docs]
    focus_keys = {c.get("key") for c in criteria if c.get("key")}

    facet_rows = (
        await db.execute(
            select(DocumentFacet).where(DocumentFacet.document_id.in_(doc_ids))
        )
    ).scalars().all()

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
        if focus_keys and f.key not in focus_keys:
            continue
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


async def get_document_facets(db: AsyncSession, doc_id: str) -> list[dict]:
    """Return all facets for a single document, ordered by field."""
    rows = (
        await db.execute(
            select(DocumentFacet)
            .where(DocumentFacet.document_id == doc_id)
            .order_by(DocumentFacet.key, DocumentFacet.value)
        )
    ).scalars().all()
    return [
        {"key": f.key, "label": f.label, "value": f.value, "confidence": f.confidence}
        for f in rows
    ]


async def get_stats(db: AsyncSession, owner_id: str) -> dict:
    """Summary counts for the Document Map header / reprocess affordance."""
    total = (
        await db.execute(
            select(func.count(Document.id)).where(Document.owner_id == owner_id)
        )
    ).scalar_one()
    profiled = (
        await db.execute(
            select(func.count(distinct(DocumentFacet.document_id)))
            .join(Document, Document.id == DocumentFacet.document_id)
            .where(Document.owner_id == owner_id)
        )
    ).scalar_one()
    by_cat_rows = (
        await db.execute(
            select(Document.doc_category, func.count(Document.id))
            .where(Document.owner_id == owner_id, Document.doc_category.isnot(None))
            .group_by(Document.doc_category)
        )
    ).all()
    return {
        "total_documents": int(total or 0),
        "profiled_documents": int(profiled or 0),
        "by_category": {row[0]: int(row[1]) for row in by_cat_rows},
    }
