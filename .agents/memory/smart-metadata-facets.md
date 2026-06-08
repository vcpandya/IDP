---
name: Smart Metadata facet filtering
description: Why DocumentFacet AND/OR filtering uses a single GROUP BY/HAVING count and what that relies on
---

# Document Map facet filtering (idpkit/metadata/queries.py)

`filter_documents` resolves AND/OR multi-criterion facet filters with a single
grouped aggregation, not one query per criterion:

- De-duplicate the `(key, value_norm)` criteria pairs in Python first.
- `select(document_id).where(owner, tuple_(key, value_norm).in_(pairs)).group_by(document_id)`.
- `match="all"` → `.having(func.count() == len(pairs))`; `match="any"` → no HAVING.

**Why:** the `HAVING count() == len(pairs)` AND-logic is only correct because a
matched row count can equal the criteria count *only* if every distinct pair
matched once. That depends on two invariants both being true:
1. The unique constraint `uq_facet_doc_key_value (document_id, key, value_norm)`
   on `DocumentFacet` — guarantees a doc matches any one pair at most once, so
   duplicates can't inflate the count.
2. The in-code pair de-duplication — so `len(pairs)` is the true distinct count.

**How to apply:** if you ever drop/relax that unique constraint, or stop
de-duplicating pairs, the AND filter silently over- or under-matches. Keep both
in lockstep. Capped queries (`MAX_FILTER_DOCS`, `MAX_GRAPH_FACET_ROWS`,
`MAX_FACET_ROWS`) must keep a deterministic `ORDER BY` before `LIMIT` so a
truncated subset is stable across calls.
