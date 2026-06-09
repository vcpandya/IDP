---
name: Document Map is IDA's cross-document file selector
description: The Document Map's real purpose and why agent-side scoping/denoising (not just the UI) is what makes it work.
---

# Document Map = cross-document pre-filter for the agent

The Document Map (smart-metadata facets) exists primarily so IDA can decide **which files in a knowledge base to run per-document PageIndex (`search_document`) on** — PageIndex itself is per-document and has no across-documents step. The human browse UI is secondary.

The real value flows through the agent tool `query_document_map` (`list_facets` → `filter_documents` → returns `document_id`s → `search_document` on the subset), NOT through the `/facets` UI endpoint.

**Why this matters:** Two quality fixes that only touch the UI endpoint do nothing for the agent path:
1. **Scope to the attachment.** The tool must receive the conversation's resolved document set so it filters *within the attached knowledge base*, not the whole library. Scope is tri-state: non-empty attached set → that list; attachment that resolves to zero docs → `[]` (none); no attachment → `None` (whole library, for discovery). This maps to `metadata/queries.py` semantics (`None`=all, `[]`=none). The route knows "was anything attached" (`bool(tag_ids or document_ids)`); the agent only sees the merged `document_ids`, so that boolean must be threaded in.
2. **Denoise for the agent too.** `list_facets` must hide singleton facet values (`min_count = 1 if (search or key) else 2`) so IDA sees groupable dimensions instead of drowning in unique-per-doc topics/titles — mirror the UI logic in the tool.

**How to apply:** Any "improve Document Map quality" request must be evaluated against the **agent tool path**, and prompt guidance must be reconciled in BOTH places: the static `SYSTEM_PROMPT` Case A rules AND the dynamically-built in-scope block in `_build_messages()`. A "search ALL attached docs" directive in either one will override the "narrow with query_document_map first" guidance and make the map pointless for attached KBs.
