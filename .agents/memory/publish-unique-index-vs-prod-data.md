---
name: Publish-time unique index vs. existing prod data
description: Why DB unique indexes that prod data could violate break Replit deploys, and the advisory-lock alternative.
---

# Publish-time unique index creation can abort a deploy

**Rule:** Do not introduce a DB-level UNIQUE index/constraint whose uniqueness the *existing production data* might violate. Replit's Publish flow diffs the dev DB schema against prod and applies the diff (e.g. `CREATE UNIQUE INDEX ...`) to production **before the app boots**. So any app-startup data-cleanup migration (`_migrate_dedupe_tags` etc.) runs *too late* — the index creation fails on the still-dirty prod rows and the whole deploy aborts.

**How to spot it:** deploy error "Failed to run database migration statement / could not create unique index ...". The failing DDL is in introspected form (quoted identifiers, `USING btree`, `text_ops`) — that's the publish differ replicating a dev index, NOT your guarded `CREATE ... IF NOT EXISTS` startup code.

**Why you can't just fix prod:** `executeSql({environment:"production"})` is read-only (SELECT only); the agent must not script prod migrations or work around the publish flow (see database skill + database-migrations-on-publish.md).

**The fix that worked (tags case):** drop the unique index from both the dev DB *and* the code that recreates it at startup (a physical index in dev is what the differ replicates), and enforce uniqueness at the application layer with a transaction-scoped advisory lock instead: `pg_advisory_xact_lock(hashtext(:k))` keyed on the logical key, acquired before the check-then-insert, released at commit. Keep the idempotent DML dedupe migration to clean legacy rows on boot. Cover ALL writer paths (create, rename, every create-or-reuse site). When one transaction grabs multiple such locks, acquire them in a deterministic sorted order to avoid AB/BA deadlocks. Pattern already used in `metadata/extractor.py` and now `db/session.py:lock_tag_name`.
