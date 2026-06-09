---
name: Stale .pyc on overlay filesystem
description: Persistent NameError/AttributeError for clearly-imported names after edits, caused by stale bytecode cache on the overlay/btrfs workspace.
---

# Stale `.pyc` causing phantom NameError on startup

**Symptom:** App startup crashes with `NameError: name 'X' is not defined` (or similar) on a line where `X` is plainly imported in the current source. Disassembly of a freshly-compiled module shows correct bytecode, yet the running uvicorn process keeps raising the error across restarts. A separately-started healthy process may keep serving while new starts crash, which masks the issue and creates confusing mixed signals.

**Root cause:** The workspace is an `overlay` mount (lowerdirs in `/run/vmgo/.../mnt/*`, btrfs backing). Python's `.pyc` invalidation compares the source mtime stored in the `.pyc` header against the source file mtime; on this filesystem those can falsely match after restores/syncs, so Python loads **old bytecode** even though the `.py` is current. `NameError` from a `LOAD_GLOBAL` (vs `STORE_FAST`/`LOAD_FAST`) is the tell that old bytecode is running.

**Why:** Bytecode cache freshness is not reliable across the overlay layers, so editing the source alone does not guarantee a recompile.

**How to apply / fix (do all three):**
1. Make the name resolvable regardless of bytecode shape — add a **module-level** import as a global fallback in addition to any function-level import (a stale `LOAD_GLOBAL` will then still find it).
2. Purge every `__pycache__` under both the workspace AND any `/mnt/*/working_subv/<pkg>` copy: `find /home/runner/workspace/<pkg> /mnt/*/working_subv/<pkg> -name __pycache__ -type d -prune -exec rm -rf {} +`.
3. Restart the workflow and verify via `refresh_all_logs` (NOT `ls -t /tmp/logs | head`, which can return a stale crash log) that the newest log shows "Application startup complete".

**Gotcha:** `ls -t /tmp/logs/*.log | head -1` returned an old crash log after successful restarts; trust `refresh_all_logs` and `ps -eo pid,etime` (a fresh low-etime process serving 302) for the real state.
