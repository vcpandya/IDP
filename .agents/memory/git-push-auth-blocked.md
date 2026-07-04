---
name: Git push from agent env is unauthenticated
description: Why command-line git push to the GitHub remote fails, and the correct path to push branches.
---

Command-line `git push` to the GitHub `origin` remote fails from the agent/task environment with:
`remote: Invalid username or token. Password authentication is not supported for Git operations.` (exit 128).

**Why:** The shell git has no authorized GitHub credential/token. Replit's Git integration stores auth in the Git pane, not in the CLI environment. So the CLI can `fetch` (public read) and do all local ref inspection, but cannot `push`.

**How to apply:** Do not attempt to resolve a "push my changes to GitHub" request by running `git push` — it will always fail on auth. The user must push through the Replit Git pane (which is authenticated). The agent's role is limited to inspecting divergence (rev-list/merge-base/merge-tree read-only) and explaining conflicts. Also note: `rm` of `.git/**` lock files and any `git` write (push/merge/commit/merge-tree --write-tree) are blocked for the main agent; only read-only git works there.
