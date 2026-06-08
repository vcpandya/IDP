---
name: Agent owner-scoped tool injection
description: How IDA injects the calling user's id into owner-scoped agent tools, and the gotcha when adding new ones
---

# Owner context for IDA agent tools (idpkit/agent/agent.py + tools.py)

Owner-scoped agent tools read the caller from `args["_user_id"]`. The LLM never
supplies this — `IDPAgent.chat()` and `chat_stream()` inject it just before
`execute_tool`, gated by the `_USER_CONTEXT_TOOLS` allowlist in agent.py.

**Why:** the value must come from the authenticated session, not the model
(otherwise a tool could be tricked into reading another tenant's data). It is
deliberately NOT part of any tool's JSON schema.

**How to apply:** when you add a new first-party tool whose executor needs the
user id (e.g. anything calling owner-scoped `idpkit.metadata.queries`), you MUST
add its name to `_USER_CONTEXT_TOOLS`. There are TWO injection sites (the
non-streaming and streaming loops) — the allowlist covers both. Forgetting this
makes the tool silently fail at runtime with "User context not available" even
though unit-calling the executor with a hand-passed `_user_id` works.

Also: tool results meant for chaining into `search_document`/`extract_data`
should expose the document id under the key `document_id` (alias `id` if the
underlying query returns `id`), since those tools take a `document_id` arg.
