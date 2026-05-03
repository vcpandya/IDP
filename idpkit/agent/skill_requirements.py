"""Parse and check skill compatibility manifests.

Skills declare their dependencies in YAML frontmatter:

    ---
    name: my-slack-bot
    description: Posts daily reports to Slack.
    requires:
      connectors: [slack]
      tools: [slack_send_message, search_document]
    allowed-tools: [slack_send_message]   # Claude-Code-style alias
    ---

This module:
- Normalises the various accepted shapes into a single dict.
- Compares a skill's requirements against the user's active connections to
  produce a ready/missing summary used by the import-preview UI and the
  Connections page.
"""
from __future__ import annotations

from typing import Any

from idpkit.connectors.registry import REGISTRY, tool_to_connector_map


def parse_requirements(frontmatter: dict[str, Any]) -> dict[str, list[str]]:
    """Return {'connectors': [...], 'tools': [...]} from a parsed YAML frontmatter dict."""
    out_connectors: list[str] = []
    out_tools: list[str] = []

    requires = frontmatter.get("requires") or {}
    if isinstance(requires, dict):
        c = requires.get("connectors")
        if isinstance(c, list):
            out_connectors.extend(str(x).strip().lower() for x in c if x)
        elif isinstance(c, str):
            out_connectors.append(c.strip().lower())
        t = requires.get("tools")
        if isinstance(t, list):
            out_tools.extend(str(x).strip() for x in t if x)
        elif isinstance(t, str):
            out_tools.append(t.strip())

    # `allowed-tools` (Claude Code spec) — additive
    allowed = frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools")
    if isinstance(allowed, list):
        out_tools.extend(str(x).strip() for x in allowed if x)
    elif isinstance(allowed, str):
        out_tools.extend(s.strip() for s in allowed.split(",") if s.strip())

    # `connectors` shorthand at top level
    cshort = frontmatter.get("connectors")
    if isinstance(cshort, list):
        out_connectors.extend(str(x).strip().lower() for x in cshort if x)

    # Infer missing connectors from referenced tools (e.g. requires `slack_send_message`
    # implies the `slack` connector is needed).
    tmap = tool_to_connector_map()
    for tname in out_tools:
        cid = tmap.get(tname)
        if cid and cid not in out_connectors:
            out_connectors.append(cid)

    # Dedupe but preserve order
    return {
        "connectors": list(dict.fromkeys(out_connectors)),
        "tools": list(dict.fromkeys(out_tools)),
    }


def check_compatibility(
    requirements: dict[str, list[str]],
    active_connector_ids: set[str],
) -> dict[str, Any]:
    """Compare requirements against the user's connected integrations.

    Returns a structured checklist consumable by the import-preview UI.
    """
    needed = requirements.get("connectors", []) if requirements else []
    items: list[dict] = []
    missing: list[str] = []
    for cid in needed:
        conn = REGISTRY.get(cid)
        if conn is None:
            items.append({
                "id": cid,
                "display_name": cid,
                "status": "unknown",
                "message": f"Skill requires unknown connector '{cid}'.",
            })
            continue
        if cid in active_connector_ids:
            items.append({
                "id": cid,
                "display_name": conn.display_name,
                "status": "ok",
                "message": "Connected.",
            })
        else:
            items.append({
                "id": cid,
                "display_name": conn.display_name,
                "status": "missing",
                "message": "Connect this integration before using the skill.",
            })
            missing.append(cid)
    return {
        "ready": len(missing) == 0,
        "items": items,
        "missing_connectors": missing,
        "required_tools": (requirements or {}).get("tools", []),
    }
