"""Mail-merge field substitution for envelope templates.

Syntax: ``{{key}}`` — replaced with values from the merge dictionary.
- Unknown keys are left intact (so missing data is visible, not silent).
- Whitespace inside the braces is tolerated: ``{{ key }}`` works too.
- Keys are case-insensitive on lookup but the original placeholder casing is preserved on miss.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")


def render_merge(template_text: str | None, values: Dict[str, Any] | None) -> str | None:
    """Substitute ``{{key}}`` placeholders with values. Returns None if input is None."""
    if template_text is None:
        return None
    if not values:
        return template_text
    lower_map = {str(k).lower(): ("" if v is None else str(v)) for k, v in values.items()}

    def _sub(match: re.Match) -> str:
        key = match.group(1).lower()
        if key in lower_map:
            return lower_map[key]
        return match.group(0)  # leave unknown placeholders visible

    return _PLACEHOLDER_RE.sub(_sub, template_text)


def extract_merge_keys(*texts: str | None) -> List[str]:
    """Return distinct merge keys referenced in the given texts, preserving first-seen order."""
    seen: List[str] = []
    seen_lower: set[str] = set()
    for t in texts:
        if not t:
            continue
        for m in _PLACEHOLDER_RE.finditer(t):
            k = m.group(1)
            kl = k.lower()
            if kl not in seen_lower:
                seen_lower.add(kl)
                seen.append(k)
    return seen


def validate_merge_values(required_keys: Iterable[str], values: Dict[str, Any] | None) -> List[str]:
    """Return the list of required keys missing from ``values`` (case-insensitive)."""
    have = {str(k).lower() for k in (values or {}).keys()}
    return [k for k in required_keys if str(k).lower() not in have]
