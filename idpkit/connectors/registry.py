"""Central registry of all available connectors."""
from __future__ import annotations

from typing import Optional

from idpkit.connectors.base import Connector
from idpkit.connectors.impl.dropbox import CONNECTOR as DROPBOX
from idpkit.connectors.impl.github import CONNECTOR as GITHUB
from idpkit.connectors.impl.google import CONNECTOR as GOOGLE
from idpkit.connectors.impl.hubspot import CONNECTOR as HUBSPOT
from idpkit.connectors.impl.jira import CONNECTOR as JIRA
from idpkit.connectors.impl.linear import CONNECTOR as LINEAR
from idpkit.connectors.impl.notion import CONNECTOR as NOTION
from idpkit.connectors.impl.s3 import CONNECTOR as S3
from idpkit.connectors.impl.slack import CONNECTOR as SLACK

REGISTRY: dict[str, Connector] = {
    c.id: c
    for c in (SLACK, NOTION, GITHUB, GOOGLE, LINEAR, JIRA, HUBSPOT, DROPBOX, S3)
}


def get_connector(connector_id: str) -> Optional[Connector]:
    return REGISTRY.get(connector_id)


def list_connectors() -> list[Connector]:
    return list(REGISTRY.values())


def tool_to_connector_map() -> dict[str, str]:
    """Map every connector tool name → its connector id."""
    out: dict[str, str] = {}
    for cid, conn in REGISTRY.items():
        for t in conn.tools:
            out[t.name] = cid
    return out
