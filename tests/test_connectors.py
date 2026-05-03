"""Tests for the connector framework — encryption, manifest parsing,
runtime tool registration, and the /api/connectors endpoint surface.

Network-bound tools are not exercised here (they are covered by their
respective per-connector integration tests once real creds are available).
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Crypto round-trip
# ---------------------------------------------------------------------------

def test_encrypt_decrypt_round_trip(_env):
    from idpkit.connectors.crypto import decrypt_credentials, encrypt_credentials

    payload = {"bot_token": "xoxb-secret-123", "team": "acme"}
    token = encrypt_credentials(payload)
    assert isinstance(token, str)
    assert "xoxb-secret-123" not in token  # no plaintext in ciphertext
    out = decrypt_credentials(token)
    assert out == payload


def test_decrypt_with_wrong_key_fails(_env, monkeypatch):
    from idpkit.connectors.crypto import decrypt_credentials, encrypt_credentials

    token = encrypt_credentials({"a": 1})
    # Rotate SECRET_KEY → derived Fernet key changes → decrypt must fail
    import idpkit.api.deps as deps_mod
    monkeypatch.setattr(deps_mod, "SECRET_KEY", "different-key-for-rotation-test")
    with pytest.raises(ValueError):
        decrypt_credentials(token)


# ---------------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------------

def test_registry_has_all_expected_connectors():
    from idpkit.connectors.registry import REGISTRY

    expected = {"slack", "notion", "github", "linear", "hubspot", "dropbox", "s3", "google", "jira"}
    assert expected.issubset(REGISTRY.keys())


def test_every_connector_has_unique_tool_names():
    from idpkit.connectors.registry import REGISTRY, tool_to_connector_map

    seen: set[str] = set()
    for c in REGISTRY.values():
        for t in c.tools:
            assert t.name not in seen, f"Duplicate tool name {t.name}"
            seen.add(t.name)
    # Sanity: tool→connector reverse map covers the same set
    assert set(tool_to_connector_map().keys()) == seen


def test_public_metadata_does_not_expose_executors_or_creds():
    from idpkit.connectors.registry import REGISTRY

    for c in REGISTRY.values():
        meta = c.public_metadata()
        # Must not contain callable executors or credential fields.
        for tool in meta["tools"]:
            assert "executor" not in tool
        for f in meta["fields"]:
            assert "value" not in f


# ---------------------------------------------------------------------------
# Skill requirements parsing
# ---------------------------------------------------------------------------

def test_parse_requirements_basic():
    from idpkit.agent.skill_requirements import parse_requirements

    fm = {"requires": {"connectors": ["slack"], "tools": ["slack_send_message"]}}
    out = parse_requirements(fm)
    assert out["connectors"] == ["slack"]
    assert "slack_send_message" in out["tools"]


def test_parse_requirements_infers_connector_from_tool():
    from idpkit.agent.skill_requirements import parse_requirements

    fm = {"requires": {"tools": ["github_create_issue"]}}
    out = parse_requirements(fm)
    assert "github" in out["connectors"]


def test_parse_requirements_allowed_tools_alias():
    from idpkit.agent.skill_requirements import parse_requirements

    fm = {"allowed-tools": ["notion_search_pages", "search_document"]}
    out = parse_requirements(fm)
    assert "notion_search_pages" in out["tools"]
    assert "search_document" in out["tools"]
    assert "notion" in out["connectors"]


def test_check_compatibility_ready_and_missing():
    from idpkit.agent.skill_requirements import check_compatibility

    req = {"connectors": ["slack", "notion"]}
    out = check_compatibility(req, {"slack"})
    assert out["ready"] is False
    assert out["missing_connectors"] == ["notion"]
    statuses = {i["id"]: i["status"] for i in out["items"]}
    assert statuses == {"slack": "ok", "notion": "missing"}

    out2 = check_compatibility(req, {"slack", "notion"})
    assert out2["ready"] is True
    assert out2["missing_connectors"] == []


def test_capability_prompt_section_lists_available_and_unavailable():
    from idpkit.connectors.runtime import build_capability_prompt_section

    class _Conn:  # minimal stub matching the .connector_id attribute
        def __init__(self, cid):
            self.connector_id = cid

    txt = build_capability_prompt_section([_Conn("slack")])
    assert "Connector Availability" in txt
    assert "Slack" in txt
    # Other registered connectors are unavailable
    assert "Notion" in txt or "GitHub" in txt
    assert "Not connected" in txt


def test_capability_prompt_section_no_connections():
    from idpkit.connectors.runtime import build_capability_prompt_section
    txt = build_capability_prompt_section([])
    assert "no external integrations connected" in txt


# ---------------------------------------------------------------------------
# Skill importer wires requirements through to ParsedSkill
# ---------------------------------------------------------------------------

def test_skill_import_extracts_requirements():
    from idpkit.agent.skill_import import import_from_md_bytes

    md = b"""---
name: my-test-skill
description: A test skill.
requires:
  connectors: [slack]
  tools: [slack_send_message]
---

# Body
Do stuff.
"""
    parsed = import_from_md_bytes(md)
    assert parsed.requirements["connectors"] == ["slack"]
    assert "slack_send_message" in parsed.requirements["tools"]
    preview = parsed.to_preview_dict()
    assert preview["requirements"]["connectors"] == ["slack"]


# ---------------------------------------------------------------------------
# HTTP API smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_connectors_endpoint(auth_client):
    res = await auth_client.get("/api/connectors")
    assert res.status_code == 200
    data = res.json()
    ids = {c["id"] for c in data["connectors"]}
    for needed in ("slack", "notion", "github", "linear", "hubspot", "dropbox", "s3", "google", "jira"):
        assert needed in ids
    # Sanity: tools and fields are present, no executor leaked
    one = data["connectors"][0]
    assert "tools" in one and "fields" in one
    assert "auth_type" in one


@pytest.mark.asyncio
async def test_list_user_connections_initially_empty(auth_client):
    res = await auth_client.get("/api/connectors/connections")
    assert res.status_code == 200
    assert res.json() == {"connections": []}


@pytest.mark.asyncio
async def test_connect_validates_required_fields(auth_client):
    # Slack requires bot_token; missing it must 400
    res = await auth_client.post("/api/connectors/slack/connect", json={"credentials": {}})
    assert res.status_code == 400
    assert "bot_token" in res.text.lower() or "required" in res.text.lower()


@pytest.mark.asyncio
async def test_connect_unknown_connector_404(auth_client):
    res = await auth_client.post(
        "/api/connectors/does-not-exist/connect",
        json={"credentials": {"x": "y"}},
    )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_oauth_start_requires_oauth_connector(auth_client):
    # Slack is api_key, not oauth — must reject
    res = await auth_client.get("/api/connectors/slack/oauth/start")
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_oauth_start_for_google_without_env_returns_503(auth_client, monkeypatch):
    # Without GOOGLE_OAUTH_CLIENT_ID set, the start endpoint must surface a
    # configuration error rather than crash.
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET", raising=False)
    res = await auth_client.get("/api/connectors/google/oauth/start")
    assert res.status_code == 503
    assert "oauth" in res.text.lower() or "missing" in res.text.lower()


# ---------------------------------------------------------------------------
# Connect-and-disconnect end-to-end with a stubbed health check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_persists_encrypted_creds_and_disconnect_removes(auth_client, monkeypatch):
    # Stub Slack health_check to avoid network — return ok.
    from idpkit.connectors.impl import slack as slack_mod

    async def _fake_health(creds):
        assert creds.get("bot_token") == "xoxb-test-token"
        return True, "fake-team / fake-bot"

    monkeypatch.setattr(slack_mod.CONNECTOR, "health_check", _fake_health)

    res = await auth_client.post(
        "/api/connectors/slack/connect",
        json={"credentials": {"bot_token": "xoxb-test-token"}},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["connector_id"] == "slack"
    assert body["status"] == "active"
    conn_id = body["id"]

    listing = (await auth_client.get("/api/connectors/connections")).json()
    assert any(c["id"] == conn_id for c in listing["connections"])

    # Verify the stored credential is encrypted (not plaintext) by reading the row directly.
    from sqlalchemy import select
    from idpkit.db.session import async_session
    from idpkit.db.models import Connection
    async with async_session() as db:
        row = (await db.execute(select(Connection).where(Connection.id == conn_id))).scalar_one()
        assert "xoxb-test-token" not in row.encrypted_credentials

    # Disconnect.
    del_res = await auth_client.delete(f"/api/connectors/connections/{conn_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True


# ---------------------------------------------------------------------------
# Skill import preview now carries compatibility info
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_skill_import_preview_includes_compatibility(auth_client):
    md = (
        "---\n"
        "name: needs-slack\n"
        "description: requires slack.\n"
        "requires:\n"
        "  connectors: [slack]\n"
        "---\n\n"
        "Body."
    )
    res = await auth_client.post(
        "/api/skills/import",
        files={"file": ("SKILL.md", md.encode("utf-8"), "text/markdown")},
        data={"preview": "true"},
    )
    assert res.status_code == 200, res.text
    preview = res.json()["preview"]
    assert preview["requirements"]["connectors"] == ["slack"]
    compat = preview["compatibility"]
    assert compat is not None
    assert compat["ready"] is False
    assert "slack" in compat["missing_connectors"]
