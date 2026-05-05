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


def test_capability_prompt_section_lists_connected_only_when_no_skills():
    from idpkit.connectors.runtime import build_capability_prompt_section

    class _Conn:
        def __init__(self, cid):
            self.connector_id = cid

    # No active skills → no "missing required" section is shown.
    txt = build_capability_prompt_section([_Conn("slack")], active_skills=[])
    assert "Connector Availability" in txt
    assert "Slack" in txt
    assert "Required by installed skills" not in txt


def test_capability_prompt_section_calls_out_skill_required_missing():
    from idpkit.connectors.runtime import build_capability_prompt_section

    class _Conn:
        def __init__(self, cid):
            self.connector_id = cid

    skills = [
        {"name": "daily-digest", "requirements": {"connectors": ["slack", "notion"]}},
        {"name": "ticket-bot", "requirements": {"connectors": ["github"]}},
    ]
    txt = build_capability_prompt_section([_Conn("slack")], active_skills=skills)
    assert "Notion" in txt and "GitHub" in txt
    assert "Required by installed skills" in txt
    assert "daily-digest" in txt
    assert "ticket-bot" in txt
    # Connectors NOT mentioned by any skill must NOT be listed (e.g. dropbox).
    assert "Dropbox" not in txt


def test_capability_prompt_section_no_connections_no_skills():
    from idpkit.connectors.runtime import build_capability_prompt_section
    txt = build_capability_prompt_section([], active_skills=[])
    assert "no external integrations connected" in txt
    assert "Required by installed skills" not in txt


# ---------------------------------------------------------------------------
# OAuth redirect URI pinning
# ---------------------------------------------------------------------------

def test_oauth_redirect_uri_uses_explicit_base_when_set(monkeypatch):
    from types import SimpleNamespace
    from idpkit.api.routes.connectors import _oauth_redirect_uri

    monkeypatch.setenv("OAUTH_REDIRECT_BASE_URL", "https://idpkit.example.com")
    monkeypatch.delenv("OAUTH_ALLOWED_HOSTS", raising=False)
    fake_req = SimpleNamespace(
        url=SimpleNamespace(hostname="attacker.evil"),
        base_url="https://attacker.evil/",
    )
    uri = _oauth_redirect_uri(fake_req)
    assert uri == "https://idpkit.example.com/api/connectors/oauth/callback"


def test_oauth_redirect_uri_rejects_unallowed_host(monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException
    from idpkit.api.routes.connectors import _oauth_redirect_uri

    monkeypatch.delenv("OAUTH_REDIRECT_BASE_URL", raising=False)
    monkeypatch.setenv("OAUTH_ALLOWED_HOSTS", "idpkit.example.com,localhost")
    fake_req = SimpleNamespace(
        url=SimpleNamespace(hostname="attacker.evil"),
        base_url="https://attacker.evil/",
    )
    with pytest.raises(HTTPException) as ei:
        _oauth_redirect_uri(fake_req)
    assert ei.value.status_code == 400


def test_oauth_redirect_uri_falls_back_to_request_in_dev(monkeypatch):
    from types import SimpleNamespace
    from idpkit.api.routes.connectors import _oauth_redirect_uri

    monkeypatch.delenv("OAUTH_REDIRECT_BASE_URL", raising=False)
    monkeypatch.delenv("OAUTH_ALLOWED_HOSTS", raising=False)
    fake_req = SimpleNamespace(
        url=SimpleNamespace(hostname="localhost"),
        base_url="http://localhost:5000/",
    )
    uri = _oauth_redirect_uri(fake_req)
    assert uri == "http://localhost:5000/api/connectors/oauth/callback"


# ---------------------------------------------------------------------------
# Pluggable OAuth state store
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oauth_state_store_db_roundtrip(_env, monkeypatch):
    """Default DB-backed store: put then pop returns the payload exactly once."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    from idpkit.connectors import oauth as oauth_mod
    from idpkit.db.session import async_session, init_db

    await init_db()
    oauth_mod.reset_state_store()
    assert isinstance(oauth_mod.get_state_store(), oauth_mod.DBOAuthStateStore)

    async with async_session() as db:
        token = await oauth_mod.new_state(db, {"user_id": "u1", "connector_id": "google"})
    async with async_session() as db:
        payload = await oauth_mod.consume_state(db, token)
    assert payload == {"user_id": "u1", "connector_id": "google"}
    # Second consume must return None (single-use).
    async with async_session() as db:
        assert await oauth_mod.consume_state(db, token) is None
    oauth_mod.reset_state_store()


@pytest.mark.asyncio
async def test_oauth_state_store_uses_redis_when_configured(_env, monkeypatch):
    """When REDIS_URL is set and the redis package is importable, the Redis
    backend is selected and a put/pop round-trip works without touching the DB."""
    from idpkit.connectors import oauth as oauth_mod

    class _FakeRedis:
        def __init__(self):
            self.store: dict[str, tuple[str, int | None]] = {}
            self.set_calls = 0
            self.getdel_calls = 0

        async def set(self, key, value, ex=None):
            self.set_calls += 1
            self.store[key] = (value, ex)

        async def getdel(self, key):
            self.getdel_calls += 1
            entry = self.store.pop(key, None)
            return None if entry is None else entry[0]

    fake = _FakeRedis()

    import sys
    import types
    fake_redis_pkg = types.ModuleType("redis")
    fake_async_mod = types.ModuleType("redis.asyncio")

    def _from_url(url, decode_responses=True):
        fake._url = url
        fake._decode_responses = decode_responses
        return fake

    fake_async_mod.from_url = _from_url
    fake_redis_pkg.asyncio = fake_async_mod
    monkeypatch.setitem(sys.modules, "redis", fake_redis_pkg)
    monkeypatch.setitem(sys.modules, "redis.asyncio", fake_async_mod)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    oauth_mod.reset_state_store()
    store = oauth_mod.get_state_store()
    assert isinstance(store, oauth_mod.RedisOAuthStateStore)

    # ``db`` is unused by the Redis backend — pass None to prove it.
    token = await oauth_mod.new_state(None, {"user_id": "u2", "connector_id": "linear"})
    assert fake.set_calls == 1
    # TTL must be applied so Redis expires the key for us (no manual prune).
    (_, ttl), = list(fake.store.values())
    assert ttl == int(oauth_mod.STATE_TTL.total_seconds())

    payload = await oauth_mod.consume_state(None, token)
    assert payload == {"user_id": "u2", "connector_id": "linear"}
    # Single-use: a second pop returns None.
    assert await oauth_mod.consume_state(None, token) is None

    oauth_mod.reset_state_store()
    monkeypatch.delenv("REDIS_URL", raising=False)


@pytest.mark.asyncio
async def test_oauth_state_store_falls_back_when_redis_missing(_env, monkeypatch):
    """If REDIS_URL is set but the redis package can't be imported, the store
    silently falls back to the DB backend rather than crashing on startup."""
    from idpkit.connectors import oauth as oauth_mod

    import builtins
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "redis" or name.startswith("redis."):
            raise ImportError("simulated missing redis")
        return real_import(name, *args, **kwargs)

    import sys
    monkeypatch.delitem(sys.modules, "redis", raising=False)
    monkeypatch.delitem(sys.modules, "redis.asyncio", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    oauth_mod.reset_state_store()
    assert isinstance(oauth_mod.get_state_store(), oauth_mod.DBOAuthStateStore)

    oauth_mod.reset_state_store()
    monkeypatch.delenv("REDIS_URL", raising=False)


# ---------------------------------------------------------------------------
# Runtime OAuth refresh
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_runtime_oauth_refresh_retries_after_auth_error(_env, monkeypatch):
    """If a tool raises ConnectorAuthError but the connector supports refresh
    and we have a refresh_token, the runtime must refresh, persist the new
    token (encrypted), and retry the call once."""
    from idpkit.connectors import encrypt_credentials
    from idpkit.connectors.base import (
        Connector, ConnectorAuthError, ConnectorAuthType, ConnectorTool,
    )
    from idpkit.connectors.crypto import decrypt_credentials
    from idpkit.connectors.runtime import build_runtime_executors
    from idpkit.connectors.registry import REGISTRY
    from idpkit.db.models import Connection
    from idpkit.db.session import init_db, async_session

    await init_db()

    calls = {"count": 0, "refresh_count": 0}

    async def _flaky_executor(args, creds):
        calls["count"] += 1
        if creds.get("access_token") == "old-token":
            raise ConnectorAuthError("expired")
        return {"ok": True, "saw_token": creds["access_token"]}

    async def _refresher(creds):
        calls["refresh_count"] += 1
        assert creds.get("refresh_token") == "rt-1"
        return {"access_token": "new-token", "expires_in": 3600}

    fake = Connector(
        id="fake_oauth",
        display_name="Fake OAuth",
        description="test",
        auth_type=ConnectorAuthType.OAUTH2,
        tools=[ConnectorTool(
            name="fake_oauth_do",
            description="x",
            parameters={"type": "object", "properties": {}},
            executor=_flaky_executor,
        )],
        oauth_refresh=_refresher,
    )
    REGISTRY[fake.id] = fake
    try:
        async with async_session() as db:
            row = Connection(
                owner_id="user-test-refresh",
                connector_id="fake_oauth",
                encrypted_credentials=encrypt_credentials({
                    "access_token": "old-token", "refresh_token": "rt-1",
                }),
                status="active",
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
            row_id = row.id

            execs = build_runtime_executors(db, "user-test-refresh")
            assert "fake_oauth_do" in execs
            result = await execs["fake_oauth_do"]({}, None, db)
            assert result == {"ok": True, "saw_token": "new-token"}
            assert calls["refresh_count"] == 1
            assert calls["count"] == 2  # called twice (failed then succeeded)

            await db.refresh(row)
            stored = decrypt_credentials(row.encrypted_credentials)
            assert stored["access_token"] == "new-token"
            assert stored["refresh_token"] == "rt-1"  # preserved
            assert row.status == "active"
    finally:
        REGISTRY.pop("fake_oauth", None)


@pytest.mark.asyncio
async def test_runtime_marks_disconnected_when_refresh_fails(_env, monkeypatch):
    from idpkit.connectors import encrypt_credentials
    from idpkit.connectors.base import (
        Connector, ConnectorAuthError, ConnectorAuthType, ConnectorTool,
    )
    from idpkit.connectors.runtime import build_runtime_executors
    from idpkit.connectors.registry import REGISTRY
    from idpkit.db.models import Connection
    from idpkit.db.session import init_db, async_session

    await init_db()

    async def _always_fails(args, creds):
        raise ConnectorAuthError("expired")

    async def _refresh_fails(creds):
        raise ConnectorAuthError("refresh denied")

    fake = Connector(
        id="fake_oauth_bad",
        display_name="Fake OAuth Bad",
        description="test",
        auth_type=ConnectorAuthType.OAUTH2,
        tools=[ConnectorTool(
            name="fake_oauth_bad_do",
            description="x",
            parameters={"type": "object", "properties": {}},
            executor=_always_fails,
        )],
        oauth_refresh=_refresh_fails,
    )
    REGISTRY[fake.id] = fake
    try:
        async with async_session() as db:
            row = Connection(
                owner_id="user-bad-refresh",
                connector_id="fake_oauth_bad",
                encrypted_credentials=encrypt_credentials({
                    "access_token": "x", "refresh_token": "y",
                }),
                status="active",
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)

            execs = build_runtime_executors(db, "user-bad-refresh")
            result = await execs["fake_oauth_bad_do"]({}, None, db)
            assert "error" in result

            await db.refresh(row)
            assert row.status == "disconnected"
            assert "refresh failed" in (row.last_error or "")
    finally:
        REGISTRY.pop("fake_oauth_bad", None)


# ---------------------------------------------------------------------------
# Google has wave-1 tools
# ---------------------------------------------------------------------------

def test_google_connector_includes_drive_gmail_sheets_calendar():
    from idpkit.connectors.registry import get_connector
    g = get_connector("google")
    names = {t.name for t in g.tools}
    assert {
        "google_drive_search",
        "google_gmail_send",
        "google_sheets_read_range",
        "google_sheets_append_row",
        "google_calendar_list_events",
        "google_calendar_create_event",
    }.issubset(names)


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


# ---------------------------------------------------------------------------
# Org-wide shared connections (Task #10)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_share_unshare_connection_admin_only(auth_client, monkeypatch):
    """Admin can share their own connection org-wide and revoke sharing."""
    from idpkit.connectors.impl import slack as slack_mod

    async def _fake_health(creds):
        return True, "fake-team"

    monkeypatch.setattr(slack_mod.CONNECTOR, "health_check", _fake_health)

    res = await auth_client.post(
        "/api/connectors/slack/connect",
        json={"credentials": {"bot_token": "xoxb-share-test"}},
    )
    assert res.status_code == 200
    conn_id = res.json()["id"]

    listing = (await auth_client.get("/api/connectors/connections")).json()
    me = next(c for c in listing["connections"] if c["id"] == conn_id)
    assert me["scope"] == "private" and me["is_shared"] is False
    assert me["is_owner"] is True

    shared = await auth_client.post(f"/api/connectors/connections/{conn_id}/share")
    assert shared.status_code == 200, shared.text
    assert shared.json()["scope"] == "org"
    assert shared.json()["is_shared"] is True
    assert shared.json()["owner_org"] == "default"

    unshared = await auth_client.post(f"/api/connectors/connections/{conn_id}/unshare")
    assert unshared.status_code == 200
    assert unshared.json()["scope"] == "private"

    await auth_client.delete(f"/api/connectors/connections/{conn_id}")


@pytest.mark.asyncio
async def test_non_admin_sees_shared_connection_and_cannot_disconnect(auth_client, client, monkeypatch):
    """A regular user sees shared connections, can use them via the runtime,
    but cannot share, unshare, or disconnect them; audit rows record the use."""
    from idpkit.connectors.impl import slack as slack_mod
    from idpkit.connectors import encrypt_credentials
    from idpkit.connectors.base import (
        Connector, ConnectorAuthType, ConnectorTool,
    )
    from idpkit.connectors.registry import REGISTRY
    from idpkit.connectors.runtime import (
        build_runtime_executors, get_active_connection, list_active_connections,
    )
    from idpkit.db.models import Connection, ConnectionAuditLog, User
    from idpkit.db.session import async_session
    from idpkit.api.deps import hash_password
    from sqlalchemy import select

    async def _fake_health(creds):
        return True, "fake-team"

    monkeypatch.setattr(slack_mod.CONNECTOR, "health_check", _fake_health)

    # Admin creates and shares a connection.
    res = await auth_client.post(
        "/api/connectors/slack/connect",
        json={"credentials": {"bot_token": "xoxb-org-shared"}},
    )
    conn_id = res.json()["id"]
    share = await auth_client.post(f"/api/connectors/connections/{conn_id}/share")
    assert share.status_code == 200

    # Create a regular (active) user and log in as them.
    async with async_session() as db:
        u = User(
            username="member-user",
            hashed_password=hash_password("memberpw"),
            role="user",
            is_active=1,
        )
        db.add(u)
        await db.commit()
        await db.refresh(u)
        member_id = u.id

    login = await client.post(
        "/api/auth/login",
        json={"username": "member-user", "password": "memberpw"},
    )
    member_token = login.json()["access_token"]
    member_headers = {"Authorization": f"Bearer {member_token}"}

    # Member sees the shared connection.
    listing = (await client.get(
        "/api/connectors/connections", headers=member_headers,
    )).json()
    seen = next(c for c in listing["connections"] if c["id"] == conn_id)
    assert seen["is_shared"] is True
    assert seen["is_owner"] is False

    # Member cannot share/unshare or disconnect.
    assert (await client.post(
        f"/api/connectors/connections/{conn_id}/share", headers=member_headers,
    )).status_code == 403
    assert (await client.post(
        f"/api/connectors/connections/{conn_id}/unshare", headers=member_headers,
    )).status_code == 403
    assert (await client.delete(
        f"/api/connectors/connections/{conn_id}", headers=member_headers,
    )).status_code == 403

    # Runtime lookup: member resolves the shared connection.
    async with async_session() as db:
        active = await list_active_connections(db, member_id)
        assert any(c.id == conn_id for c in active)
        resolved = await get_active_connection(db, member_id, "slack")
        assert resolved is not None and resolved.id == conn_id

    # Register an audit-friendly fake connector and a shared connection.
    calls = {"n": 0}

    async def _ok(args, creds):
        calls["n"] += 1
        return {"ok": True}

    fake = Connector(
        id="fake_shared",
        display_name="Fake Shared",
        description="t",
        auth_type=ConnectorAuthType.API_KEY,
        tools=[ConnectorTool(
            name="fake_shared_do",
            description="x",
            parameters={"type": "object", "properties": {}},
            executor=_ok,
        )],
    )
    REGISTRY[fake.id] = fake
    try:
        async with async_session() as db:
            row = Connection(
                owner_id="some-other-admin",
                connector_id="fake_shared",
                encrypted_credentials=encrypt_credentials({"k": "v"}),
                status="active",
                scope="org",
                owner_org="default",
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
            shared_conn_id = row.id

            execs = build_runtime_executors(db, member_id)
            result = await execs["fake_shared_do"]({}, None, db)
            assert result == {"ok": True}

            audit_rows = (await db.execute(
                select(ConnectionAuditLog).where(
                    ConnectionAuditLog.connection_id == shared_conn_id,
                )
            )).scalars().all()
            assert len(audit_rows) == 1
            assert audit_rows[0].user_id == member_id
            assert audit_rows[0].tool_name == "fake_shared_do"
            assert audit_rows[0].success == 1
    finally:
        REGISTRY.pop("fake_shared", None)

    # Audit endpoint visible to the admin (connection owner is admin).
    audit = await auth_client.get(f"/api/connectors/connections/{conn_id}/audit")
    assert audit.status_code == 200
    # No usage on this particular connection (slack one) yet, so empty list.
    assert audit.json()["events"] == []

    # Cleanup the slack connection.
    await auth_client.delete(f"/api/connectors/connections/{conn_id}")


# ---------------------------------------------------------------------------
# Live integration tests (Task #12)
# ---------------------------------------------------------------------------
#
# These tests exercise one read-only tool per connector against a real sandbox
# account. They are gated on per-connector env vars (see the ``*_live_creds``
# fixtures in conftest.py) — when the secret is not set the test is silently
# skipped, so local ``pytest`` runs and CI jobs without the secrets stay green.
#
# Run only the live suite (e.g. on a nightly job):    pytest -m live
# Skip the live suite (default for fast feedback):    pytest -m "not live"
#
# Each test deliberately calls a non-mutating tool (list / search / health) so
# repeated runs don't pollute the sandbox account.

@pytest.mark.live
async def test_live_slack_list_channels(slack_live_creds):
    from idpkit.connectors.impl.slack import CONNECTOR

    ok, label = await CONNECTOR.health_check(slack_live_creds)
    assert ok and label
    list_tool = next(t for t in CONNECTOR.tools if t.name == "slack_list_channels")
    out = await list_tool.executor({"limit": 5}, slack_live_creds)
    assert "channels" in out, out
    assert isinstance(out["channels"], list)


@pytest.mark.live
async def test_live_notion_search_pages(notion_live_creds):
    from idpkit.connectors.impl.notion import CONNECTOR

    ok, label = await CONNECTOR.health_check(notion_live_creds)
    assert ok and label
    search_tool = next(t for t in CONNECTOR.tools if t.name == "notion_search_pages")
    out = await search_tool.executor({"query": "", "page_size": 5}, notion_live_creds)
    assert "results" in out, out
    assert isinstance(out["results"], list)


@pytest.mark.live
async def test_live_github_list_repos(github_live_creds):
    from idpkit.connectors.impl.github import CONNECTOR

    ok, label = await CONNECTOR.health_check(github_live_creds)
    assert ok and label.startswith("@")
    list_tool = next(t for t in CONNECTOR.tools if t.name == "github_list_repos")
    out = await list_tool.executor({"per_page": 5}, github_live_creds)
    assert "repos" in out, out
    assert isinstance(out["repos"], list)


@pytest.mark.live
async def test_live_linear_list_issues(linear_live_creds):
    from idpkit.connectors.impl.linear import CONNECTOR

    ok, label = await CONNECTOR.health_check(linear_live_creds)
    assert ok and label
    list_tool = next(t for t in CONNECTOR.tools if t.name == "linear_list_issues")
    out = await list_tool.executor({"first": 5}, linear_live_creds)
    assert "issues" in out, out
    assert isinstance(out["issues"], list)


@pytest.mark.live
async def test_live_hubspot_search_contacts(hubspot_live_creds):
    from idpkit.connectors.impl.hubspot import CONNECTOR

    ok, label = await CONNECTOR.health_check(hubspot_live_creds)
    assert ok and label
    search_tool = next(t for t in CONNECTOR.tools if t.name == "hubspot_search_contacts")
    out = await search_tool.executor({"query": "test", "limit": 5}, hubspot_live_creds)
    assert "results" in out, out
    assert isinstance(out["results"], list)


@pytest.mark.live
async def test_live_dropbox_list_files(dropbox_live_creds):
    from idpkit.connectors.impl.dropbox import CONNECTOR

    ok, label = await CONNECTOR.health_check(dropbox_live_creds)
    assert ok and label
    list_tool = next(t for t in CONNECTOR.tools if t.name == "dropbox_list_files")
    out = await list_tool.executor({"path": "", "limit": 5}, dropbox_live_creds)
    assert "entries" in out, out
    assert isinstance(out["entries"], list)


@pytest.mark.live
async def test_live_jira_health_and_search(jira_live_creds):
    from idpkit.connectors.impl.jira import CONNECTOR

    ok, label = await CONNECTOR.health_check(jira_live_creds)
    assert ok and label
    # ``order by created DESC`` is universally valid JQL even on empty projects.
    search_tool = next(t for t in CONNECTOR.tools if t.name == "jira_search_issues")
    out = await search_tool.executor(
        {"jql": "order by created DESC", "max_results": 5}, jira_live_creds,
    )
    assert "issues" in out, out
    assert isinstance(out["issues"], list)


@pytest.mark.live
async def test_live_s3_list_objects(s3_live_creds):
    from idpkit.connectors.impl.s3 import CONNECTOR

    ok, label = await CONNECTOR.health_check(s3_live_creds)
    assert ok and label.startswith("s3://")
    list_tool = next(t for t in CONNECTOR.tools if t.name == "s3_list_objects")
    out = await list_tool.executor({"prefix": "", "max_keys": 5}, s3_live_creds)
    assert "objects" in out, out
    assert isinstance(out["objects"], list)


# ---------------------------------------------------------------------------
# Audit-log retention / pruning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prune_connection_audit_log_respects_retention(_env, monkeypatch):
    """Rows older than the retention window are deleted; recent rows survive,
    and the retention window honours the env var override."""
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import select
    from idpkit.db.models import Connection, ConnectionAuditLog
    from idpkit.db.session import async_session, init_db
    from idpkit.db.audit_prune import prune_connection_audit_log
    from idpkit.connectors import encrypt_credentials

    await init_db()

    async with async_session() as db:
        conn = Connection(
            owner_id="prune-owner",
            connector_id="fake_prune",
            encrypted_credentials=encrypt_credentials({"k": "v"}),
            status="active",
            scope="org",
            owner_org="default",
        )
        db.add(conn)
        await db.commit()
        await db.refresh(conn)
        cid = conn.id

        now = datetime.now(timezone.utc)
        old = ConnectionAuditLog(
            connection_id=cid, connector_id="fake_prune",
            user_id=None, tool_name="t",
            created_at=now - timedelta(days=120),
        )
        recent = ConnectionAuditLog(
            connection_id=cid, connector_id="fake_prune",
            user_id=None, tool_name="t",
            created_at=now - timedelta(days=5),
        )
        db.add_all([old, recent])
        await db.commit()
        old_id, recent_id = old.id, recent.id

    # Default 90-day window: only the 120-day-old row should be deleted.
    deleted = await prune_connection_audit_log(async_session)
    assert deleted == 1

    async with async_session() as db:
        remaining_ids = set((await db.execute(
            select(ConnectionAuditLog.id).where(
                ConnectionAuditLog.connection_id == cid
            )
        )).scalars().all())
    assert recent_id in remaining_ids
    assert old_id not in remaining_ids

    # Env var override: a 1-day window prunes the 5-day-old row too.
    monkeypatch.setenv("CONNECTION_AUDIT_RETENTION_DAYS", "1")
    deleted2 = await prune_connection_audit_log(async_session)
    assert deleted2 == 1

    async with async_session() as db:
        leftover = (await db.execute(
            select(ConnectionAuditLog).where(
                ConnectionAuditLog.connection_id == cid
            )
        )).scalars().all()
    assert leftover == []

    # Cleanup the test connection row.
    async with async_session() as db:
        row = (await db.execute(
            select(Connection).where(Connection.id == cid)
        )).scalar_one()
        await db.delete(row)
        await db.commit()


@pytest.mark.live
async def test_live_google_drive_search(google_live_creds):
    from idpkit.connectors.impl.google import CONNECTOR

    ok, label = await CONNECTOR.health_check(google_live_creds)
    assert ok and label
    search_tool = next(t for t in CONNECTOR.tools if t.name == "google_drive_search")
    out = await search_tool.executor({"query": "", "page_size": 5}, google_live_creds)
    assert "files" in out, out
    assert isinstance(out["files"], list)
