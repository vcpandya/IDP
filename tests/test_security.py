"""Tests for the production security hardening (task #30).

Covers: fail-closed SECRET_KEY in production, CORS allowlist pinning,
auth rate limiting, e-sign detail auth guard, CSRF double-submit cookie,
OAuth redirect derived from DEPLOYED_DOMAIN, and that the LLM module no
longer logs API key fragments.
"""
from __future__ import annotations

import importlib
import logging
import os
from types import SimpleNamespace

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# 1. Fail-closed SECRET_KEY in production
# ---------------------------------------------------------------------------

def test_load_secret_key_raises_in_production_without_env(monkeypatch):
    from idpkit.api import deps as deps_mod

    monkeypatch.setenv("DEPLOYED_DOMAIN", "idpkit.example.com")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.delenv("IDP_SECRET_KEY", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    with pytest.raises(RuntimeError) as ei:
        deps_mod._load_secret_key()
    assert "SECRET_KEY" in str(ei.value)


def test_load_secret_key_dev_falls_back_to_ephemeral(monkeypatch, caplog):
    from idpkit.api import deps as deps_mod

    monkeypatch.delenv("DEPLOYED_DOMAIN", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.delenv("IDP_SECRET_KEY", raising=False)

    with caplog.at_level(logging.WARNING, logger="idpkit.api.deps"):
        key = deps_mod._load_secret_key()
    assert isinstance(key, str) and len(key) >= 32
    assert any("ephemeral" in r.message for r in caplog.records)


def test_is_production_true_when_deployed_domain_set(monkeypatch):
    from idpkit.api import deps as deps_mod

    monkeypatch.setenv("DEPLOYED_DOMAIN", "idpkit.example.com")
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert deps_mod.is_production() is True

    monkeypatch.delenv("DEPLOYED_DOMAIN", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert deps_mod.is_production() is True

    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert deps_mod.is_production() is False


# ---------------------------------------------------------------------------
# 2. OAuth redirect URI derives from DEPLOYED_DOMAIN
# ---------------------------------------------------------------------------

def test_oauth_redirect_uri_uses_deployed_domain(monkeypatch):
    from idpkit.api.routes.connectors import _oauth_redirect_uri

    monkeypatch.delenv("OAUTH_REDIRECT_BASE_URL", raising=False)
    monkeypatch.delenv("OAUTH_ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("DEPLOYED_DOMAIN", "idpkit.example.com")
    fake_req = SimpleNamespace(
        url=SimpleNamespace(hostname="attacker.evil"),
        base_url="https://attacker.evil/",
    )
    uri = _oauth_redirect_uri(fake_req)
    assert uri == "https://idpkit.example.com/api/connectors/oauth/callback"


def test_oauth_redirect_uri_explicit_base_overrides_deployed_domain(monkeypatch):
    from idpkit.api.routes.connectors import _oauth_redirect_uri

    monkeypatch.setenv("DEPLOYED_DOMAIN", "should-not-be-used.example")
    monkeypatch.setenv("OAUTH_REDIRECT_BASE_URL", "https://canonical.example.com")
    fake_req = SimpleNamespace(
        url=SimpleNamespace(hostname="x"),
        base_url="https://x/",
    )
    uri = _oauth_redirect_uri(fake_req)
    assert uri == "https://canonical.example.com/api/connectors/oauth/callback"


# ---------------------------------------------------------------------------
# 3. e-sign detail page is auth-guarded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_esign_detail_redirects_to_login_when_unauthenticated(client):
    res = await client.get("/esign/some-random-id/detail", follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["location"] == "/login"


# ---------------------------------------------------------------------------
# 4. CSRF for cookie-authenticated state-changing routes
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def cookie_client(app):
    """A client authenticated only by session_token cookie (NO Authorization).

    Uses an https base_url so httpx will actually send the Secure cookies set
    by the login endpoint (httpx, like browsers, drops Secure cookies on http).
    """
    import httpx
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="https://testserver") as c:
        res = await c.post("/api/auth/login", json={
            "username": "admin",
            "password": os.environ["IDP_ADMIN_PASSWORD"],
        })
        assert res.status_code == 200, res.text
        assert c.cookies.get("session_token")
        assert c.cookies.get("csrftoken")
        yield c


@pytest.mark.asyncio
async def test_cookie_post_without_csrf_header_is_rejected(cookie_client):
    # /api/auth/apikey is a state-changing POST that requires auth.
    res = await cookie_client.post("/api/auth/apikey")
    assert res.status_code == 403
    assert "CSRF" in res.text


@pytest.mark.asyncio
async def test_cookie_post_with_csrf_header_succeeds(cookie_client):
    csrf = cookie_client.cookies.get("csrftoken")
    assert csrf
    res = await cookie_client.post("/api/auth/apikey", headers={"X-CSRF-Token": csrf})
    assert res.status_code == 200, res.text
    assert "api_key" in res.json()


@pytest.mark.asyncio
async def test_bearer_post_does_not_require_csrf(auth_client):
    """Bearer-token requests bypass CSRF — they cannot be triggered cross-origin."""
    res = await auth_client.post("/api/auth/apikey")
    assert res.status_code == 200, res.text


# ---------------------------------------------------------------------------
# 5. Auth rate limiting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_is_rate_limited(client):
    # 5/minute → 6th request from the same IP must be rejected with 429.
    last = None
    for _ in range(7):
        last = await client.post("/api/auth/login", json={
            "username": "no-such-user",
            "password": "wrong",
        })
    assert last is not None
    assert last.status_code == 429


@pytest.mark.asyncio
async def test_register_is_rate_limited(client):
    # Same 5/min throttle on /api/auth/register.
    last = None
    for i in range(7):
        last = await client.post("/api/auth/register", json={
            "username": f"throttle_test_user_{i}",
            "password": "password123",
        })
    assert last is not None
    assert last.status_code == 429


# ---------------------------------------------------------------------------
# 6. CORS pinning to DEPLOYED_DOMAIN; never "*" with credentials
# ---------------------------------------------------------------------------

def _build_app_with_env(monkeypatch, **env):
    """Force-reload the app factory after applying env overrides."""
    import importlib
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import idpkit.api.app as app_mod
    importlib.reload(app_mod)
    return app_mod.create_app()


def _cors_middleware_kwargs(app):
    from fastapi.middleware.cors import CORSMiddleware
    for mw in app.user_middleware:
        if mw.cls is CORSMiddleware:
            return mw.kwargs
    raise AssertionError("CORSMiddleware not mounted")


def test_cors_pins_to_deployed_domain(monkeypatch):
    app = _build_app_with_env(
        monkeypatch,
        DEPLOYED_DOMAIN="idpkit.example.com",
        CORS_EXTRA_ORIGINS=None,
        ALLOWED_ORIGINS=None,
    )
    cfg = _cors_middleware_kwargs(app)
    assert cfg["allow_origins"] == ["https://idpkit.example.com"]
    assert cfg["allow_credentials"] is True
    # The wildcard must never be combined with allow_credentials=True.
    assert "*" not in cfg["allow_origins"]


def test_cors_fails_closed_in_production_without_origins(monkeypatch):
    app = _build_app_with_env(
        monkeypatch,
        ENVIRONMENT="production",
        DEPLOYED_DOMAIN=None,
        CORS_EXTRA_ORIGINS=None,
        ALLOWED_ORIGINS=None,
    )
    cfg = _cors_middleware_kwargs(app)
    assert cfg["allow_origins"] == []
    assert cfg["allow_credentials"] is False


@pytest.mark.asyncio
async def test_cors_preflight_from_untrusted_origin_is_rejected(monkeypatch):
    """An OPTIONS preflight from an origin not on the allowlist must NOT
    receive an Access-Control-Allow-Origin header — the browser will then
    block the actual request."""
    import httpx
    app = _build_app_with_env(
        monkeypatch,
        DEPLOYED_DOMAIN="idpkit.example.com",
        CORS_EXTRA_ORIGINS=None,
        ALLOWED_ORIGINS=None,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="https://testserver") as c:
        res = await c.options(
            "/api/auth/login",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    # Starlette's CORSMiddleware returns 400 for a disallowed preflight and,
    # critically, does not echo the Access-Control-Allow-Origin header.
    assert res.headers.get("access-control-allow-origin") != "https://evil.example"
    assert "*" not in (res.headers.get("access-control-allow-origin") or "")


def test_cors_extra_origins_appended(monkeypatch):
    app = _build_app_with_env(
        monkeypatch,
        DEPLOYED_DOMAIN="idpkit.example.com",
        CORS_EXTRA_ORIGINS="https://app.partner.com,staging.example.com",
        ALLOWED_ORIGINS=None,
    )
    cfg = _cors_middleware_kwargs(app)
    assert cfg["allow_origins"] == [
        "https://idpkit.example.com",
        "https://app.partner.com",
        "https://staging.example.com",
    ]


# ---------------------------------------------------------------------------
# 7. LLM module never logs API key fragments
# ---------------------------------------------------------------------------

def test_llm_does_not_log_api_key_fragments(monkeypatch, caplog):
    """The debug log line must not include any prefix/suffix of the resolved
    key — only an opaque length is acceptable."""
    from idpkit.core import llm as llm_mod

    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789zzzz"
    monkeypatch.setattr(llm_mod, "_resolve_api_key_for_model", lambda m: secret)

    client = llm_mod.LLMClient()
    client.api_key = None  # force the env-resolution branch
    with caplog.at_level("DEBUG", logger="idpkit.core.llm"):
        kwargs = client._get_completion_kwargs("hello", model="gpt-4o-mini")
    assert kwargs["api_key"] == secret  # still wired through to litellm

    joined = "\n".join(r.getMessage() for r in caplog.records)
    # No fragment of the secret may appear in any log line.
    assert secret not in joined
    assert secret[:4] not in joined
    assert secret[-4:] not in joined


# ---------------------------------------------------------------------------
# 8. CSRF: API-key auth bypasses CSRF (header-based, not cross-origin-triggerable)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apikey_auth_bypasses_csrf(auth_client):
    """A POST authenticated via X-API-Key header must succeed without a
    CSRF token, even when a session_token cookie is also present."""
    # First, mint an API key for the admin user via the bearer-authed client.
    res = await auth_client.post("/api/auth/apikey")
    assert res.status_code == 200, res.text
    api_key = res.json()["api_key"]

    # Now build a fresh client that authenticates ONLY via X-API-Key, with
    # no session cookie and no Authorization header.
    import httpx
    from idpkit.api.app import create_app
    transport = httpx.ASGITransport(app=auth_client._transport.app)
    async with httpx.AsyncClient(transport=transport, base_url="https://testserver") as c:
        r = await c.post("/api/auth/apikey", headers={"X-API-Key": api_key})
        assert r.status_code == 200, r.text
