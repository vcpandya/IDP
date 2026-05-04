"""Shared pytest fixtures for IDP Kit E2E tests.

Spins up the FastAPI app against an isolated SQLite database and a tmp filesystem
storage backend, seeds the default admin, monkey-patches the e-sign email sender
to capture signing URLs in-memory, and exposes an authenticated httpx AsyncClient.
"""
from __future__ import annotations

import os
import tempfile
import shutil
from pathlib import Path

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def _tmp_workspace():
    root = Path(tempfile.mkdtemp(prefix="idpkit_test_"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _env(_tmp_workspace):
    db_path = _tmp_workspace / "test.db"
    storage_path = _tmp_workspace / "storage"
    storage_path.mkdir(exist_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    os.environ["SECRET_KEY"] = "test-secret-key-for-e2e-tests-only-not-for-prod"
    os.environ["IDP_STORAGE_PATH"] = str(storage_path)
    os.environ.pop("DEFAULT_OBJECT_STORAGE_BUCKET_ID", None)
    os.environ.pop("PRIVATE_OBJECT_DIR", None)
    os.environ["IDP_ADMIN_PASSWORD"] = "test-admin-pw"
    os.environ["ESIGN_EXPIRY_DAYS"] = "30"
    os.environ.setdefault("OBJECT_STORAGE_BUCKET", "")
    yield


@pytest.fixture
def captured_invitations(monkeypatch):
    """Patch send_signing_invitation to record (and not actually send) invitations."""
    captured: list[dict] = []

    async def _fake_send(**kwargs):
        captured.append(kwargs)
        return True

    from idpkit.esign import email as esign_email
    monkeypatch.setattr(esign_email, "send_signing_invitation", _fake_send)
    # Also patch the imported reference in the routes module
    from idpkit.api.routes import esign as esign_routes
    if hasattr(esign_routes, "send_signing_invitation"):
        monkeypatch.setattr(esign_routes, "send_signing_invitation", _fake_send)
    return captured


@pytest_asyncio.fixture
async def app(_env):
    from idpkit.api.app import create_app
    from idpkit.db.session import init_db, async_session
    from idpkit.db.seed import seed_default_admin

    application = create_app()
    await init_db()
    await seed_default_admin(async_session)
    yield application


@pytest_asyncio.fixture
async def client(app):
    import httpx
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest_asyncio.fixture
async def auth_client(client):
    """Logged-in admin client (Bearer token in Authorization header)."""
    res = await client.post("/api/auth/login", json={
        "username": "admin",
        "password": os.environ["IDP_ADMIN_PASSWORD"],
    })
    assert res.status_code == 200, res.text
    token = res.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client


# ---------------------------------------------------------------------------
# Live-integration sandbox credentials
# ---------------------------------------------------------------------------
#
# Each fixture below returns a credential dict for a single connector, sourced
# from environment variables. When any required variable is missing the fixture
# calls ``pytest.skip`` so the test is silently skipped on developer machines
# (and on CI runs where the secret is not configured), but exercised on the
# nightly job that does have the sandbox secrets.
#
# Naming convention for env vars: ``IDPKIT_LIVE_<CONNECTOR>_<FIELD>``. We use
# this prefix to make it obvious in CI configs that these are sandbox-only
# credentials for the live integration suite — never reuse production tokens.

def _require_env(*names: str) -> dict[str, str]:
    """Return a dict of env values, or skip the test if any are unset/empty."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"live test skipped — missing env: {', '.join(missing)}")
    return {n: os.environ[n] for n in names}


@pytest.fixture
def slack_live_creds() -> dict:
    env = _require_env("IDPKIT_LIVE_SLACK_BOT_TOKEN")
    return {"bot_token": env["IDPKIT_LIVE_SLACK_BOT_TOKEN"]}


@pytest.fixture
def notion_live_creds() -> dict:
    env = _require_env("IDPKIT_LIVE_NOTION_TOKEN")
    return {"integration_token": env["IDPKIT_LIVE_NOTION_TOKEN"]}


@pytest.fixture
def github_live_creds() -> dict:
    env = _require_env("IDPKIT_LIVE_GITHUB_TOKEN")
    return {"token": env["IDPKIT_LIVE_GITHUB_TOKEN"]}


@pytest.fixture
def linear_live_creds() -> dict:
    env = _require_env("IDPKIT_LIVE_LINEAR_API_KEY")
    return {"api_key": env["IDPKIT_LIVE_LINEAR_API_KEY"]}


@pytest.fixture
def hubspot_live_creds() -> dict:
    env = _require_env("IDPKIT_LIVE_HUBSPOT_TOKEN")
    return {"access_token": env["IDPKIT_LIVE_HUBSPOT_TOKEN"]}


@pytest.fixture
def dropbox_live_creds() -> dict:
    env = _require_env("IDPKIT_LIVE_DROPBOX_TOKEN")
    return {"access_token": env["IDPKIT_LIVE_DROPBOX_TOKEN"]}


@pytest.fixture
def jira_live_creds() -> dict:
    env = _require_env(
        "IDPKIT_LIVE_JIRA_SITE",
        "IDPKIT_LIVE_JIRA_EMAIL",
        "IDPKIT_LIVE_JIRA_API_TOKEN",
    )
    return {
        "site": env["IDPKIT_LIVE_JIRA_SITE"],
        "email": env["IDPKIT_LIVE_JIRA_EMAIL"],
        "api_token": env["IDPKIT_LIVE_JIRA_API_TOKEN"],
    }


@pytest.fixture
def s3_live_creds() -> dict:
    env = _require_env(
        "IDPKIT_LIVE_S3_ACCESS_KEY_ID",
        "IDPKIT_LIVE_S3_SECRET_ACCESS_KEY",
        "IDPKIT_LIVE_S3_BUCKET",
    )
    return {
        "access_key_id": env["IDPKIT_LIVE_S3_ACCESS_KEY_ID"],
        "secret_access_key": env["IDPKIT_LIVE_S3_SECRET_ACCESS_KEY"],
        "bucket": env["IDPKIT_LIVE_S3_BUCKET"],
        "region": os.environ.get("IDPKIT_LIVE_S3_REGION", "us-east-1"),
    }


@pytest.fixture
def google_live_creds() -> dict:
    # Google connector is OAuth2 — for live tests we accept a pre-minted access
    # token (e.g. produced by an offline ``oauth2l`` flow) plus optional refresh
    # token so the runtime refresh path can be exercised separately if desired.
    env = _require_env("IDPKIT_LIVE_GOOGLE_ACCESS_TOKEN")
    creds = {"access_token": env["IDPKIT_LIVE_GOOGLE_ACCESS_TOKEN"]}
    rt = os.environ.get("IDPKIT_LIVE_GOOGLE_REFRESH_TOKEN")
    if rt:
        creds["refresh_token"] = rt
    return creds


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    p = Path("tests/pdfs/2023-annual-report-truncated.pdf")
    if not p.exists():
        # Fallback: any pdf in tests/pdfs/
        candidates = list(Path("tests/pdfs").glob("*.pdf"))
        assert candidates, "No sample PDF available for tests"
        p = candidates[0]
    return p.read_bytes()
