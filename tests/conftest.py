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
    # Force dev mode for the test suite so production fail-closed paths
    # (CORS, SECRET_KEY) don't engage. Tests that need to exercise the
    # production paths set DEPLOYED_DOMAIN explicitly via monkeypatch.
    os.environ.pop("DEPLOYED_DOMAIN", None)
    os.environ.pop("ENVIRONMENT", None)
    yield


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the SlowAPI in-memory bucket between tests so the per-IP
    login/register rate limit (5/min) doesn't bleed across tests."""
    try:
        from idpkit.api.deps import limiter
        limiter.reset()
    except Exception:
        pass
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


# ---------------------------------------------------------------------------
# Sandbox targets for *mutating* live tests (Task #18)
# ---------------------------------------------------------------------------
#
# These extend the read-only ``*_live_creds`` fixtures above with the extra
# IDs / paths needed to safely create-and-clean a record in a dedicated
# sandbox project, channel, mailbox, etc. They follow the same naming
# convention (``IDPKIT_LIVE_<CONNECTOR>_<FIELD>``) and skip cleanly when the
# extra env vars are missing — so the read-only suite continues to run on
# nightly even if the sandbox-target secrets are not yet provisioned.

@pytest.fixture
def slack_sandbox(slack_live_creds) -> dict:
    env = _require_env("IDPKIT_LIVE_SLACK_CHANNEL")
    return {**slack_live_creds, "channel": env["IDPKIT_LIVE_SLACK_CHANNEL"]}


@pytest.fixture
def github_sandbox(github_live_creds) -> dict:
    env = _require_env("IDPKIT_LIVE_GITHUB_REPO")
    return {**github_live_creds, "repo": env["IDPKIT_LIVE_GITHUB_REPO"]}


@pytest.fixture
def linear_sandbox(linear_live_creds) -> dict:
    env = _require_env("IDPKIT_LIVE_LINEAR_TEAM_ID")
    return {**linear_live_creds, "team_id": env["IDPKIT_LIVE_LINEAR_TEAM_ID"]}


@pytest.fixture
def jira_sandbox(jira_live_creds) -> dict:
    env = _require_env("IDPKIT_LIVE_JIRA_PROJECT_KEY")
    return {**jira_live_creds, "project_key": env["IDPKIT_LIVE_JIRA_PROJECT_KEY"]}


@pytest.fixture
def notion_sandbox(notion_live_creds) -> dict:
    env = _require_env("IDPKIT_LIVE_NOTION_PARENT_PAGE_ID")
    return {**notion_live_creds, "parent_page_id": env["IDPKIT_LIVE_NOTION_PARENT_PAGE_ID"]}


@pytest.fixture
def hubspot_sandbox(hubspot_live_creds) -> dict:
    # HubSpot needs no extra target — we generate a unique sandbox email per run.
    return dict(hubspot_live_creds)


@pytest.fixture
def dropbox_sandbox(dropbox_live_creds) -> dict:
    env = _require_env("IDPKIT_LIVE_DROPBOX_PATH")
    return {**dropbox_live_creds, "path": env["IDPKIT_LIVE_DROPBOX_PATH"]}


@pytest.fixture
def google_sandbox(google_live_creds) -> dict:
    # Optional sandbox targets — individual tests _require_env on what they need.
    creds = dict(google_live_creds)
    for k in (
        "IDPKIT_LIVE_GMAIL_TO",
        "IDPKIT_LIVE_GOOGLE_SHEET_ID",
        "IDPKIT_LIVE_GOOGLE_SHEET_RANGE",
        "IDPKIT_LIVE_GOOGLE_CALENDAR_ID",
    ):
        v = os.environ.get(k)
        if v:
            creds[k.lower()] = v
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
