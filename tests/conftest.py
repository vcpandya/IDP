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


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    p = Path("tests/pdfs/2023-annual-report-truncated.pdf")
    if not p.exists():
        # Fallback: any pdf in tests/pdfs/
        candidates = list(Path("tests/pdfs").glob("*.pdf"))
        assert candidates, "No sample PDF available for tests"
        p = candidates[0]
    return p.read_bytes()
