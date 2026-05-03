"""Tests for the expanded Skills importer (ZIP, URL, community catalog)."""
from __future__ import annotations

import io
import zipfile

import pytest

from idpkit.agent import skill_import as si


_VALID_MD = b"""---
name: test-skill
description: A skill used in tests.
---

# Test Skill

Body content.
"""


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, blob in files.items():
            zf.writestr(path, blob)
    return buf.getvalue()


# --- Pure unit tests for the import helpers ---------------------------------

def test_safe_extract_rejects_path_traversal():
    bad = _make_zip({"../evil.md": b"x"})
    with pytest.raises(si.SkillImportError):
        si.safe_extract_zip(bad)


def test_safe_extract_rejects_absolute_paths():
    bad = _make_zip({"/etc/passwd": b"x"})
    with pytest.raises(si.SkillImportError):
        si.safe_extract_zip(bad)


def test_safe_extract_rejects_too_many_files():
    big = _make_zip({f"f{i}.txt": b"a" for i in range(si.MAX_FILES + 1)})
    with pytest.raises(si.SkillImportError):
        si.safe_extract_zip(big)


def test_safe_extract_rejects_bad_zip():
    with pytest.raises(si.SkillImportError):
        si.safe_extract_zip(b"not a zip")


def test_safe_extract_strips_common_prefix():
    z = _make_zip({"my-skill/SKILL.md": _VALID_MD, "my-skill/scripts/run.py": b"print(1)"})
    files = si.safe_extract_zip(z)
    assert "SKILL.md" in files
    assert "scripts/run.py" in files


def test_parse_files_requires_skill_md():
    with pytest.raises(si.SkillImportError):
        si.parse_files_to_skill({"README.md": b"hello"})


def test_parse_files_requires_name_in_frontmatter():
    bad = b"---\ndescription: no name here\n---\nbody"
    with pytest.raises(si.SkillImportError):
        si.parse_files_to_skill({"SKILL.md": bad})


def test_parse_files_bundles_scripts_and_resources():
    parsed = si.parse_files_to_skill({
        "SKILL.md": _VALID_MD,
        "scripts/run.py": b"print('hi')",
        "resources/notes.md": b"helpful notes",
        "binary.bin": b"\x00\x01\x02",
    })
    assert parsed.name == "test-skill"
    paths = {s["path"] for s in parsed.scripts}
    assert "scripts/run.py" in paths
    assert "resources/notes.md" in paths
    kinds = {s["path"]: s["kind"] for s in parsed.scripts}
    assert kinds["scripts/run.py"] == "script"
    assert kinds["resources/notes.md"] == "resource"
    assert any("binary.bin" in w for w in parsed.warnings)


def test_parse_files_skips_oversized_resource():
    big_text = b"x" * (si.MAX_RESOURCE_BYTES + 1)
    parsed = si.parse_files_to_skill({"SKILL.md": _VALID_MD, "huge.md": big_text})
    assert all(s["path"] != "huge.md" for s in parsed.scripts)
    assert any("huge.md" in w for w in parsed.warnings)


def test_validate_url_requires_https():
    with pytest.raises(si.SkillImportError):
        si.validate_url("http://example.com/x")


def test_validate_url_blocks_localhost():
    with pytest.raises(si.SkillImportError):
        si.validate_url("https://localhost/x")
    with pytest.raises(si.SkillImportError):
        si.validate_url("https://127.0.0.1/x")


def test_normalize_github_blob_to_raw():
    out = si.normalize_github_url(
        "https://github.com/owner/repo/blob/main/SKILL.md"
    )
    assert out == "https://raw.githubusercontent.com/owner/repo/main/SKILL.md"


def test_import_from_md_bytes_round_trip():
    parsed = si.import_from_md_bytes(_VALID_MD)
    assert parsed.name == "test-skill"
    assert parsed.description.startswith("A skill")


def test_import_from_zip_bytes_round_trip():
    z = _make_zip({"my-skill/SKILL.md": _VALID_MD, "my-skill/run.py": b"print(1)"})
    parsed = si.import_from_zip_bytes(z)
    assert parsed.name == "test-skill"
    assert any(s["path"] == "run.py" for s in parsed.scripts)


# --- End-to-end via the API -------------------------------------------------

@pytest.mark.asyncio
async def test_api_import_md_file(auth_client):
    res = await auth_client.post(
        "/api/skills/import",
        files={"file": ("SKILL.md", _VALID_MD, "text/markdown")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["name"] == "test-skill"
    assert body["scripts_count"] == 0


@pytest.mark.asyncio
async def test_api_import_zip_file(auth_client):
    z = _make_zip({
        "kit/SKILL.md": b"---\nname: zipped-skill\ndescription: zipped\n---\n\nbody",
        "kit/scripts/x.py": b"print(1)",
    })
    res = await auth_client.post(
        "/api/skills/import",
        files={"file": ("skill.zip", z, "application/zip")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["name"] == "zipped-skill"
    assert body["scripts_count"] == 1


@pytest.mark.asyncio
async def test_api_import_preview_does_not_persist(auth_client):
    payload = b"---\nname: preview-only\ndescription: nope\n---\n\nbody"
    res = await auth_client.post(
        "/api/skills/import",
        files={"file": ("SKILL.md", payload, "text/markdown")},
        data={"preview": "true"},
    )
    assert res.status_code == 200
    assert "preview" in res.json()
    listing = await auth_client.get("/api/skills")
    assert all(s["name"] != "preview-only" for s in listing.json())


@pytest.mark.asyncio
async def test_api_import_url_rejects_http(auth_client):
    res = await auth_client.post(
        "/api/skills/import",
        data={"url": "http://example.com/SKILL.md"},
    )
    assert res.status_code == 400
    assert "https" in res.json()["detail"].lower()


@pytest.mark.asyncio
async def test_api_import_url_blocks_loopback(auth_client):
    res = await auth_client.post(
        "/api/skills/import",
        data={"url": "https://127.0.0.1/SKILL.md"},
    )
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_api_import_overwrite_flag(auth_client, monkeypatch):
    md1 = b"---\nname: dup-skill\ndescription: v1\n---\n\nv1"
    md2 = b"---\nname: dup-skill\ndescription: v2\n---\n\nv2"
    r1 = await auth_client.post(
        "/api/skills/import",
        files={"file": ("SKILL.md", md1, "text/markdown")},
    )
    assert r1.status_code == 200
    r2 = await auth_client.post(
        "/api/skills/import",
        files={"file": ("SKILL.md", md2, "text/markdown")},
    )
    assert r2.status_code == 409
    r3 = await auth_client.post(
        "/api/skills/import",
        files={"file": ("SKILL.md", md2, "text/markdown")},
        data={"overwrite": "true"},
    )
    assert r3.status_code == 200
    assert r3.json()["description"] == "v2"


@pytest.mark.asyncio
async def test_api_community_catalog_uses_cache(auth_client, monkeypatch):
    sample = [{"id": "x", "name": "Sample", "description": "d", "url": "https://example.com/x.md"}]
    si._CATALOG_CACHE.update({"ts": 9_999_999_999, "data": sample})
    res = await auth_client.get("/api/skills/community")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 1
    assert body["items"][0]["name"] == "Sample"


@pytest.mark.asyncio
async def test_api_community_install_uses_url_resolver(auth_client, monkeypatch):
    async def _fake(url: str):
        return si.parse_files_to_skill({"SKILL.md": b"---\nname: from-catalog\ndescription: c\n---\n\nbody"})
    monkeypatch.setattr("idpkit.api.routes.skills.import_from_url", _fake)
    res = await auth_client.post(
        "/api/skills/community/install",
        json={"url": "https://example.com/skill.md"},
    )
    assert res.status_code == 200
    assert res.json()["name"] == "from-catalog"
