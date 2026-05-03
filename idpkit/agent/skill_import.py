"""Skill import utilities — safe ZIP extraction, URL fetching with SSRF defense,
GitHub URL normalization, and agentskills.io community-catalog browsing.

Used by /api/skills/import and /api/skills/community endpoints. Designed to share
one validator with the existing single-.md flow (no schema change to Skill).
"""
from __future__ import annotations

import io
import ipaddress
import json
import logging
import os
import socket
import time
import zipfile
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
import yaml

logger = logging.getLogger(__name__)

# Hard limits — shared with the single-.md flow.
MAX_TOTAL_SIZE = 5 * 1024 * 1024          # 5 MB cap on uploads / fetched bytes / uncompressed ZIP
MAX_FILES = 50                            # Max files inside a ZIP
MAX_RESOURCE_BYTES = 64 * 1024            # Per-resource cap; oversize → dropped with warning
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
USER_AGENT = "IDPKit-SkillImporter/1.0"
MAX_REDIRECTS = 5

SCRIPT_EXTS = {".py", ".js", ".ts", ".sh", ".rb", ".pl", ".bash"}
RESOURCE_EXTS = {".md", ".txt", ".json", ".yaml", ".yml", ".html", ".css", ".csv"}

DEFAULT_CATALOG_URL = os.environ.get(
    "AGENTSKILLS_CATALOG_URL",
    "https://agentskills.io/api/skills.json",
)
_CATALOG_CACHE: dict = {"ts": 0.0, "data": None}
_CATALOG_TTL_SECONDS = 300


class SkillImportError(ValueError):
    """Raised for any user-facing import validation failure."""


# ---------------------------------------------------------------------------
# SSRF-safe URL handling
# ---------------------------------------------------------------------------

def _resolve_host_safe(host: str) -> bool:
    """Resolve host to all IPs and reject if any are private/loopback/etc.

    Returns True only when EVERY resolved address is a public, routable IP.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


def validate_url(url: str) -> str:
    """Validate scheme + host. Raises SkillImportError on failure."""
    if not url or not isinstance(url, str):
        raise SkillImportError("URL is required")
    parsed = urlparse(url.strip())
    if parsed.scheme != "https":
        raise SkillImportError("Only https:// URLs are allowed")
    if not parsed.hostname:
        raise SkillImportError("Invalid URL: missing host")
    if not _resolve_host_safe(parsed.hostname):
        raise SkillImportError(
            "URL host is unreachable or resolves to a private/loopback address (blocked for SSRF protection)"
        )
    return url.strip()


def normalize_github_url(url: str) -> str:
    """Convert github.com 'blob' URLs to raw.githubusercontent.com so we can fetch the file directly."""
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return url
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 4 and parts[2] in ("blob", "raw"):
        owner, repo, _, branch, *rest = parts
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{'/'.join(rest)}"
    return url


async def fetch_bytes(url: str, max_size: int = MAX_TOTAL_SIZE) -> tuple[bytes, str]:
    """Fetch with manual redirect handling so each hop is SSRF-validated."""
    current = validate_url(url)
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            resp = await client.get(current)
            if resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("location")
                if not loc:
                    raise SkillImportError("Redirect response missing Location header")
                current = loc if loc.startswith("https://") else urljoin(current, loc)
                current = validate_url(current)
                continue
            if resp.status_code >= 400:
                raise SkillImportError(f"HTTP {resp.status_code} fetching {current}")
            data = resp.content
            if len(data) > max_size:
                raise SkillImportError(f"Response too large: {len(data)} bytes (max {max_size})")
            return data, resp.headers.get("content-type", "")
    raise SkillImportError("Too many redirects")


# ---------------------------------------------------------------------------
# Safe ZIP extraction
# ---------------------------------------------------------------------------

def safe_extract_zip(data: bytes) -> dict[str, bytes]:
    """Extract a ZIP into an in-memory dict of {normalized_path: bytes}.

    Rejects: path traversal, absolute paths, symlinks, >50 files, >5MB uncompressed.
    Strips a single common top-level directory if present.
    """
    if len(data) > MAX_TOTAL_SIZE:
        raise SkillImportError(f"ZIP file too large (>{MAX_TOTAL_SIZE} bytes)")
    try:
        zf_ctx = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise SkillImportError(f"Invalid ZIP archive: {exc}")

    files: dict[str, bytes] = {}
    with zf_ctx as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > MAX_FILES:
            raise SkillImportError(f"ZIP contains too many files ({len(members)} > {MAX_FILES})")
        total = 0
        for info in members:
            name = info.filename
            if name.startswith("/") or name.startswith("\\") or ".." in name.replace("\\", "/").split("/"):
                raise SkillImportError(f"Unsafe path in ZIP: {name}")
            if info.create_system == 3:
                mode = info.external_attr >> 16
                if mode and (mode & 0o170000) == 0o120000:
                    raise SkillImportError(f"Symlinks not allowed in ZIP: {name}")
            if info.file_size > MAX_TOTAL_SIZE:
                raise SkillImportError(f"File too large in ZIP: {name}")
            total += info.file_size
            if total > MAX_TOTAL_SIZE:
                raise SkillImportError(f"ZIP contents exceed {MAX_TOTAL_SIZE} bytes uncompressed")
            files[name.replace("\\", "/")] = zf.read(info)

    # Strip common single top-level dir
    if files and all("/" in n for n in files):
        prefixes = {n.split("/", 1)[0] for n in files}
        if len(prefixes) == 1:
            prefix = next(iter(prefixes)) + "/"
            files = {n[len(prefix):]: c for n, c in files.items() if n[len(prefix):]}

    return files


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedSkill:
    name: str
    description: str
    skill_content: str
    scripts: list[dict] = field(default_factory=list)  # [{path, content, kind, language?}]
    warnings: list[str] = field(default_factory=list)

    def to_preview_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "skill_content": self.skill_content,
            "scripts": [
                {"path": s["path"], "kind": s.get("kind"), "size": len(s.get("content", ""))}
                for s in self.scripts
            ],
            "warnings": self.warnings,
        }


def _parse_frontmatter(content: str) -> dict:
    text = content.strip()
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}


def parse_files_to_skill(files: dict[str, bytes]) -> ParsedSkill:
    """Locate SKILL.md, parse frontmatter, bundle adjacent scripts/resources."""
    skill_md_path: Optional[str] = None
    # Prefer top-level SKILL.md, then any nested one
    candidates = [n for n in files if n.lower().endswith("skill.md")]
    candidates.sort(key=lambda n: (n.count("/"), len(n)))
    if candidates:
        skill_md_path = candidates[0]
    if not skill_md_path:
        raise SkillImportError("No SKILL.md found in upload")

    try:
        skill_md_text = files[skill_md_path].decode("utf-8")
    except UnicodeDecodeError:
        raise SkillImportError("SKILL.md is not valid UTF-8")

    fm = _parse_frontmatter(skill_md_text)
    name = (fm.get("name") or "").strip()
    if not name:
        raise SkillImportError("SKILL.md frontmatter is missing required 'name' field")
    description = (fm.get("description") or "").strip()

    base = skill_md_path.rsplit("/", 1)[0] + "/" if "/" in skill_md_path else ""
    scripts: list[dict] = []
    warnings: list[str] = []

    for path, blob in files.items():
        if path == skill_md_path:
            continue
        if base and not path.startswith(base):
            continue
        rel = path[len(base):] if base else path
        if not rel or rel.endswith("/"):
            continue
        ext = os.path.splitext(rel)[1].lower()
        if ext in SCRIPT_EXTS:
            try:
                text = blob.decode("utf-8")
            except UnicodeDecodeError:
                warnings.append(f"Skipped non-UTF-8 script: {rel}")
                continue
            scripts.append({
                "path": rel,
                "content": text,
                "kind": "script",
                "language": ext.lstrip("."),
            })
        elif ext in RESOURCE_EXTS:
            if len(blob) > MAX_RESOURCE_BYTES:
                warnings.append(f"Skipped oversized resource (>{MAX_RESOURCE_BYTES} bytes): {rel}")
                continue
            try:
                text = blob.decode("utf-8")
            except UnicodeDecodeError:
                warnings.append(f"Skipped non-UTF-8 resource: {rel}")
                continue
            scripts.append({"path": rel, "content": text, "kind": "resource"})
        else:
            warnings.append(f"Skipped unsupported file type: {rel}")

    return ParsedSkill(
        name=name,
        description=description,
        skill_content=skill_md_text,
        scripts=scripts,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

async def import_from_url(url: str) -> ParsedSkill:
    """Fetch a public URL — supports raw .md, .zip, and github.com/blob/... links."""
    url = normalize_github_url(url.strip())
    data, ctype = await fetch_bytes(url)
    path_lower = urlparse(url).path.lower()
    is_zip = path_lower.endswith(".zip") or "zip" in (ctype or "").lower()
    if is_zip:
        files = safe_extract_zip(data)
        return parse_files_to_skill(files)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SkillImportError("Fetched content is not valid UTF-8")
    return parse_files_to_skill({"SKILL.md": text.encode("utf-8")})


def import_from_zip_bytes(data: bytes) -> ParsedSkill:
    files = safe_extract_zip(data)
    return parse_files_to_skill(files)


def import_from_md_bytes(data: bytes) -> ParsedSkill:
    if len(data) > MAX_TOTAL_SIZE:
        raise SkillImportError(f"File too large (>{MAX_TOTAL_SIZE} bytes)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SkillImportError("Uploaded file is not valid UTF-8")
    return parse_files_to_skill({"SKILL.md": text.encode("utf-8")})


# ---------------------------------------------------------------------------
# Community catalog (agentskills.io)
# ---------------------------------------------------------------------------

async def fetch_community_catalog(force: bool = False) -> list[dict]:
    """Fetch and cache the agentskills.io catalog (5 min TTL).

    Returns a list of {id, name, description, url, category?, author?} dicts.
    On network error returns the cached copy or [] (never raises).
    """
    now = time.time()
    if not force and _CATALOG_CACHE["data"] is not None and now - _CATALOG_CACHE["ts"] < _CATALOG_TTL_SECONDS:
        return _CATALOG_CACHE["data"]
    try:
        data, _ = await fetch_bytes(DEFAULT_CATALOG_URL, max_size=2 * 1024 * 1024)
        parsed = json.loads(data.decode("utf-8"))
        if isinstance(parsed, dict):
            parsed = parsed.get("skills") or parsed.get("items") or []
        if not isinstance(parsed, list):
            parsed = []
        # Normalize entry shape and require a usable URL
        normalized = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            url = entry.get("url") or entry.get("download_url") or entry.get("source_url")
            if not url:
                continue
            normalized.append({
                "id": entry.get("id") or entry.get("slug") or entry.get("name") or url,
                "name": entry.get("name") or entry.get("title") or "Untitled skill",
                "description": entry.get("description") or "",
                "category": entry.get("category"),
                "author": entry.get("author"),
                "url": url,
            })
        _CATALOG_CACHE.update({"ts": now, "data": normalized})
        return normalized
    except Exception as exc:  # noqa: BLE001
        logger.warning("Community catalog fetch failed: %s", exc)
        return _CATALOG_CACHE.get("data") or []
