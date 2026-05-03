"""Skill import utilities — safe ZIP extraction (temp-dir based with guaranteed
cleanup), streaming SSRF-defended URL fetcher with hard size cap, GitHub
blob/tree/repo-folder + agentskills.io URL resolvers, and a cached community
catalog browser.

Used by /api/skills/import and /api/skills/community endpoints. Shares one
validator with the existing single-.md flow (no schema change to Skill).
"""
from __future__ import annotations

import contextlib
import io
import ipaddress
import json
import logging
import os
import re
import socket
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
import yaml

logger = logging.getLogger(__name__)

# Hard limits — shared across single-.md, ZIP, and URL paths.
MAX_TOTAL_SIZE = 5 * 1024 * 1024          # 5 MB cap on uploads / fetched bytes / uncompressed ZIP
MAX_FILES = 50                            # Max files inside a ZIP / fetched folder
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
# SSRF-safe URL handling (per-hop validation, streaming download cap)
# ---------------------------------------------------------------------------

def _resolve_host_safe(host: str) -> bool:
    """All resolved IPs must be public, routable. Single private/loopback IP fails the check."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
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


async def fetch_bytes(
    url: str,
    max_size: int = MAX_TOTAL_SIZE,
    extra_headers: Optional[dict] = None,
) -> tuple[bytes, str]:
    """Stream-download with per-chunk size cap and per-hop SSRF revalidation.

    Aborts mid-stream as soon as the running byte count exceeds `max_size` so a
    malicious server cannot force us to buffer past the cap.
    """
    current = validate_url(url)
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if extra_headers:
        headers.update(extra_headers)
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        follow_redirects=False,
        headers=headers,
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            async with client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location")
                    if not loc:
                        raise SkillImportError("Redirect response missing Location header")
                    current = loc if loc.startswith("https://") else urljoin(current, loc)
                    current = validate_url(current)
                    continue
                if resp.status_code >= 400:
                    raise SkillImportError(f"HTTP {resp.status_code} fetching {current}")
                # Server-declared length cap (cheap rejection before reading)
                cl = resp.headers.get("content-length")
                if cl and cl.isdigit() and int(cl) > max_size:
                    raise SkillImportError(f"Response too large: declared {cl} bytes (max {max_size})")
                buf = bytearray()
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    buf.extend(chunk)
                    if len(buf) > max_size:
                        raise SkillImportError(f"Response exceeded {max_size} bytes during download")
                return bytes(buf), resp.headers.get("content-type", "")
    raise SkillImportError("Too many redirects")


# ---------------------------------------------------------------------------
# GitHub + agentskills.io URL resolvers
# ---------------------------------------------------------------------------

def normalize_github_url(url: str) -> str:
    """Convert github.com 'blob' URLs to raw.githubusercontent.com (folders untouched)."""
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return url
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 5 and parts[2] in ("blob", "raw"):
        owner, repo, _, branch, *rest = parts
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{'/'.join(rest)}"
    return url


def _parse_github_tree(url: str) -> Optional[tuple[str, str, str, str]]:
    """Return (owner, repo, branch, path) for a github.com tree URL, else None.

    Also handles repo-root URLs (`/owner/repo`) treated as tree of default branch.
    """
    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 4 and parts[2] == "tree":
        owner, repo, _, branch, *rest = parts
        return owner, repo, branch, "/".join(rest)
    if len(parts) == 2:
        return parts[0], parts[1], "HEAD", ""
    return None


async def _fetch_github_tree(owner: str, repo: str, branch: str, path: str) -> dict[str, bytes]:
    """Walk a GitHub folder via the Contents API. Returns {relative_path: bytes}.

    Adds Authorization header if GITHUB_TOKEN is set (raises rate limits).
    Caps at MAX_FILES total / MAX_TOTAL_SIZE bytes.
    """
    api_base = f"https://api.github.com/repos/{owner}/{repo}/contents"
    ref_qs = f"?ref={branch}" if branch and branch != "HEAD" else ""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    files: dict[str, bytes] = {}
    total_bytes = 0
    queue = [path.strip("/")]

    while queue:
        cur = queue.pop(0)
        list_url = f"{api_base}/{cur}{ref_qs}" if cur else f"{api_base}{ref_qs}"
        data, _ = await fetch_bytes(list_url, max_size=2 * 1024 * 1024, extra_headers=headers)
        try:
            entries = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SkillImportError(f"GitHub API returned invalid JSON: {exc}")
        if isinstance(entries, dict) and entries.get("type") == "file":
            entries = [entries]
        if not isinstance(entries, list):
            raise SkillImportError("GitHub path is not a folder")

        for entry in entries:
            etype = entry.get("type")
            ename = entry.get("name") or ""
            epath = entry.get("path") or ""
            if etype == "dir":
                queue.append(epath)
                continue
            if etype != "file":
                continue
            size = entry.get("size") or 0
            if total_bytes + size > MAX_TOTAL_SIZE:
                raise SkillImportError(f"Folder exceeds {MAX_TOTAL_SIZE} bytes")
            if len(files) >= MAX_FILES:
                raise SkillImportError(f"Folder exceeds {MAX_FILES} files")
            dl = entry.get("download_url")
            if not dl:
                continue
            blob, _ = await fetch_bytes(dl)
            total_bytes += len(blob)
            if total_bytes > MAX_TOTAL_SIZE:
                raise SkillImportError(f"Folder exceeds {MAX_TOTAL_SIZE} bytes after download")
            # Make path relative to the requested folder root
            rel = epath
            if path and rel.startswith(path.rstrip("/") + "/"):
                rel = rel[len(path.rstrip("/")) + 1:]
            files[rel.replace("\\", "/")] = blob

    if not files:
        raise SkillImportError("GitHub folder is empty or unreadable")
    return files


_AGENTSKILLS_HOSTS = {"agentskills.io", "www.agentskills.io"}


async def _resolve_agentskills_page(url: str) -> str:
    """Resolve an agentskills.io skill-page URL to a downloadable artifact URL.

    Strategy: try `{url}.json` first; fall back to scraping a 'download_url'
    field or the first <a href> ending in SKILL.md / .zip in the HTML.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host not in _AGENTSKILLS_HOSTS:
        return url
    # Try JSON variant
    try:
        json_url = url.rstrip("/") + ".json"
        data, ctype = await fetch_bytes(json_url, max_size=512 * 1024)
        meta = json.loads(data.decode("utf-8"))
        for key in ("download_url", "raw_url", "source_url", "url"):
            v = meta.get(key) if isinstance(meta, dict) else None
            if isinstance(v, str) and v.startswith("https://"):
                return v
    except Exception as exc:  # noqa: BLE001
        logger.debug("agentskills .json variant failed for %s: %s", url, exc)
    # Fallback: fetch HTML and scan for a link
    try:
        data, _ = await fetch_bytes(url, max_size=1 * 1024 * 1024)
        html = data.decode("utf-8", errors="replace")
        m = re.search(
            r'href=["\'](https://[^"\']+\.(?:zip|md))["\']',
            html,
            re.IGNORECASE,
        )
        if m:
            return m.group(1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("agentskills HTML scrape failed for %s: %s", url, exc)
    raise SkillImportError("Could not resolve agentskills.io page to a downloadable skill")


# ---------------------------------------------------------------------------
# Safe ZIP extraction — extracts into a temp dir with guaranteed cleanup
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _temp_workdir():
    """Yield a tempfile.TemporaryDirectory Path; always cleaned up on exit."""
    td = tempfile.mkdtemp(prefix="skill_import_")
    try:
        yield Path(td)
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def _validate_zip_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
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
    return members


def safe_extract_zip(data: bytes) -> dict[str, bytes]:
    """Extract a ZIP into a temp dir, then read files back. Cleans up on every path.

    Rejects: path traversal, absolute paths, symlinks, >50 files, >5MB total.
    Strips a single common top-level directory if present.
    """
    if len(data) > MAX_TOTAL_SIZE:
        raise SkillImportError(f"ZIP file too large (>{MAX_TOTAL_SIZE} bytes)")
    try:
        zf_open = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise SkillImportError(f"Invalid ZIP archive: {exc}")

    files: dict[str, bytes] = {}
    with zf_open as zf, _temp_workdir() as td:
        members = _validate_zip_members(zf)
        zf.extractall(td, members=members)
        # Read back from disk (paths already validated)
        for member in members:
            disk_path = td / member.filename
            if not disk_path.is_file():
                # Could have been a directory marker; skip safely
                continue
            files[member.filename.replace("\\", "/")] = disk_path.read_bytes()

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
    source: Optional[str] = None  # human-readable origin

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
            "source": self.source,
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


def parse_files_to_skill(files: dict[str, bytes], source: Optional[str] = None) -> ParsedSkill:
    """Locate SKILL.md, parse frontmatter, bundle adjacent scripts/resources."""
    candidates = [n for n in files if n.lower().endswith("skill.md")]
    candidates.sort(key=lambda n: (n.count("/"), len(n)))
    if not candidates:
        raise SkillImportError("No SKILL.md found in upload")
    skill_md_path = candidates[0]

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
        source=source,
    )


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

async def import_from_url(url: str) -> ParsedSkill:
    """Resolve any of: raw .md, .zip download, github blob/tree/repo, or agentskills.io page.

    All downloads run inside a temp workdir that is cleaned up before return.
    """
    raw = url.strip()
    parsed = urlparse(raw)
    host = parsed.hostname or ""

    # 1) agentskills.io page → resolve to downloadable artifact URL
    if host in _AGENTSKILLS_HOSTS and not raw.lower().endswith((".md", ".zip", ".json")):
        raw = await _resolve_agentskills_page(raw)
        parsed = urlparse(raw)
        host = parsed.hostname or ""

    # 2) GitHub tree URL or repo-root → walk Contents API
    tree = _parse_github_tree(raw)
    if tree is not None:
        owner, repo, branch, path = tree
        with _temp_workdir():  # ensures cleanup even if parsing fails
            files = await _fetch_github_tree(owner, repo, branch, path)
            return parse_files_to_skill(files, source=raw)

    # 3) GitHub blob URL → raw
    raw = normalize_github_url(raw)

    # 4) Direct download (md or zip)
    data, ctype = await fetch_bytes(raw)
    path_lower = urlparse(raw).path.lower()
    is_zip = path_lower.endswith(".zip") or "zip" in (ctype or "").lower()
    with _temp_workdir() as td:
        # Persist download to disk inside the temp workdir (cleaned up below)
        artifact = td / ("download.zip" if is_zip else "download.md")
        artifact.write_bytes(data)
        if is_zip:
            files = safe_extract_zip(artifact.read_bytes())
            return parse_files_to_skill(files, source=raw)
        try:
            text = artifact.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise SkillImportError("Fetched content is not valid UTF-8")
        return parse_files_to_skill({"SKILL.md": text.encode("utf-8")}, source=raw)


def import_from_zip_bytes(data: bytes, source: Optional[str] = None) -> ParsedSkill:
    files = safe_extract_zip(data)
    return parse_files_to_skill(files, source=source)


def import_from_md_bytes(data: bytes, source: Optional[str] = None) -> ParsedSkill:
    if len(data) > MAX_TOTAL_SIZE:
        raise SkillImportError(f"File too large (>{MAX_TOTAL_SIZE} bytes)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SkillImportError("Uploaded file is not valid UTF-8")
    return parse_files_to_skill({"SKILL.md": text.encode("utf-8")}, source=source)


# ---------------------------------------------------------------------------
# Community catalog (agentskills.io)
# ---------------------------------------------------------------------------

async def fetch_community_catalog(force: bool = False) -> list[dict]:
    """Fetch + cache the catalog (5 min TTL). Falls back to last-good on error."""
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


async def find_community_entry(skill_id: str) -> Optional[dict]:
    catalog = await fetch_community_catalog()
    for entry in catalog:
        if entry.get("id") == skill_id:
            return entry
    return None
