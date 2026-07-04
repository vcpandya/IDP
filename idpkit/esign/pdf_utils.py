"""PDF utilities for e-signature: render pages, overlay signatures, generate audit reports."""

import base64
import hashlib
import hmac
import io
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

CONSENT_TEXT = (
    "By clicking \u2018I Agree & Sign\u2019 you are signing this document electronically. "
    "You agree that your electronic signature is the legal equivalent of your manual "
    "signature on this document."
)


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_page_to_image(pdf_bytes: bytes, page_num: int, dpi: int = 120) -> str:
    """Render a PDF page to a base64-encoded PNG string."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_num < 1 or page_num > len(doc):
        doc.close()
        raise ValueError(f"Page {page_num} out of range (1-{len(doc)})")
    page = doc[page_num - 1]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode()


def get_page_count(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


def overlay_signatures(pdf_bytes: bytes, fields: list[dict]) -> bytes:
    """
    Overlay signed field values onto the PDF.
    Each field dict has: page, x_pct, y_pct, w_pct, h_pct, field_type, value.
    value is either a base64 PNG (for signature/initials) or plain text (date/text).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for field in fields:
        if not field.get("value"):
            continue
        page_num = field.get("page", 1)
        if page_num < 1 or page_num > len(doc):
            continue
        page = doc[page_num - 1]
        w = page.rect.width
        h = page.rect.height
        x0 = (field["x_pct"] / 100.0) * w
        y0 = (field["y_pct"] / 100.0) * h
        x1 = x0 + (field["w_pct"] / 100.0) * w
        y1 = y0 + (field["h_pct"] / 100.0) * h
        rect = fitz.Rect(x0, y0, x1, y1)

        ftype = field.get("field_type", "text")
        value = field["value"]

        if ftype in ("signature", "initials"):
            # Fail loud — if a signature image cannot be overlaid the envelope
            # MUST NOT be marked completed with a missing visual. The caller
            # (submit_signature) catches this and rolls back.
            try:
                img_data = _decode_image_value(value)
            except Exception as exc:
                logger.error("Signature value for field on page %d is not valid base64: %s", page_num, exc)
                raise ValueError(f"Signature image on page {page_num} is malformed") from exc
            try:
                page.insert_image(rect, stream=img_data, keep_proportion=True)
            except Exception as exc:
                logger.error("Failed to overlay signature image on page %d: %s", page_num, exc)
                raise ValueError(f"Failed to render signature on page {page_num}: {exc}") from exc
        else:
            # date or text
            font_size = max(8, min(14, int((y1 - y0) * 0.55)))
            page.draw_rect(rect, color=(0.9, 0.95, 1.0), fill=(0.9, 0.95, 1.0))
            page.insert_textbox(
                rect,
                value,
                fontsize=font_size,
                fontname="helv",
                color=(0.1, 0.1, 0.1),
                align=0,
            )

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


def _decode_image_value(value: str) -> bytes:
    """Strip data URL prefix and decode base64."""
    if "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def append_audit_certificate_page(
    pdf_bytes: bytes,
    envelope_id: str,
    title: str,
    doc_sha256: str,
    signers: list[dict],
    events: list[dict],
) -> bytes:
    """Append a one-page Audit Certificate summary to the signed PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    cert_page = doc.new_page(-1, width=595, height=842)  # A4

    margin = 40
    y = margin
    gray = (0.4, 0.4, 0.4)
    dark = (0.1, 0.1, 0.1)
    blue = (0.18, 0.33, 0.72)

    def write(text, x=margin, size=10, color=dark, bold=False):
        nonlocal y
        fname = "hebo" if bold else "helv"
        cert_page.insert_text((x, y), text, fontsize=size, fontname=fname, color=color)
        y += size + 4

    def line():
        nonlocal y
        cert_page.draw_line((margin, y), (595 - margin, y), color=(0.8, 0.8, 0.8), width=0.5)
        y += 6

    write("AUDIT CERTIFICATE", size=16, bold=True, color=blue)
    write("Electronic Signature Record", size=10, color=gray)
    y += 4
    line()

    write(f"Document: {title}", size=10, bold=True)
    write(f"Envelope ID: {envelope_id}", size=9, color=gray)
    write(f"Document SHA-256: {doc_sha256}", size=8, color=gray)
    write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", size=9, color=gray)
    y += 4
    line()

    write("SIGNERS", size=11, bold=True, color=blue)
    y += 2
    for s in signers:
        write(f"  {s.get('name', '')} <{s.get('email', '')}>", size=9, bold=True)
        signed_at = s.get("signed_at") or "—"
        ip = s.get("ip_address") or "—"
        write(f"    Status: {s.get('status', '')}   Signed at: {signed_at}   IP: {ip}", size=8, color=gray)
        y += 2
    y += 4
    line()

    write("SIGNATURE EVENTS", size=11, bold=True, color=blue)
    y += 2
    for ev in events[-30:]:
        ts = ev.get("created_at") or ""
        actor = ev.get("actor_email") or "system"
        etype = ev.get("event_type") or ""
        ip = ev.get("ip_address") or ""
        write(f"  {ts}  {actor}  [{etype}]  {ip}", size=7.5, color=gray)
        y += 1

    y += 6
    line()
    write("This certificate is part of the legally binding electronic signing record.", size=8, color=gray)
    write(CONSENT_TEXT[:120] + "...", size=7.5, color=gray)

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


def generate_audit_report_pdf(
    envelope_id: str,
    title: str,
    doc_sha256: str,
    signers: list[dict],
    events: list[dict],
    hmac_key: str,
    envelope_url: str = "",
) -> bytes:
    """Generate a standalone forensic Audit Report PDF."""
    doc = fitz.Document()
    _build_report_page(doc, envelope_id, title, doc_sha256, signers, events, hmac_key, envelope_url)
    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


def _build_report_page(doc, envelope_id, title, doc_sha256, signers, events, hmac_key, envelope_url):
    PAGE_W = 595
    PAGE_H = 842  # A4
    margin = 40
    gray = (0.45, 0.45, 0.45)
    dark = (0.05, 0.05, 0.05)
    blue = (0.18, 0.33, 0.72)

    # --- Pagination helpers ---
    pages = []
    current_page = [None]
    y_ref = [margin]

    def _new_page():
        p = doc.new_page(-1, width=PAGE_W, height=PAGE_H)
        pages.append(p)
        current_page[0] = p
        y_ref[0] = margin

    def _ensure_space(needed=20):
        if y_ref[0] + needed > PAGE_H - margin:
            _new_page()

    def write(text, x=margin, size=9, color=dark, bold=False):
        _ensure_space(size + 5)
        fname = "hebo" if bold else "helv"
        safe = str(text)[:300]
        current_page[0].insert_text((x, y_ref[0]), safe, fontsize=size, fontname=fname, color=color)
        y_ref[0] += size + 3

    def line(color=(0.8, 0.8, 0.8)):
        _ensure_space(8)
        current_page[0].draw_line((margin, y_ref[0]), (PAGE_W - margin, y_ref[0]), color=color, width=0.5)
        y_ref[0] += 5

    def section(label):
        _ensure_space(30)
        y_ref[0] += 6
        current_page[0].draw_rect(fitz.Rect(margin, y_ref[0], PAGE_W - margin, y_ref[0] + 18), color=blue, fill=blue)
        current_page[0].insert_text((margin + 4, y_ref[0] + 13), label, fontsize=10, fontname="hebo", color=(1, 1, 1))
        y_ref[0] += 22

    # Start first page
    _new_page()

    # Header
    write("FORENSIC AUDIT REPORT", size=18, bold=True, color=blue)
    write("Electronic Signature Provenance Record", size=10, color=gray)
    y_ref[0] += 4
    line(color=blue)

    # Summary
    write(f"Document Title:   {title}", size=9)
    write(f"Envelope ID:      {envelope_id}", size=9)
    write(f"Document SHA-256: {doc_sha256}", size=8, color=gray)
    write(f"Report Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", size=9)
    y_ref[0] += 4
    line()

    # Legal notice
    write(CONSENT_TEXT, size=8, color=gray)
    y_ref[0] += 4

    # QR code (placed on first page, top-right)
    if envelope_url:
        try:
            import qrcode
            qr = qrcode.QRCode(box_size=3, border=2)
            qr.add_data(envelope_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            qr_rect = fitz.Rect(PAGE_W - margin - 80, margin, PAGE_W - margin, margin + 80)
            pages[0].insert_image(qr_rect, stream=buf.getvalue())
            pages[0].insert_text(
                (PAGE_W - margin - 80, margin + 84),
                "Verify online",
                fontsize=6,
                color=gray,
            )
        except Exception as exc:
            logger.warning("QR code generation failed: %s", exc)

    # Signers section
    section("SIGNERS & IDENTITY")
    for s in signers:
        write(f"Name:   {s.get('name', '')}", bold=True)
        write(f"Email:  {s.get('email', '')}", size=8)
        write(f"Status: {s.get('status', '')}    Signed At: {s.get('signed_at') or '—'}", size=8)
        write(f"IP:     {s.get('ip_address') or '—'}", size=7.5, color=gray)
        ua_full = (s.get("user_agent") or "")[:200]
        if ua_full:
            write(f"UA:     {ua_full}", size=7, color=gray)
        y_ref[0] += 4

    # Detailed audit events — fully paginated, no truncation
    section("DETAILED AUDIT TRAIL")
    cols = ["Timestamp (UTC)", "Actor", "Event", "IP Address", "Browser", "OS", "Geo"]
    col_w = [110, 115, 80, 80, 75, 55, 55]
    col_x = [margin]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w)

    def _draw_col_headers():
        for i, col in enumerate(cols):
            current_page[0].insert_text((col_x[i], y_ref[0] + 10), col, fontsize=6.5, fontname="hebo", color=blue)
        y_ref[0] += 14
        current_page[0].draw_line((margin, y_ref[0]), (PAGE_W - margin, y_ref[0]), color=blue, width=0.4)
        y_ref[0] += 4

    _draw_col_headers()

    for ev in events:
        # Each event row needs ~28 px (main row + UA sub-row + extras + divider)
        _ensure_space(32)
        # If we started a new page, re-draw column headers
        if y_ref[0] == margin:
            _draw_col_headers()

        ts = str(ev.get("created_at") or "")[:19]
        actor = (ev.get("actor_email") or "system")[:18]
        etype = (ev.get("event_type") or "")[:14]
        ip = (ev.get("ip_address") or "")[:15]
        browser = (f"{ev.get('browser_name') or ''} {ev.get('browser_version') or ''}").strip()[:12]
        os_ = (ev.get("os_name") or "")[:10]
        geo = f"{ev.get('geo_country') or ''}/{ev.get('geo_city') or ''}".strip("/")[:10]

        row_data = [ts, actor, etype, ip, browser, os_, geo]
        for i, val in enumerate(row_data):
            current_page[0].insert_text((col_x[i], y_ref[0] + 9), val, fontsize=6.5, fontname="helv", color=dark)
        y_ref[0] += 12

        # Full User-Agent string sub-row (required forensic field)
        ua_full = (ev.get("user_agent") or "").strip()
        if ua_full:
            write(f"    UA: {ua_full[:180]}", size=6, color=gray)

        # Fingerprint / browser-environment sub-row
        extras = []
        if ev.get("canvas_fingerprint_hash"):
            extras.append(f"CanvasFP: {ev['canvas_fingerprint_hash'][:16]}…")
        if ev.get("screen_resolution"):
            extras.append(f"Screen: {ev['screen_resolution']}")
        if ev.get("timezone"):
            extras.append(f"TZ: {ev['timezone']}")
        if ev.get("language"):
            extras.append(f"Lang: {ev['language']}")
        if ev.get("session_id"):
            extras.append(f"Session: {ev['session_id'][:12]}…")
        if extras:
            write("    " + "  |  ".join(extras), size=6, color=gray)

        current_page[0].draw_line((margin, y_ref[0]), (PAGE_W - margin, y_ref[0]), color=(0.9, 0.9, 0.9), width=0.2)
        y_ref[0] += 2

    # Technical notes — on whatever page we end up on
    y_ref[0] += 8
    section("TECHNICAL NOTES")
    write("MAC Address: Not available — web browsers do not expose device MAC addresses for privacy reasons.", size=8, color=gray)
    write("Browser fingerprint is computed via HTML5 Canvas API (silent, no cookies required).", size=8, color=gray)
    write("IP geolocation is approximate and based on public IP address lookup.", size=8, color=gray)

    # HMAC signature — covers the full ordered audit event log
    y_ref[0] += 8
    section("TAMPER-EVIDENT SEAL")
    canonical_events = [
        {
            "ts": ev.get("created_at", ""),
            "actor": ev.get("actor_email", ""),
            "event": ev.get("event_type", ""),
            "ip": ev.get("ip_address", ""),
            "user_agent": ev.get("user_agent", ""),
            "browser": ev.get("browser_name", ""),
            "browser_ver": ev.get("browser_version", ""),
            "os": ev.get("os_name", ""),
            "geo_country": ev.get("geo_country", ""),
            "geo_city": ev.get("geo_city", ""),
            "canvas_fp": ev.get("canvas_fingerprint_hash", ""),
            "screen": ev.get("screen_resolution", ""),
            "tz": ev.get("timezone", ""),
            "lang": ev.get("language", ""),
            "session": ev.get("session_id", ""),
        }
        for ev in events
    ]
    audit_payload = json.dumps({
        "envelope_id": envelope_id,
        "doc_sha256": doc_sha256,
        "signers": [
            {
                "name": s.get("name", ""),
                "email": s.get("email", ""),
                "status": s.get("status", ""),
                "ip_address": s.get("ip_address", ""),
                "signed_at": str(s.get("signed_at", "")),
            }
            for s in signers
        ],
        "events": canonical_events,
    }, sort_keys=True, separators=(",", ":"))
    seal = hmac.new(hmac_key.encode(), audit_payload.encode("utf-8"), hashlib.sha256).hexdigest()
    payload_hash = hashlib.sha256(audit_payload.encode("utf-8")).hexdigest()
    write(f"HMAC-SHA256 Seal: {seal}", size=8, bold=True)
    write(f"Payload SHA-256:  {payload_hash}", size=7.5)
    write(f"Covers: {len(events)} audit event(s), {len(signers)} signer(s), full forensic fields.", size=7.5, color=gray)
    write("Verify: recompute HMAC-SHA256 over the canonical JSON payload (sort_keys=True, no spaces) with SECRET_KEY.", size=7, color=gray)
