"""E-signature email delivery via the external email API."""

import asyncio
import base64
import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Optional

logger = logging.getLogger(__name__)


def _smtp_config() -> Optional[dict]:
    """Read SMTP settings from the environment.

    Self-hosted deployments configure SMTP directly (the Replit email relay
    used on the `replit` branch is intentionally not used here). Returns None
    when ``SMTP_HOST`` is unset, in which case emails are logged rather than
    sent — preserving the previous no-credentials fallback behavior.
    """
    host = os.getenv("SMTP_HOST")
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER"),
        "password": os.getenv("SMTP_PASSWORD"),
        "from_addr": os.getenv("SMTP_FROM") or os.getenv("SMTP_USER") or "no-reply@localhost",
        "use_ssl": os.getenv("SMTP_SSL", "").lower() in ("1", "true", "yes"),
        "use_starttls": os.getenv("SMTP_STARTTLS", "true").lower() in ("1", "true", "yes"),
    }


def _html_wrapper(body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f4f6f9; margin: 0; padding: 20px; }}
  .container {{ max-width: 600px; margin: 0 auto; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .header {{ background: linear-gradient(135deg, #4f46e5, #7c3aed); padding: 32px 40px; text-align: center; }}
  .header h1 {{ color: #fff; margin: 0; font-size: 22px; font-weight: 700; }}
  .header p {{ color: rgba(255,255,255,0.85); margin: 8px 0 0; font-size: 14px; }}
  .body {{ padding: 36px 40px; }}
  .btn {{ display: inline-block; background: #4f46e5; color: #fff !important; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-size: 16px; font-weight: 600; margin: 24px 0; }}
  .footer {{ background: #f8fafc; padding: 20px 40px; text-align: center; font-size: 12px; color: #9ca3af; border-top: 1px solid #e5e7eb; }}
  .meta {{ background: #f8fafc; border-radius: 8px; padding: 16px; margin: 16px 0; font-size: 13px; color: #64748b; }}
  .meta strong {{ color: #374151; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>IDP Kit &mdash; E-Sign</h1>
    <p>Secure Electronic Signature Platform</p>
  </div>
  <div class="body">
    {body_html}
  </div>
  <div class="footer">
    This email was sent by IDP Kit E-Sign. Your signature has legal validity under the ESIGN Act and eIDAS regulations.
  </div>
</div>
</body>
</html>"""


async def send_signing_invitation(
    signer_name: str,
    signer_email: str,
    sender_name: str,
    envelope_title: str,
    signing_url: str,
    message: Optional[str] = None,
    expires_at: Optional[str] = None,
    is_resend: bool = False,
) -> bool:
    """Send a signing invitation email to a signer."""
    msg_block = f'<blockquote style="border-left:3px solid #4f46e5;margin:16px 0;padding:8px 16px;color:#374151;font-style:italic;">{message}</blockquote>' if message else ""
    expiry_block = f'<p style="font-size:13px;color:#9ca3af;">This link expires on {expires_at}.</p>' if expires_at else ""
    resend_block = (
        '<p style="background:#fef3c7;border-left:3px solid #f59e0b;padding:8px 12px;margin:0 0 12px;color:#92400e;font-size:13px;">'
        '<strong>This is a fresh invitation.</strong> Any earlier signing link you received for this document is no longer valid — please use the button below.'
        '</p>'
        if is_resend else ""
    )

    body_html = f"""
    <p style="font-size:16px;color:#374151;">Hello <strong>{signer_name}</strong>,</p>
    {resend_block}
    <p style="color:#374151;"><strong>{sender_name}</strong> has requested your electronic signature on:</p>
    <div class="meta">
      <strong>Document:</strong> {envelope_title}
    </div>
    {msg_block}
    <p style="color:#374151;">Click the button below to review and sign the document. No account is required.</p>
    <div style="text-align:center;">
      <a href="{signing_url}" class="btn">Review &amp; Sign Document</a>
    </div>
    {expiry_block}
    <p style="font-size:12px;color:#9ca3af;">If the button does not work, copy and paste this link: <br>{signing_url}</p>
    """

    subject_prefix = "Reminder: Please sign" if is_resend else "Action Required: Please sign"
    return await _send(
        to=signer_email,
        subject=f"{subject_prefix} \"{envelope_title}\"",
        body=_html_wrapper(body_html),
        signing_url=signing_url,
    )


async def send_completion_notice(
    recipient_email: str,
    recipient_name: str,
    envelope_title: str,
    download_url: str,
    pdf_bytes: Optional[bytes] = None,
    filename: str = "signed_document.pdf",
) -> bool:
    """Notify all parties that the envelope is complete and the signed document is available."""
    body_html = f"""
    <p style="font-size:16px;color:#374151;">Hello <strong>{recipient_name}</strong>,</p>
    <p style="color:#374151;">All parties have signed <strong>{envelope_title}</strong>. The completed document is now available.</p>
    <div style="text-align:center;">
      <a href="{download_url}" class="btn">Download Signed Document</a>
    </div>
    <p style="font-size:12px;color:#9ca3af;">The signed PDF is attached to this email for your records.</p>
    <p style="font-size:12px;color:#9ca3af;">Direct link: {download_url}</p>
    """

    attachments = []
    if pdf_bytes:
        attachments = [{
            "filename": filename,
            "content": base64.b64encode(pdf_bytes).decode(),
            "content_type": "application/pdf",
            "encoding": "base64",
        }]

    return await _send(
        to=recipient_email,
        subject=f"Completed: \"{envelope_title}\" — All parties have signed",
        body=_html_wrapper(body_html),
        attachments=attachments,
        signing_url=download_url,
    )


async def send_decline_notice(
    recipient_email: str,
    recipient_name: str,
    envelope_title: str,
    declined_by_name: str,
    declined_by_email: str,
    reason: Optional[str] = None,
    is_owner: bool = False,
) -> bool:
    """Notify the owner (and remaining signers) that a signer has declined.
    All user-supplied strings are HTML-escaped before interpolation so a
    malicious decline reason or signer name cannot inject markup or scripts
    into the rendered email."""
    import html as _html
    e_recipient = _html.escape(recipient_name or "")
    e_title = _html.escape(envelope_title or "")
    e_decl_name = _html.escape(declined_by_name or "")
    e_decl_email = _html.escape(declined_by_email or "")
    e_reason = _html.escape(reason) if reason else None
    reason_block = (
        f'<blockquote style="border-left:3px solid #ef4444;margin:16px 0;padding:8px 16px;color:#374151;font-style:italic;">{e_reason}</blockquote>'
        if e_reason else ""
    )
    if is_owner:
        intro = (
            f"<strong>{e_decl_name}</strong> &lt;{e_decl_email}&gt; has <strong>declined</strong> to sign "
            f"<strong>{e_title}</strong>. The envelope has been marked as declined and will not be completed."
        )
    else:
        intro = (
            f"The envelope <strong>{e_title}</strong> has been declined by another signer "
            f"(<strong>{e_decl_name}</strong>). No further action is required from you — any signing link "
            f"you previously received is now inactive."
        )
    body_html = f"""
    <p style="font-size:16px;color:#374151;">Hello <strong>{e_recipient}</strong>,</p>
    <p style="color:#374151;">{intro}</p>
    {reason_block}
    <p style="font-size:12px;color:#9ca3af;">If you believe this was in error, please contact the sender directly.</p>
    """
    return await _send(
        to=recipient_email,
        subject=f"Declined: \"{envelope_title}\"",
        body=_html_wrapper(body_html),
        signing_url=None,
    )


async def send_reactivate_notice(
    recipient_email: str,
    recipient_name: str,
    envelope_title: str,
    reactivated_by: str,
) -> bool:
    """Notify a prior signer that a previously-declined envelope has been reset to draft."""
    import html as _html
    e_recipient = _html.escape(recipient_name or "")
    e_title = _html.escape(envelope_title or "")
    e_by = _html.escape(reactivated_by or "")
    body_html = f"""
    <p style="font-size:16px;color:#374151;">Hello <strong>{e_recipient}</strong>,</p>
    <p style="color:#374151;">The envelope <strong>{e_title}</strong> has been reactivated by <strong>{e_by}</strong>.</p>
    <p style="color:#9ca3af;font-size:13px;">You will receive a fresh signing invitation when the envelope is sent again. Any earlier signing link you received is no longer valid.</p>
    """
    return await _send(
        to=recipient_email,
        subject=f"Reactivated: \"{envelope_title}\"",
        body=_html_wrapper(body_html),
        signing_url=None,
    )


async def send_void_notice(
    recipient_email: str,
    recipient_name: str,
    envelope_title: str,
    voided_by: str,
) -> bool:
    """Notify parties that an envelope has been voided."""
    body_html = f"""
    <p style="font-size:16px;color:#374151;">Hello <strong>{recipient_name}</strong>,</p>
    <p style="color:#374151;">The envelope for <strong>{envelope_title}</strong> has been voided by <strong>{voided_by}</strong>.</p>
    <p style="color:#9ca3af;font-size:13px;">Any previously sent signing links are now invalid.</p>
    """
    return await _send(
        to=recipient_email,
        subject=f"Voided: \"{envelope_title}\"",
        body=_html_wrapper(body_html),
        signing_url=None,
    )


def _send_smtp_sync(cfg: dict, to: str, subject: str, body_html: str, attachments: Optional[list]) -> None:
    """Blocking SMTP send. Run via asyncio.to_thread from the async _send()."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from_addr"]
    msg["To"] = to
    msg.set_content("This message requires an HTML-capable email client to view.")
    msg.add_alternative(body_html, subtype="html")

    for att in attachments or []:
        content = att.get("content")
        if att.get("encoding") == "base64" and isinstance(content, str):
            content = base64.b64decode(content)
        elif isinstance(content, str):
            content = content.encode()
        maintype, _, subtype = (att.get("content_type") or "application/octet-stream").partition("/")
        msg.add_attachment(
            content,
            maintype=maintype or "application",
            subtype=subtype or "octet-stream",
            filename=att.get("filename", "attachment"),
        )

    if cfg["use_ssl"]:
        server = smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=30)
    else:
        server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=30)
    try:
        if cfg["use_starttls"] and not cfg["use_ssl"]:
            server.starttls()
        if cfg["user"]:
            server.login(cfg["user"], cfg["password"])
        server.send_message(msg)
    finally:
        server.quit()


async def _send(
    to: str,
    subject: str,
    body: str,
    signing_url: Optional[str] = None,
    attachments: Optional[list] = None,
) -> bool:
    cfg = _smtp_config()
    if cfg is None:
        body_preview = body[:2000] + "…[truncated]" if len(body) > 2000 else body
        logger.warning(
            "[E-Sign EMAIL FALLBACK] No SMTP_HOST set — email not delivered.\n"
            "  To:      %s\n"
            "  Subject: %s\n"
            "  URL:     %s\n"
            "  Body:\n%s",
            to, subject, signing_url or "(no url)", body_preview,
        )
        return False

    try:
        await asyncio.to_thread(_send_smtp_sync, cfg, to, subject, body, attachments)
        logger.info("E-sign email sent to %s via SMTP (subject: %s)", to, subject)
        return True
    except Exception as exc:
        logger.error("E-sign SMTP send failed: %s", exc)
        return False
