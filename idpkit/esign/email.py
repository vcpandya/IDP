"""E-signature email delivery via the external email API."""

import base64
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

EMAIL_API_URL = "https://myutils.replit.app/send_email"


def _api_key() -> Optional[str]:
    return os.getenv("EMAIL_API_KEY")


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
) -> bool:
    """Send a signing invitation email to a signer."""
    msg_block = f'<blockquote style="border-left:3px solid #4f46e5;margin:16px 0;padding:8px 16px;color:#374151;font-style:italic;">{message}</blockquote>' if message else ""
    expiry_block = f'<p style="font-size:13px;color:#9ca3af;">This link expires on {expires_at}.</p>' if expires_at else ""

    body_html = f"""
    <p style="font-size:16px;color:#374151;">Hello <strong>{signer_name}</strong>,</p>
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

    return await _send(
        to=signer_email,
        subject=f"Action Required: Please sign \"{envelope_title}\"",
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


async def _send(
    to: str,
    subject: str,
    body: str,
    signing_url: Optional[str] = None,
    attachments: Optional[list] = None,
) -> bool:
    api_key = _api_key()
    if not api_key:
        logger.warning(
            "[E-Sign EMAIL FALLBACK] No EMAIL_API_KEY set.\n"
            "  To: %s\n  Subject: %s\n  URL: %s",
            to, subject, signing_url or "(no url)",
        )
        return False

    payload: dict = {
        "to": to,
        "subject": subject,
        "body": body,
    }
    if attachments:
        payload["attachments"] = attachments

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                EMAIL_API_URL,
                json=payload,
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            )
        if resp.status_code == 200:
            logger.info("E-sign email sent to %s (subject: %s)", to, subject)
            return True
        else:
            logger.error("E-sign email API error %d: %s", resp.status_code, resp.text[:200])
            return False
    except Exception as exc:
        logger.error("E-sign email send failed: %s", exc)
        return False
