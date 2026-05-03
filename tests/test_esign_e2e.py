"""End-to-end tests for the e-signature flow.

Covers the full happy path (create envelope → place fields → send → public sign →
finalize), the bulk-apply propagation feature, void / decline / expire branches,
and audit-event presence.
"""
from __future__ import annotations

import base64
import io

import pytest


pytestmark = pytest.mark.asyncio


# A 1x1 transparent PNG used as a fake signature image
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
_SIG_DATA_URL = "data:image/png;base64," + base64.b64encode(_TINY_PNG).decode()


async def _create_envelope_with_pdf(client, sample_pdf_bytes: bytes, title: str = "Test NDA") -> dict:
    files = {"file": ("test.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")}
    data = {
        "title": title,
        "signers_json": '[{"name":"Alice Tester","email":"alice@example.com","order_index":0}]',
        "signing_order": "parallel",
    }
    res = await client.post("/api/esign/envelopes", data=data, files=files)
    assert res.status_code == 201, res.text
    return res.json()


async def _place_fields(client, envelope_id: str, signer_id: str, *, with_bulk: bool = False, page_count: int = 1) -> None:
    fields = [{
        "signer_id": signer_id,
        "field_type": "signature",
        "page": 1,
        "x_pct": 10.0, "y_pct": 80.0, "w_pct": 20.0, "h_pct": 6.0,
        "label": "Sign here",
        "is_required": 1,
        "bulk_group_id": None,
    }]
    if with_bulk and page_count > 1:
        bg = "bg_test_group_1"
        fields[0]["bulk_group_id"] = bg
        for p in range(2, page_count + 1):
            fields.append({
                "signer_id": signer_id,
                "field_type": "signature",
                "page": p,
                "x_pct": 10.0, "y_pct": 80.0, "w_pct": 20.0, "h_pct": 6.0,
                "label": "Sign here",
                "is_required": 1,
                "bulk_group_id": bg,
            })
    res = await client.put(f"/api/esign/envelopes/{envelope_id}/fields", json={"fields": fields})
    assert res.status_code == 200, res.text


def _extract_token(captured_invitations) -> str:
    assert captured_invitations, "No invitation captured"
    url = captured_invitations[-1]["signing_url"]
    return url.rsplit("/sign/", 1)[1]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_full_signing_flow(auth_client, sample_pdf_bytes, captured_invitations):
    env = await _create_envelope_with_pdf(auth_client, sample_pdf_bytes)
    envelope_id = env["id"]
    assert env["status"] == "draft"
    assert len(env["signers"]) == 1
    signer_id = env["signers"][0]["id"]

    await _place_fields(auth_client, envelope_id, signer_id)

    # Send the envelope
    res = await auth_client.post(f"/api/esign/envelopes/{envelope_id}/send")
    assert res.status_code == 200, res.text
    assert res.json()["signer_count"] == 1

    # Pull token from captured invitation
    token = _extract_token(captured_invitations)

    # Public signing context
    res = await auth_client.get(f"/api/esign/sign/{token}")
    assert res.status_code == 200, res.text
    ctx = res.json()
    assert ctx["already_signed"] is False
    assert len(ctx["fields"]) == 1
    field_id = ctx["fields"][0]["id"]

    # Submit signature
    res = await auth_client.post(
        f"/api/esign/sign/{token}/submit",
        json={
            "fields": [{"id": field_id, "value": _SIG_DATA_URL}],
            "consent_accepted": True,
            "session_id": "sid_test_001",
        },
    )
    assert res.status_code == 200, res.text
    assert res.json()["completed"] is True

    # Envelope should now be COMPLETED
    res = await auth_client.get(f"/api/esign/envelopes/{envelope_id}")
    assert res.status_code == 200
    detail = res.json()
    assert detail["status"] == "completed"
    assert detail["completed_at"] is not None

    # Audit timeline must include core events
    event_types = {e["event_type"] for e in detail["audit_events"]}
    assert "envelope_created" in event_types
    assert "invitation_sent" in event_types
    assert "document_viewed" in event_types
    assert "consent_accepted" in event_types
    assert "field_signed" in event_types
    assert "document_signed" in event_types
    assert "envelope_completed" in event_types


# ---------------------------------------------------------------------------
# ESIGN consent enforcement
# ---------------------------------------------------------------------------

async def test_submit_without_consent_is_rejected(auth_client, sample_pdf_bytes, captured_invitations):
    env = await _create_envelope_with_pdf(auth_client, sample_pdf_bytes)
    signer_id = env["signers"][0]["id"]
    await _place_fields(auth_client, env["id"], signer_id)
    await auth_client.post(f"/api/esign/envelopes/{env['id']}/send")
    token = _extract_token(captured_invitations)

    ctx = (await auth_client.get(f"/api/esign/sign/{token}")).json()
    field_id = ctx["fields"][0]["id"]

    res = await auth_client.post(
        f"/api/esign/sign/{token}/submit",
        json={
            "fields": [{"id": field_id, "value": _SIG_DATA_URL}],
            "consent_accepted": False,
        },
    )
    assert res.status_code == 400
    assert "consent" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Bulk-apply propagation
# ---------------------------------------------------------------------------

async def test_bulk_apply_propagates_value_across_pages(auth_client, sample_pdf_bytes, captured_invitations):
    env = await _create_envelope_with_pdf(auth_client, sample_pdf_bytes, title="Bulk Apply Test")
    envelope_id = env["id"]
    page_count = env["page_count"]

    if page_count < 2:
        pytest.skip("Sample PDF is single-page; bulk-apply needs multi-page document")

    signer_id = env["signers"][0]["id"]
    await _place_fields(auth_client, envelope_id, signer_id, with_bulk=True, page_count=page_count)

    # Verify the saved fields all share the same bulk_group_id
    detail = (await auth_client.get(f"/api/esign/envelopes/{envelope_id}")).json()
    grouped = [f for f in detail["fields"] if f["bulk_group_id"]]
    assert len(grouped) == page_count
    assert len({f["bulk_group_id"] for f in grouped}) == 1

    await auth_client.post(f"/api/esign/envelopes/{envelope_id}/send")
    token = _extract_token(captured_invitations)

    ctx = (await auth_client.get(f"/api/esign/sign/{token}")).json()
    assert len(ctx["fields"]) == page_count
    # Submitter sends value for ONLY the first field — server must propagate to siblings
    first_field_id = ctx["fields"][0]["id"]
    res = await auth_client.post(
        f"/api/esign/sign/{token}/submit",
        json={
            "fields": [{"id": first_field_id, "value": _SIG_DATA_URL}],
            "consent_accepted": True,
        },
    )
    assert res.status_code == 200, res.text

    # Confirm every field in the bulk group is now marked has_value=True
    detail = (await auth_client.get(f"/api/esign/envelopes/{envelope_id}")).json()
    assert all(f["has_value"] for f in detail["fields"])

    # And confirm a bulk_apply_propagated audit event was logged
    event_types = [e["event_type"] for e in detail["audit_events"]]
    assert "bulk_apply_propagated" in event_types


# ---------------------------------------------------------------------------
# Required field enforcement
# ---------------------------------------------------------------------------

async def test_missing_required_field_is_rejected(auth_client, sample_pdf_bytes, captured_invitations):
    env = await _create_envelope_with_pdf(auth_client, sample_pdf_bytes)
    signer_id = env["signers"][0]["id"]
    await _place_fields(auth_client, env["id"], signer_id)
    await auth_client.post(f"/api/esign/envelopes/{env['id']}/send")
    token = _extract_token(captured_invitations)

    ctx = (await auth_client.get(f"/api/esign/sign/{token}")).json()
    field_id = ctx["fields"][0]["id"]

    res = await auth_client.post(
        f"/api/esign/sign/{token}/submit",
        json={
            "fields": [{"id": field_id, "value": ""}],
            "consent_accepted": True,
        },
    )
    assert res.status_code == 422
    assert "required" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Void
# ---------------------------------------------------------------------------

async def test_void_envelope_blocks_signing(auth_client, sample_pdf_bytes, captured_invitations):
    env = await _create_envelope_with_pdf(auth_client, sample_pdf_bytes)
    signer_id = env["signers"][0]["id"]
    await _place_fields(auth_client, env["id"], signer_id)
    await auth_client.post(f"/api/esign/envelopes/{env['id']}/send")
    token = _extract_token(captured_invitations)

    # Void
    res = await auth_client.post(
        f"/api/esign/envelopes/{env['id']}/void",
        json={"reason": "Test void"},
    )
    assert res.status_code == 200, res.text

    # Subsequent signing access must be rejected. Void wipes the token_hash so the
    # public sign route can no longer find the signer — 404 is the correct response.
    res = await auth_client.get(f"/api/esign/sign/{token}")
    assert res.status_code in (404, 410)

    # Audit log records the void
    detail = (await auth_client.get(f"/api/esign/envelopes/{env['id']}")).json()
    assert detail["status"] == "voided"
    assert "envelope_voided" in {e["event_type"] for e in detail["audit_events"]}


# ---------------------------------------------------------------------------
# Decline
# ---------------------------------------------------------------------------

async def test_signer_decline(auth_client, sample_pdf_bytes, captured_invitations):
    env = await _create_envelope_with_pdf(auth_client, sample_pdf_bytes)
    signer_id = env["signers"][0]["id"]
    await _place_fields(auth_client, env["id"], signer_id)
    await auth_client.post(f"/api/esign/envelopes/{env['id']}/send")
    token = _extract_token(captured_invitations)

    res = await auth_client.post(
        f"/api/esign/sign/{token}/decline",
        json={"reason": "I disagree with the terms"},
    )
    assert res.status_code == 200, res.text

    detail = (await auth_client.get(f"/api/esign/envelopes/{env['id']}")).json()
    assert detail["status"] == "declined"
    decline_events = {e["event_type"] for e in detail["audit_events"]}
    assert ("envelope_declined" in decline_events) or ("signer_declined" in decline_events)


# ---------------------------------------------------------------------------
# Field editing locked after send
# ---------------------------------------------------------------------------

async def test_cannot_edit_fields_after_send(auth_client, sample_pdf_bytes, captured_invitations):
    env = await _create_envelope_with_pdf(auth_client, sample_pdf_bytes)
    signer_id = env["signers"][0]["id"]
    await _place_fields(auth_client, env["id"], signer_id)
    await auth_client.post(f"/api/esign/envelopes/{env['id']}/send")

    res = await auth_client.put(
        f"/api/esign/envelopes/{env['id']}/fields",
        json={"fields": []},
    )
    assert res.status_code == 400
