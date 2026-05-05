"""Parse a pytest junit XML report from the live-connector suite and, if any
``test_live_<connector>_*`` cases failed, post a single notification naming the
broken connectors.

Designed to be wired into a nightly CI job after ``pytest -m live
--junitxml=live-results.xml``. A clean run is silent — we only ping when at
least one live test failed or errored, so the channel doesn't get a daily
"all green" notification.

Notification channels (pick whichever the CI environment provides):

* ``SLACK_WEBHOOK_URL`` — incoming-webhook URL; we POST a JSON ``{"text": ...}``
* ``ALERT_EMAIL_TO`` + ``SMTP_HOST`` (optional ``SMTP_PORT``, ``SMTP_USER``,
  ``SMTP_PASSWORD``, ``ALERT_EMAIL_FROM``) — sends a plain-text email
* ``PAGERDUTY_ROUTING_KEY`` — triggers a PagerDuty Events API v2 alert

If none of those are set the script falls back to printing the summary to
stderr (still useful as a CI annotation) and exits non-zero so the job stays
red.

Usage:
    python scripts/notify_live_failures.py live-results.xml
"""
from __future__ import annotations

import json
import os
import re
import smtplib
import sys
import urllib.request
import xml.etree.ElementTree as ET
from email.message import EmailMessage
from pathlib import Path


# ``test_live_<connector>_<rest>`` — the connector slug is the second token.
# ``google_drive`` is special-cased because the test names use ``google``.
_LIVE_TEST_RE = re.compile(r"^test_live_(?P<connector>[a-z0-9]+)(?:_.*)?$")
_CONNECTOR_DISPLAY = {
    "slack": "Slack",
    "notion": "Notion",
    "github": "GitHub",
    "linear": "Linear",
    "hubspot": "HubSpot",
    "dropbox": "Dropbox",
    "jira": "Jira",
    "s3": "Amazon S3",
    "google": "Google Drive",
}


def _connector_for(test_name: str) -> str | None:
    m = _LIVE_TEST_RE.match(test_name)
    if not m:
        return None
    return _CONNECTOR_DISPLAY.get(m["connector"], m["connector"])


def collect_failures(junit_path: Path) -> list[dict]:
    """Return a list of ``{connector, test, message}`` for every failed/errored
    live test in the junit report. Skipped tests (missing creds) are ignored.

    Suite-level errors (collection failures, internal pytest errors that pytest
    records as a ``<testsuite>``/``<testcase>`` ``error`` without matching any
    ``test_live_<connector>_*`` name) are surfaced as a single synthetic
    ``"pytest"`` failure so the notifier still pages someone — connector
    regressions can show up as import errors, and we don't want a false-green.
    """
    if not junit_path.exists():
        # Treated as a failure by the caller — see ``main``.
        raise FileNotFoundError(f"junit report not found: {junit_path}")
    tree = ET.parse(junit_path)
    failures: list[dict] = []
    for case in tree.iter("testcase"):
        name = case.get("name", "")
        connector = _connector_for(name)
        # NB: ElementTree Elements with no children are falsy, so don't use
        # ``or`` — explicitly fall back when ``failure`` isn't present.
        bad = case.find("failure")
        if bad is None:
            bad = case.find("error")
        if bad is None:
            continue
        message = (bad.get("message") or bad.text or "").strip()
        first_line = message.splitlines()[0] if message else "(no message)"
        if connector is None:
            # Collection / import / internal error attached to a non-live
            # testcase node. We still want to alert.
            failures.append(
                {
                    "connector": "pytest",
                    "test": name or "(collection error)",
                    "message": first_line[:500],
                }
            )
            continue
        failures.append(
            {"connector": connector, "test": name, "message": first_line[:500]}
        )
    # Suite-level <error>/<failure> children (e.g. ``<testsuite errors="1">``
    # with an ``<error>`` directly under it for a top-level collection crash).
    for suite in tree.iter("testsuite"):
        for tag in ("error", "failure"):
            node = suite.find(tag)
            if node is None:
                continue
            message = (node.get("message") or node.text or "").strip()
            first_line = message.splitlines()[0] if message else "(no message)"
            failures.append(
                {
                    "connector": "pytest",
                    "test": f"(testsuite {tag})",
                    "message": first_line[:500],
                }
            )
    return failures


def _format_summary(failures: list[dict]) -> str:
    connectors = sorted({f["connector"] for f in failures})
    header = (
        f"Nightly live-connector tests failed for: {', '.join(connectors)}"
    )
    body_lines = [header, ""]
    for f in failures:
        body_lines.append(f"• [{f['connector']}] {f['test']}: {f['message']}")
    run_url = os.environ.get("CI_RUN_URL") or os.environ.get(
        "GITHUB_SERVER_URL"
    )
    if os.environ.get("GITHUB_RUN_ID") and os.environ.get("GITHUB_REPOSITORY"):
        run_url = (
            f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
            f"{os.environ['GITHUB_RUN_ID']}"
        )
    if run_url:
        body_lines.append("")
        body_lines.append(f"Run: {run_url}")
    return "\n".join(body_lines)


def _post_slack(webhook_url: str, summary: str) -> bool:
    payload = json.dumps({"text": summary}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:  # pragma: no cover - network path
        print(f"Slack notification failed: {exc}", file=sys.stderr)
        return False


def _send_email(summary: str, failures: list[dict]) -> bool:
    to_addr = os.environ["ALERT_EMAIL_TO"]
    from_addr = os.environ.get("ALERT_EMAIL_FROM", to_addr)
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")

    msg = EmailMessage()
    connectors = sorted({f["connector"] for f in failures})
    msg["Subject"] = (
        f"[idpkit] Nightly live tests failed: {', '.join(connectors)}"
    )
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(summary)
    try:
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls()
            if user and password:
                s.login(user, password)
            s.send_message(msg)
        return True
    except Exception as exc:  # pragma: no cover - network path
        print(f"Email notification failed: {exc}", file=sys.stderr)
        return False


def _post_pagerduty(routing_key: str, summary: str, failures: list[dict]) -> bool:
    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "dedup_key": "idpkit-nightly-live-connectors",
        "payload": {
            "summary": summary.splitlines()[0],
            "severity": "error",
            "source": os.environ.get("GITHUB_REPOSITORY", "idpkit-ci"),
            "custom_details": {
                "failures": failures,
                "full_summary": summary,
            },
        },
    }
    req = urllib.request.Request(
        "https://events.pagerduty.com/v2/enqueue",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:  # pragma: no cover - network path
        print(f"PagerDuty notification failed: {exc}", file=sys.stderr)
        return False


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: notify_live_failures.py <junit-xml>", file=sys.stderr)
        return 2
    try:
        failures = collect_failures(Path(argv[1]))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # The CI workflow stashes pytest's real exit code in this env var so we
    # can distinguish "pytest ran cleanly" from "pytest blew up but the
    # junit happened to contain no recognizable failures". A non-zero
    # pytest exit *must* alert even if the parser found nothing.
    pytest_exit = os.environ.get("PYTEST_EXIT_CODE")
    pytest_failed = pytest_exit not in (None, "", "0")
    if pytest_failed and not failures:
        failures.append(
            {
                "connector": "pytest",
                "test": "(pytest exited non-zero with no parsed failures)",
                "message": (
                    f"pytest exit code {pytest_exit} — likely a collection or "
                    "internal error; check the run logs."
                ),
            }
        )

    if not failures:
        # Silent on a clean run, per the task spec.
        return 0

    summary = _format_summary(failures)
    print(summary, file=sys.stderr)

    delivered = False
    if webhook := os.environ.get("SLACK_WEBHOOK_URL"):
        delivered = _post_slack(webhook, summary) or delivered
    if os.environ.get("ALERT_EMAIL_TO") and os.environ.get("SMTP_HOST"):
        delivered = _send_email(summary, failures) or delivered
    if pd_key := os.environ.get("PAGERDUTY_ROUTING_KEY"):
        delivered = _post_pagerduty(pd_key, summary, failures) or delivered

    if not delivered:
        print(
            "No notification channel configured (set SLACK_WEBHOOK_URL, "
            "ALERT_EMAIL_TO+SMTP_HOST, or PAGERDUTY_ROUTING_KEY); failure "
            "summary printed above.",
            file=sys.stderr,
        )
    # Always exit non-zero so the CI step itself is also red.
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
