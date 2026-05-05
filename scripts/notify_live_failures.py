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

De-duplication
--------------

To stop the channel turning into a stream of identical "Slack still broken"
pings every night, the script persists a tiny JSON state file between runs
(see ``--state-file`` / ``NOTIFY_STATE_FILE``). On each invocation it
compares the current set of failing connectors with the previous run and
only notifies for *changes*:

* ``new`` failure (a connector that was green or unseen last run) — alert
* ``recovered`` (a connector that was failing last run but is green now) —
  separate "recovered" notification
* ``still failing`` — silent unless the last alert was more than
  ``NOTIFY_REMINDER_DAYS`` days ago (default 7), in which case we send a
  reminder so a forgotten broken connector doesn't drift forever

The script's CI exit code is unchanged: any failing connector still exits
non-zero so the workflow step itself stays red, even on quiet days where
no notification was sent.

Usage:
    python scripts/notify_live_failures.py live-results.xml \
        [--state-file path/to/state.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import smtplib
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
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

_DEFAULT_REMINDER_DAYS = 7


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


# ---------------------------------------------------------------------------
# Dedup state
# ---------------------------------------------------------------------------


def _now() -> datetime:
    """Indirection so tests can monkeypatch the clock."""
    return datetime.now(timezone.utc)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_state(path: Path | None) -> dict:
    """Load persisted dedup state. Missing/corrupt files are treated as empty
    so a first-time run (or a wiped CI cache) just behaves like every
    connector is newly-seen."""
    if path is None or not path.exists():
        return {"connectors": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"warning: ignoring unreadable state file {path}: {exc}",
            file=sys.stderr,
        )
        return {"connectors": {}}
    if not isinstance(data, dict) or not isinstance(
        data.get("connectors"), dict
    ):
        return {"connectors": {}}
    return data


def save_state(path: Path | None, state: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True))
    except OSError as exc:  # pragma: no cover - disk full / permissions
        print(f"warning: could not write state file {path}: {exc}", file=sys.stderr)


def classify(
    failures: list[dict],
    state: dict,
    *,
    reminder_days: int,
    now: datetime | None = None,
) -> tuple[list[dict], list[str], list[dict], dict]:
    """Bucket the current run against persisted state.

    Returns ``(new_or_reminder_failures, recovered_connectors,
    still_silent_failures, next_state)`` where:

    * ``new_or_reminder_failures`` — failures we should notify about: either
      a connector that wasn't failing last run, or one that's been failing
      for ``>= reminder_days`` since the last alert.
    * ``recovered_connectors`` — connectors that were failing last run but
      have no failures this run.
    * ``still_silent_failures`` — failures we're intentionally suppressing
      (already-known, not yet due for a reminder). Returned for logging /
      tests; not surfaced to the user.
    * ``next_state`` — the state dict to persist for the next run.
    """
    now = now or _now()
    prev = state.get("connectors", {})

    by_conn: dict[str, list[dict]] = {}
    for f in failures:
        by_conn.setdefault(f["connector"], []).append(f)

    new_or_reminder: list[dict] = []
    silent: list[dict] = []
    next_connectors: dict[str, dict] = {}
    reminder_delta = timedelta(days=reminder_days)

    for conn, conn_failures in by_conn.items():
        prev_entry = prev.get(conn) or {}
        prev_status = prev_entry.get("status")
        first_failed_at = (
            prev_entry.get("first_failed_at")
            if prev_status == "failing"
            else now.isoformat()
        )
        last_alerted_at_dt = _parse_iso(prev_entry.get("last_alerted_at"))

        is_new = prev_status != "failing"
        due_for_reminder = (
            last_alerted_at_dt is not None
            and (now - last_alerted_at_dt) >= reminder_delta
        )

        if is_new or due_for_reminder:
            new_or_reminder.extend(conn_failures)
            last_alerted_at = now.isoformat()
        else:
            silent.extend(conn_failures)
            last_alerted_at = prev_entry.get("last_alerted_at") or now.isoformat()

        next_connectors[conn] = {
            "status": "failing",
            "first_failed_at": first_failed_at,
            "last_alerted_at": last_alerted_at,
            "last_seen_at": now.isoformat(),
        }

    recovered: list[str] = []
    for conn, prev_entry in prev.items():
        if conn in by_conn:
            continue
        if prev_entry.get("status") == "failing":
            recovered.append(conn)
        # Drop recovered connectors from the persisted state — once they're
        # green again we don't need to remember them. PagerDuty gets an
        # explicit ``resolve`` event using the same dedup_key.

    next_state = {
        "connectors": next_connectors,
        "updated_at": now.isoformat(),
    }
    return new_or_reminder, sorted(recovered), silent, next_state


# ---------------------------------------------------------------------------
# Formatting & delivery
# ---------------------------------------------------------------------------


def _run_url() -> str | None:
    if os.environ.get("CI_RUN_URL"):
        return os.environ["CI_RUN_URL"]
    if os.environ.get("GITHUB_RUN_ID") and os.environ.get("GITHUB_REPOSITORY"):
        return (
            f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
            f"{os.environ['GITHUB_RUN_ID']}"
        )
    return None


def _format_failure_summary(failures: list[dict], *, reminder: bool = False) -> str:
    connectors = sorted({f["connector"] for f in failures})
    if reminder:
        header = (
            "Reminder: nightly live-connector tests still failing for: "
            f"{', '.join(connectors)}"
        )
    else:
        header = (
            f"Nightly live-connector tests failed for: {', '.join(connectors)}"
        )
    body_lines = [header, ""]
    for f in failures:
        body_lines.append(f"• [{f['connector']}] {f['test']}: {f['message']}")
    if (run_url := _run_url()):
        body_lines.append("")
        body_lines.append(f"Run: {run_url}")
    return "\n".join(body_lines)


def _format_recovery_summary(connectors: list[str]) -> str:
    header = (
        f"Nightly live-connector tests recovered for: {', '.join(connectors)}"
    )
    body_lines = [header]
    if (run_url := _run_url()):
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


def _send_email(summary: str, subject: str) -> bool:
    to_addr = os.environ["ALERT_EMAIL_TO"]
    from_addr = os.environ.get("ALERT_EMAIL_FROM", to_addr)
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")

    msg = EmailMessage()
    msg["Subject"] = subject
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


def _pagerduty_dedup_key(connector: str) -> str:
    # Per-connector dedup_key so PagerDuty dedupes server-side too: a second
    # ``trigger`` with the same key just updates the open incident instead
    # of opening a fresh one, and ``resolve`` closes exactly that incident.
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", connector).strip("_").lower() or "unknown"
    return f"idpkit-nightly-live-{safe}"


def _post_pagerduty_event(
    routing_key: str,
    *,
    action: str,
    dedup_key: str,
    summary: str,
    custom_details: dict | None = None,
) -> bool:
    payload: dict = {
        "routing_key": routing_key,
        "event_action": action,
        "dedup_key": dedup_key,
    }
    if action == "trigger":
        payload["payload"] = {
            "summary": summary.splitlines()[0],
            "severity": "error",
            "source": os.environ.get("GITHUB_REPOSITORY", "idpkit-ci"),
            "custom_details": custom_details or {},
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


def _notify(summary: str, *, subject: str, failures: list[dict]) -> bool:
    """Fan out a single textual notification to every configured channel.
    Returns True if at least one channel accepted the message."""
    delivered = False
    if webhook := os.environ.get("SLACK_WEBHOOK_URL"):
        delivered = _post_slack(webhook, summary) or delivered
    if os.environ.get("ALERT_EMAIL_TO") and os.environ.get("SMTP_HOST"):
        delivered = _send_email(summary, subject) or delivered
    if pd_key := os.environ.get("PAGERDUTY_ROUTING_KEY"):
        # One PD event per affected connector so each gets its own incident
        # lifecycle (trigger → resolve) keyed on the connector name.
        connectors = sorted({f["connector"] for f in failures}) or ["pytest"]
        for conn in connectors:
            conn_failures = [f for f in failures if f["connector"] == conn]
            ok = _post_pagerduty_event(
                pd_key,
                action="trigger",
                dedup_key=_pagerduty_dedup_key(conn),
                summary=summary,
                custom_details={
                    "connector": conn,
                    "failures": conn_failures,
                    "full_summary": summary,
                },
            )
            delivered = ok or delivered
    return delivered


def _notify_recovery(summary: str, *, subject: str, connectors: list[str]) -> bool:
    delivered = False
    if webhook := os.environ.get("SLACK_WEBHOOK_URL"):
        delivered = _post_slack(webhook, summary) or delivered
    if os.environ.get("ALERT_EMAIL_TO") and os.environ.get("SMTP_HOST"):
        delivered = _send_email(summary, subject) or delivered
    if pd_key := os.environ.get("PAGERDUTY_ROUTING_KEY"):
        for conn in connectors:
            ok = _post_pagerduty_event(
                pd_key,
                action="resolve",
                dedup_key=_pagerduty_dedup_key(conn),
                summary=summary,
            )
            delivered = ok or delivered
    return delivered


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="notify_live_failures.py")
    parser.add_argument("junit", type=Path, help="path to pytest junit XML")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help=(
            "JSON file used to dedupe alerts across runs. Defaults to "
            "$NOTIFY_STATE_FILE if set, otherwise no dedup is performed "
            "and every failing run alerts (legacy behaviour)."
        ),
    )
    parser.add_argument(
        "--reminder-days",
        type=int,
        default=None,
        help=(
            "Re-alert about a still-failing connector at most once per N "
            "days. Defaults to $NOTIFY_REMINDER_DAYS or "
            f"{_DEFAULT_REMINDER_DAYS}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    # Backwards-compatible argparse: previously the script accepted a single
    # positional junit path. ``argv`` here is sys.argv-style (program name
    # at index 0).
    if len(argv) < 2:
        print(
            "usage: notify_live_failures.py <junit-xml> [--state-file PATH] "
            "[--reminder-days N]",
            file=sys.stderr,
        )
        return 2
    try:
        args = _parse_args(argv[1:])
    except SystemExit as exc:
        return int(exc.code or 2)

    state_file = args.state_file
    if state_file is None and (env_path := os.environ.get("NOTIFY_STATE_FILE")):
        state_file = Path(env_path)
    reminder_days = args.reminder_days
    if reminder_days is None:
        reminder_days = int(
            os.environ.get("NOTIFY_REMINDER_DAYS", _DEFAULT_REMINDER_DAYS)
        )

    try:
        failures = collect_failures(args.junit)
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

    state = load_state(state_file)
    new_or_reminder, recovered, silent, next_state = classify(
        failures, state, reminder_days=reminder_days
    )
    save_state(state_file, next_state)

    # Determine if this is a "reminder" wave: every failure we're notifying
    # about was already known and is being re-pinged because of the time
    # threshold. A mix counts as new (the new connector is the headline).
    prev_failing = {
        conn
        for conn, entry in state.get("connectors", {}).items()
        if entry.get("status") == "failing"
    }
    notifying_connectors = sorted({f["connector"] for f in new_or_reminder})
    is_reminder_only = bool(notifying_connectors) and all(
        c in prev_failing for c in notifying_connectors
    )

    sent_any = False

    if new_or_reminder:
        summary = _format_failure_summary(
            new_or_reminder, reminder=is_reminder_only
        )
        print(summary, file=sys.stderr)
        subject_prefix = "[idpkit] Nightly live tests"
        subject = (
            f"{subject_prefix} {'reminder' if is_reminder_only else 'failed'}: "
            f"{', '.join(notifying_connectors)}"
        )
        delivered = _notify(summary, subject=subject, failures=new_or_reminder)
        if not delivered and not (
            os.environ.get("SLACK_WEBHOOK_URL")
            or (os.environ.get("ALERT_EMAIL_TO") and os.environ.get("SMTP_HOST"))
            or os.environ.get("PAGERDUTY_ROUTING_KEY")
        ):
            print(
                "No notification channel configured (set SLACK_WEBHOOK_URL, "
                "ALERT_EMAIL_TO+SMTP_HOST, or PAGERDUTY_ROUTING_KEY); failure "
                "summary printed above.",
                file=sys.stderr,
            )
        sent_any = True

    if recovered:
        summary = _format_recovery_summary(recovered)
        print(summary, file=sys.stderr)
        subject = f"[idpkit] Nightly live tests recovered: {', '.join(recovered)}"
        _notify_recovery(summary, subject=subject, connectors=recovered)
        sent_any = True

    if silent and not new_or_reminder:
        # Helpful breadcrumb in CI logs so a maintainer skimming a "silent"
        # nightly run can still see what's known-broken without digging
        # into the state file.
        connectors = sorted({f["connector"] for f in silent})
        print(
            "Suppressing duplicate alert for already-known failing "
            f"connectors: {', '.join(connectors)} (next reminder in up to "
            f"{reminder_days} days).",
            file=sys.stderr,
        )

    # Exit code contract: any failing connector this run keeps the CI step
    # red, even if we didn't notify. A pure-recovery run is green.
    if failures:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
