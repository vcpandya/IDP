"""Unit tests for ``scripts/notify_live_failures.py``.

These don't hit the network — they only exercise the junit parser and the
top-level exit-code contract that the nightly CI workflow relies on.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "notify_live_failures.py"
_spec = importlib.util.spec_from_file_location("notify_live_failures", _SCRIPT)
notify = importlib.util.module_from_spec(_spec)
sys.modules["notify_live_failures"] = notify
_spec.loader.exec_module(notify)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "results.xml"
    p.write_text(body)
    return p


def test_collect_failures_picks_up_failure_and_error(tmp_path):
    junit = _write(tmp_path, """<?xml version='1.0'?>
<testsuites><testsuite name='pytest'>
  <testcase name='test_live_slack_list_channels'/>
  <testcase name='test_live_notion_search_pages'>
    <failure message='assert False'>tb...</failure>
  </testcase>
  <testcase name='test_live_google_drive_search'>
    <error message='boom'/>
  </testcase>
  <testcase name='test_live_jira_health_and_search'><skipped/></testcase>
  <testcase name='test_unrelated'><failure message='nope'/></testcase>
</testsuite></testsuites>""")
    failures = notify.collect_failures(junit)
    by_connector = sorted(f["connector"] for f in failures)
    # ``test_unrelated`` is captured as a generic ``pytest`` failure so a
    # broken non-live test doesn't go silently green — see
    # ``test_collection_error_without_live_testcase_still_alerts``.
    assert by_connector == ["Google Drive", "Notion", "pytest"]
    notion = next(f for f in failures if f["connector"] == "Notion")
    assert notion["test"] == "test_live_notion_search_pages"
    assert notion["message"] == "assert False"


def test_clean_run_returns_silent_zero(tmp_path, monkeypatch, capsys):
    junit = _write(tmp_path, """<?xml version='1.0'?>
<testsuites><testsuite name='pytest'>
  <testcase name='test_live_slack_list_channels'/>
</testsuite></testsuites>""")
    # Strip any notification env vars that may leak from CI.
    for var in (
        "SLACK_WEBHOOK_URL",
        "ALERT_EMAIL_TO",
        "SMTP_HOST",
        "PAGERDUTY_ROUTING_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    rc = notify.main(["notify_live_failures.py", str(junit)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_failures_exit_nonzero_and_print_summary(tmp_path, monkeypatch, capsys):
    junit = _write(tmp_path, """<?xml version='1.0'?>
<testsuites><testsuite name='pytest'>
  <testcase name='test_live_slack_list_channels'>
    <failure message='401 Unauthorized'/>
  </testcase>
</testsuite></testsuites>""")
    for var in (
        "SLACK_WEBHOOK_URL",
        "ALERT_EMAIL_TO",
        "SMTP_HOST",
        "PAGERDUTY_ROUTING_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    rc = notify.main(["notify_live_failures.py", str(junit)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Slack" in err
    assert "401 Unauthorized" in err


def test_collection_error_without_live_testcase_still_alerts(tmp_path, monkeypatch, capsys):
    """A junit with no matching test_live_* names but a collection/import
    error must still produce a non-zero exit + summary, otherwise CI goes
    falsely green when pytest blows up before the live tests run."""
    junit = _write(tmp_path, """<?xml version='1.0'?>
<testsuites><testsuite name='pytest' errors='1'>
  <testcase classname='tests.test_connectors' name='tests/test_connectors.py'>
    <error message='collection failure: ImportError: no module named foo'/>
  </testcase>
</testsuite></testsuites>""")
    for var in ("SLACK_WEBHOOK_URL", "ALERT_EMAIL_TO", "SMTP_HOST", "PAGERDUTY_ROUTING_KEY"):
        monkeypatch.delenv(var, raising=False)
    rc = notify.main(["notify_live_failures.py", str(junit)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "pytest" in err
    assert "collection failure" in err


def test_suite_level_error_alerts(tmp_path, monkeypatch, capsys):
    junit = _write(tmp_path, """<?xml version='1.0'?>
<testsuites><testsuite name='pytest' errors='1'>
  <error message='internal pytest error: KeyboardInterrupt'/>
</testsuite></testsuites>""")
    for var in ("SLACK_WEBHOOK_URL", "ALERT_EMAIL_TO", "SMTP_HOST", "PAGERDUTY_ROUTING_KEY"):
        monkeypatch.delenv(var, raising=False)
    rc = notify.main(["notify_live_failures.py", str(junit)])
    assert rc == 1
    assert "internal pytest error" in capsys.readouterr().err


def test_pytest_exit_code_forces_alert_even_with_clean_junit(tmp_path, monkeypatch, capsys):
    """If pytest exited non-zero but the junit happens to have no parsed
    failures, we still page — defensive against false-greens."""
    junit = _write(tmp_path, """<?xml version='1.0'?>
<testsuites><testsuite name='pytest'>
  <testcase name='test_live_slack_list_channels'/>
</testsuite></testsuites>""")
    for var in ("SLACK_WEBHOOK_URL", "ALERT_EMAIL_TO", "SMTP_HOST", "PAGERDUTY_ROUTING_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PYTEST_EXIT_CODE", "2")
    rc = notify.main(["notify_live_failures.py", str(junit)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "exit code 2" in err


def test_pytest_exit_code_zero_stays_silent(tmp_path, monkeypatch, capsys):
    junit = _write(tmp_path, """<?xml version='1.0'?>
<testsuites><testsuite name='pytest'>
  <testcase name='test_live_slack_list_channels'/>
</testsuite></testsuites>""")
    for var in ("SLACK_WEBHOOK_URL", "ALERT_EMAIL_TO", "SMTP_HOST", "PAGERDUTY_ROUTING_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PYTEST_EXIT_CODE", "0")
    assert notify.main(["notify_live_failures.py", str(junit)]) == 0
    assert capsys.readouterr().err == ""


def test_missing_junit_is_a_failure(tmp_path, capsys):
    rc = notify.main(["notify_live_failures.py", str(tmp_path / "missing.xml")])
    assert rc == 1
    assert "junit report not found" in capsys.readouterr().err


def test_slack_payload_posted(tmp_path, monkeypatch):
    junit = _write(tmp_path, """<?xml version='1.0'?>
<testsuites><testsuite name='pytest'>
  <testcase name='test_live_github_list_repos'>
    <failure message='rate limited'/>
  </testcase>
</testsuite></testsuites>""")
    sent: dict = {}

    def fake_post(url, summary):
        sent["url"] = url
        sent["summary"] = summary
        return True

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example/abc")
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)
    monkeypatch.delenv("PAGERDUTY_ROUTING_KEY", raising=False)
    monkeypatch.setattr(notify, "_post_slack", fake_post)
    rc = notify.main(["notify_live_failures.py", str(junit)])
    assert rc == 1
    assert sent["url"] == "https://hooks.example/abc"
    assert "GitHub" in sent["summary"]


# ---------------------------------------------------------------------------
# Dedup / state tests
# ---------------------------------------------------------------------------


_FAIL_SLACK_XML = """<?xml version='1.0'?>
<testsuites><testsuite name='pytest'>
  <testcase name='test_live_slack_list_channels'>
    <failure message='401 Unauthorized'/>
  </testcase>
</testsuite></testsuites>"""

_FAIL_SLACK_AND_NOTION_XML = """<?xml version='1.0'?>
<testsuites><testsuite name='pytest'>
  <testcase name='test_live_slack_list_channels'>
    <failure message='401 Unauthorized'/>
  </testcase>
  <testcase name='test_live_notion_search_pages'>
    <failure message='boom'/>
  </testcase>
</testsuite></testsuites>"""

_CLEAN_XML = """<?xml version='1.0'?>
<testsuites><testsuite name='pytest'>
  <testcase name='test_live_slack_list_channels'/>
</testsuite></testsuites>"""


def _strip_channels(monkeypatch):
    for var in (
        "SLACK_WEBHOOK_URL",
        "ALERT_EMAIL_TO",
        "SMTP_HOST",
        "PAGERDUTY_ROUTING_KEY",
        "PYTEST_EXIT_CODE",
    ):
        monkeypatch.delenv(var, raising=False)


def _capture_notifications(monkeypatch):
    """Replace the Slack poster so we can assert exactly which alerts fired."""
    sent: list[dict] = []

    def fake_post(url, summary):
        sent.append({"url": url, "summary": summary})
        return True

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example/abc")
    monkeypatch.setattr(notify, "_post_slack", fake_post)
    return sent


def test_repeat_failure_within_window_is_silent(tmp_path, monkeypatch):
    junit = _write(tmp_path, _FAIL_SLACK_XML)
    state_file = tmp_path / "state.json"
    _strip_channels(monkeypatch)
    sent = _capture_notifications(monkeypatch)

    # First run: alerts.
    assert notify.main(
        ["notify_live_failures.py", str(junit), "--state-file", str(state_file)]
    ) == 1
    assert len(sent) == 1 and "failed" in sent[0]["summary"]

    # Second run, same day, same failure: still exits 1 (CI step stays red)
    # but no new alert is sent.
    assert notify.main(
        ["notify_live_failures.py", str(junit), "--state-file", str(state_file)]
    ) == 1
    assert len(sent) == 1, f"expected no extra alert, got: {sent}"

    state = json.loads(state_file.read_text())
    assert state["connectors"]["Slack"]["status"] == "failing"


def test_reminder_fires_after_window(tmp_path, monkeypatch):
    junit = _write(tmp_path, _FAIL_SLACK_XML)
    state_file = tmp_path / "state.json"
    _strip_channels(monkeypatch)
    sent = _capture_notifications(monkeypatch)

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = iter([base, base + timedelta(days=8)])
    monkeypatch.setattr(notify, "_now", lambda: next(times))

    assert notify.main(
        ["notify_live_failures.py", str(junit),
         "--state-file", str(state_file), "--reminder-days", "7"]
    ) == 1
    assert notify.main(
        ["notify_live_failures.py", str(junit),
         "--state-file", str(state_file), "--reminder-days", "7"]
    ) == 1
    assert len(sent) == 2
    assert "Reminder" in sent[1]["summary"]


def test_recovery_alert_fires_when_connector_goes_green(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    _strip_channels(monkeypatch)
    sent = _capture_notifications(monkeypatch)

    failing = _write(tmp_path, _FAIL_SLACK_XML)
    assert notify.main(
        ["notify_live_failures.py", str(failing), "--state-file", str(state_file)]
    ) == 1
    assert len(sent) == 1

    clean = tmp_path / "clean.xml"
    clean.write_text(_CLEAN_XML)
    rc = notify.main(
        ["notify_live_failures.py", str(clean), "--state-file", str(state_file)]
    )
    assert rc == 0  # pure recovery → green CI step
    assert len(sent) == 2
    assert "recovered" in sent[1]["summary"].lower()
    assert "Slack" in sent[1]["summary"]

    # State should no longer track Slack as failing.
    state = json.loads(state_file.read_text())
    assert "Slack" not in state["connectors"]


def test_new_connector_failing_alongside_known_failure_alerts(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    _strip_channels(monkeypatch)
    sent = _capture_notifications(monkeypatch)

    first = _write(tmp_path, _FAIL_SLACK_XML)
    assert notify.main(
        ["notify_live_failures.py", str(first), "--state-file", str(state_file)]
    ) == 1

    second = tmp_path / "second.xml"
    second.write_text(_FAIL_SLACK_AND_NOTION_XML)
    assert notify.main(
        ["notify_live_failures.py", str(second), "--state-file", str(state_file)]
    ) == 1
    # Two alerts total: initial Slack, then a new one when Notion joins.
    assert len(sent) == 2
    assert "Notion" in sent[1]["summary"]
    # And it's framed as a new failure, not a reminder.
    assert "Reminder" not in sent[1]["summary"]


def test_pagerduty_resolve_sent_on_recovery(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    _strip_channels(monkeypatch)

    events: list[dict] = []

    def fake_pd(routing_key, *, action, dedup_key, summary, custom_details=None):
        events.append({"action": action, "dedup_key": dedup_key})
        return True

    monkeypatch.setenv("PAGERDUTY_ROUTING_KEY", "pd-key")
    monkeypatch.setattr(notify, "_post_pagerduty_event", fake_pd)

    failing = _write(tmp_path, _FAIL_SLACK_XML)
    notify.main(
        ["notify_live_failures.py", str(failing), "--state-file", str(state_file)]
    )
    clean = tmp_path / "clean.xml"
    clean.write_text(_CLEAN_XML)
    notify.main(
        ["notify_live_failures.py", str(clean), "--state-file", str(state_file)]
    )

    actions = [e["action"] for e in events]
    assert actions == ["trigger", "resolve"]
    # Same dedup_key both times so PD closes the right incident.
    assert events[0]["dedup_key"] == events[1]["dedup_key"]
    assert "slack" in events[0]["dedup_key"]


def test_no_state_file_keeps_legacy_every_run_alerts(tmp_path, monkeypatch):
    junit = _write(tmp_path, _FAIL_SLACK_XML)
    _strip_channels(monkeypatch)
    sent = _capture_notifications(monkeypatch)

    # No --state-file and no NOTIFY_STATE_FILE env: every run alerts, as before.
    assert notify.main(["notify_live_failures.py", str(junit)]) == 1
    assert notify.main(["notify_live_failures.py", str(junit)]) == 1
    assert len(sent) == 2
