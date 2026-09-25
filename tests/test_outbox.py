import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from skynet.outbox import (
    BACKLOG_SUPPRESS_ERROR,
    OutboxDrainer,
    OutboxRateLimitError,
    TelegramTransport,
    render_agent_response,
    render_outbox_message,
)
from skynet.store import StateStore


class FakeTransport:
    def __init__(self, *, failures=None, fail_always=False, rate_limit_once=None, error=None):
        self.sent = []
        self.calls = 0
        self.failures = set(failures or ())
        self.fail_always = fail_always
        self.rate_limit_once = rate_limit_once
        self.error = error or RuntimeError("transport down")

    def send(self, text):
        self.calls += 1
        if self.rate_limit_once is not None and self.calls == 1:
            raise OutboxRateLimitError(self.rate_limit_once)
        if self.fail_always or self.calls in self.failures:
            raise self.error
        self.sent.append(text)


class FakeHTTP:
    def __init__(self, fail_call=None):
        self.calls = []
        self.fail_call = fail_call

    def call(self, method, params=None):
        self.calls.append((method, params or {}))
        if self.fail_call is not None and len(self.calls) == self.fail_call:
            raise RuntimeError("boom")
        return {"ok": True, "result": True}


def _seed_ordered(store, count, kind="agent_response"):
    ids = []
    for index in range(count):
        message_id = store.add_outbox(kind, {"run_id": f"r{index}", "status": "completed", "report": "{}"})
        store.connection.execute(
            "UPDATE outbox SET created_at=? WHERE message_id=?",
            (f"2020-01-01T00:00:{index:02d}Z", message_id),
        )
        ids.append(message_id)
    return ids


class OutboxDrainerTests(unittest.TestCase):
    def test_agent_response_is_rendered_delivered_and_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            report = json.dumps({"status": "COMPLETED", "summary": "did the thing"})
            store.add_outbox("agent_response", {"run_id": "r1", "status": "completed", "report": report})
            transport = FakeTransport()
            drainer = OutboxDrainer(store, transport, backlog_limit=20)

            delivered = drainer.drain()

            self.assertEqual(delivered, 1)
            self.assertIn("r1", transport.sent[0])
            self.assertIn("did the thing", transport.sent[0])
            self.assertEqual(store.pending_outbox(), [])
            row = store.connection.execute("SELECT delivery_state FROM outbox").fetchone()
            self.assertEqual(row[0], "delivered")
            store.close()

    def test_dead_letter_after_cap_emits_event_and_critical_alert(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_outbox("agent_response", {"run_id": "r1", "status": "failed", "report": "{}"})
            transport = FakeTransport(failures={1, 2, 3, 4, 5})
            drainer = OutboxDrainer(store, transport, batch=1, max_attempts=5, backlog_limit=20)

            for _ in range(5):
                drainer.drain()

            row = store.connection.execute(
                "SELECT delivery_state FROM outbox WHERE kind='agent_response'"
            ).fetchone()
            self.assertEqual(row[0], "dead")
            events = store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE kind='outbox_dead_letter'"
            ).fetchone()[0]
            self.assertEqual(events, 1)
            alerts = store.pending_alerts()
            self.assertTrue(any(item["kind"] == "outbox_dead_letter" and item["severity"] == "critical" for item in alerts))
            store.close()

    def test_chunk_failure_marks_the_whole_message_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            api = FakeHTTP(fail_call=2)
            transport = TelegramTransport(api, 8)
            message_id = store.add_outbox("custom", {"blob": "z" * 6000})
            drainer = OutboxDrainer(store, transport, backlog_limit=20)

            drainer.drain()

            row = store.connection.execute(
                "SELECT delivery_state, last_error FROM outbox WHERE message_id=?", (message_id,)
            ).fetchone()
            self.assertEqual(row["delivery_state"], "pending")
            self.assertIn("boom", row["last_error"])
            self.assertEqual(len(api.calls), 2)
            store.close()

    def test_alert_delivery_marks_alert_delivered_with_telegram_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            result = store.raise_alert("provider_lockout", {"providers": ["openrouter"]}, severity="critical")
            transport = FakeTransport()
            drainer = OutboxDrainer(store, transport, backlog_limit=20)

            drainer.drain()

            row = store.connection.execute(
                "SELECT delivered_at, channel FROM alerts WHERE alert_id=?", (result["alert_id"],)
            ).fetchone()
            self.assertIsNotNone(row["delivered_at"])
            self.assertEqual(row["channel"], "telegram")
            store.close()

    def test_backlog_suppression_kills_oldest_alerts_once_and_delivers_new(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            _seed_ordered(store, 30)
            transport = FakeTransport()
            drainer = OutboxDrainer(store, transport, batch=5, backlog_limit=20)

            drainer.drain()

            dead = store.connection.execute("SELECT COUNT(*) FROM outbox WHERE delivery_state='dead'").fetchone()[0]
            self.assertEqual(dead, 10)
            suppressed = store.connection.execute(
                "SELECT COUNT(*) FROM outbox WHERE last_error=?", (BACKLOG_SUPPRESS_ERROR,)
            ).fetchone()[0]
            self.assertEqual(suppressed, 10)
            alerts = store.connection.execute("SELECT COUNT(*) FROM alerts WHERE kind='outbox_backlog'").fetchone()[0]
            self.assertEqual(alerts, 1)
            raised = store.connection.execute("SELECT COUNT(*) FROM event_log WHERE kind='alert_raised'").fetchone()[0]
            self.assertEqual(raised, 1)
            # Suppression is a visible fact, not silence: the organism can read
            # "I was muted" back out of its own event log.
            muted = store.connection.execute(
                "SELECT payload FROM event_log WHERE kind='outbox_backlog_suppressed'"
            ).fetchone()
            self.assertIsNotNone(muted)
            self.assertEqual(json.loads(muted["payload"]), {"count": 10, "limit": 20})

            for _ in range(10):
                drainer.drain()
            self.assertEqual(store.pending_outbox(), [])
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM outbox WHERE delivery_state='dead'").fetchone()[0],
                10,
            )

            new_id = store.add_outbox("agent_response", {"run_id": "new", "status": "completed", "report": "{}"})
            drainer.drain()
            row = store.connection.execute(
                "SELECT delivery_state FROM outbox WHERE message_id=?", (new_id,)
            ).fetchone()
            self.assertEqual(row[0], "delivered")
            store.close()

    def test_rate_limit_sleeps_bounded_and_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_outbox("agent_response", {"run_id": "r1", "status": "completed", "report": "{}"})
            transport = FakeTransport(rate_limit_once=42.0)
            sleeps = []
            drainer = OutboxDrainer(store, transport, sleep=sleeps.append, rate_limit_cap=60.0, backlog_limit=20)

            delivered = drainer.drain()

            self.assertEqual(delivered, 1)
            self.assertEqual(len(sleeps), 1)
            self.assertEqual(sleeps[0], 42.0)
            store.close()

    def test_disabled_drainer_delivers_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StateStore(Path(directory) / "state.sqlite3")
            store.add_outbox("agent_response", {"run_id": "r1", "status": "completed", "report": "{}"})
            transport = FakeTransport()
            drainer = OutboxDrainer(store, transport, enabled=False)

            self.assertEqual(drainer.drain(), 0)
            self.assertEqual(transport.sent, [])
            self.assertEqual(len(store.pending_outbox()), 1)
            store.close()

    def test_render_agent_response_falls_back_to_raw_report(self):
        self.assertIn("plain summary", render_agent_response({"run_id": "r", "status": "s", "report": "plain summary"}))
        self.assertIn("from json", render_outbox_message("agent_response", {"run_id": "r", "status": "s", "report": json.dumps({"summary": "from json"})}))
        self.assertIn("weird", render_outbox_message("weird", {"a": 1}))

    def test_agent_response_is_compact_and_points_to_logs(self):
        huge = "x" * 12000
        rendered = render_agent_response({"run_id": "r9", "status": "completed", "report": json.dumps({"summary": huge})})
        self.assertLessEqual(len(rendered), 600)
        self.assertIn("r9", rendered)
        self.assertIn("completed", rendered)
        self.assertIn("/logs", rendered)
        self.assertNotIn(huge, rendered)
        raw = render_agent_response({"run_id": "r10", "status": "failed", "report": huge})
        self.assertLessEqual(len(raw), 600)
        self.assertIn("/logs", raw)


# A crash between the `delivering` write and the terminal write is the one window
# in which the outbox can lose a verified message. The two child processes below
# reproduce exactly that window: the first claims the row and exits with
# ``os._exit`` before ``mark_delivered``/``mark_failed``, the second is a fresh
# process whose first writable open of the same file is startup recovery.
SEEDER = """
import json, os, sys
from datetime import timedelta
from pathlib import Path
from skynet.store import StateStore
from skynet.time import utc_datetime_now

db, mode = Path(sys.argv[1]), sys.argv[2]
store = StateStore(db)
store.add_outbox("agent_response", {"run_id": "stranded-" + mode, "status": "completed",
                                    "report": json.dumps({"summary": "stranded-" + mode})})
claimed = store.claim_outbox(limit=5)
if len(claimed) != 1:
    raise SystemExit("expected exactly one claim, got %r" % (claimed,))
message_id = claimed[0]["message_id"]
if mode == "expired":
    stale = (utc_datetime_now() - timedelta(seconds=3600)).isoformat().replace("+00:00", "Z")
    store.connection.execute("UPDATE outbox SET claimed_at=? WHERE message_id=?", (stale, message_id))
elif mode == "null":
    store.connection.execute("UPDATE outbox SET claimed_at=NULL WHERE message_id=?", (message_id,))
store.connection.commit()
print("seeded " + message_id, flush=True)
os._exit(0)
"""

RESTARTER = """
import json, sys
from datetime import timedelta
from pathlib import Path
from skynet.outbox import OutboxDrainer
from skynet.store import StateStore
from skynet.time import utc_datetime_now

class Recorder:
    def __init__(self):
        self.sent = []
    def send(self, text):
        self.sent.append(text)

db, mode = Path(sys.argv[1]), sys.argv[2]
store = StateStore(db)
def snapshot():
    return dict(store.connection.execute(
        "SELECT delivery_state, claimed_at, attempts, delivered_at FROM outbox").fetchone())
after_open = snapshot()
transport = Recorder()
drainer = OutboxDrainer(store, transport, batch=5, backlog_limit=0)
drains = [drainer.drain(), drainer.drain()]
sends_before_late = len(transport.sent)
late_drains = None
if mode == "fresh" and snapshot()["delivery_state"] == "delivering":
    stale = (utc_datetime_now() - timedelta(seconds=3600)).isoformat().replace("+00:00", "Z")
    store.connection.execute("UPDATE outbox SET claimed_at=?", (stale,))
    store.connection.commit()
    late_drains = [drainer.drain(), drainer.drain()]
print(json.dumps({"after_open": after_open, "drains": drains,
                  "sends_before_late": sends_before_late,
                  "late_drains": late_drains, "after": snapshot(),
                  "total_sends": len(transport.sent)}))
store.close()
"""


class OutboxLeaseStrandingTests(unittest.TestCase):
    """The verdict for a row a dead sender left in `delivering`.

    Measured with this probe: an expired lease is returned to
    `pending` by startup recovery and delivered exactly once; a lease inside the
    window is left alone (no reset, no send) and is picked up once the window
    passes; a NULL `claimed_at` is recovered too. Nothing is stranded.
    """

    ROOT = Path(__file__).resolve().parent.parent

    def _run(self, db, mode, snippet):
        child_environment = os.environ.copy()
        child_environment["PYTHONPATH"] = str(self.ROOT)
        return subprocess.run(
            [sys.executable, "-c", snippet, str(db), mode],
            cwd=str(self.ROOT), env=child_environment, capture_output=True, text=True, timeout=120,
        )

    def _crash_with_lease(self, db, mode):
        result = self._run(db, mode, SEEDER)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("seeded", result.stdout)

    def _restart_and_drain(self, db, mode):
        result = self._run(db, mode, RESTARTER)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_expired_lease_after_a_dead_sender_is_delivered_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "state.sqlite3"
            self._crash_with_lease(db, "expired")
            verdict = self._restart_and_drain(db, "expired")
            self.assertEqual(verdict["after_open"]["delivery_state"], "pending")
            self.assertEqual(verdict["drains"], [1, 0])
            self.assertEqual(verdict["after"]["delivery_state"], "delivered")
            self.assertEqual(verdict["total_sends"], 1)

    def test_live_lease_is_not_reset_and_is_not_stranded(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "state.sqlite3"
            self._crash_with_lease(db, "fresh")
            verdict = self._restart_and_drain(db, "fresh")
            self.assertEqual(verdict["after_open"]["delivery_state"], "delivering")
            self.assertEqual(verdict["drains"], [0, 0])
            self.assertEqual(verdict["sends_before_late"], 0)
            self.assertEqual(verdict["late_drains"], [1, 0])
            self.assertEqual(verdict["after"]["delivery_state"], "delivered")
            self.assertEqual(verdict["total_sends"], 1)

    def test_null_claimed_at_lease_is_reclaimed_at_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "state.sqlite3"
            self._crash_with_lease(db, "null")
            verdict = self._restart_and_drain(db, "null")
            self.assertEqual(verdict["after_open"]["delivery_state"], "pending")
            self.assertEqual(verdict["drains"], [1, 0])
            self.assertEqual(verdict["total_sends"], 1)
            self.assertEqual(verdict["after"]["delivery_state"], "delivered")


class AlertScriptTests(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "alert.sh"

    def _fake_curl_env(self, tmp, extra):
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        record = tmp / "argv.txt"
        curl = bin_dir / "curl"
        curl.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >> \"$ALERT_ARGV_RECORD\"\nexit 0\n", encoding="utf-8")
        curl.chmod(0o755)
        env = dict(extra)
        env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
        env["ALERT_ARGV_RECORD"] = str(record)
        return env, record

    def test_missing_token_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            env, _ = self._fake_curl_env(Path(directory), {})
            result = subprocess.run(["bash", str(self.SCRIPT), "skynet.service", "unit failed"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn("skipped", result.stderr)

    def test_curl_failure_still_exits_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory) / "bin"
            bin_dir.mkdir()
            curl = bin_dir / "curl"
            curl.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
            curl.chmod(0o755)
            env = {
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "SKYNET_TELEGRAM_BOT_TOKEN": "123:secret",
                "SKYNET_TELEGRAM_CHAT_ID": "42",
            }
            result = subprocess.run(["bash", str(self.SCRIPT), "skynet.service", "unit failed"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)

    def test_canonical_allowed_chat_id_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            env, record = self._fake_curl_env(
                Path(directory),
                {"SKYNET_TELEGRAM_BOT_TOKEN": "123:secret", "SKYNET_TELEGRAM_ALLOWED_CHAT_ID": "42"},
            )
            result = subprocess.run(["bash", str(self.SCRIPT), "skynet.service", "unit failed"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertNotIn("skipped", result.stderr)
            self.assertTrue(record.exists(), "the canonical chat-id key must reach curl")

    def test_token_never_appears_in_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            env, record = self._fake_curl_env(
                Path(directory),
                {"SKYNET_TELEGRAM_BOT_TOKEN": "123456:SUPERSECRETTOKEN", "SKYNET_TELEGRAM_CHAT_ID": "42"},
            )
            result = subprocess.run(["bash", str(self.SCRIPT), "skynet.service", "unit failed"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            argv = record.read_text(encoding="utf-8") if record.exists() else ""
            self.assertNotIn("SUPERSECRETTOKEN", argv)

    def test_script_has_no_home_default_and_parses(self):
        text = self.SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("$HOME", text)
        result = subprocess.run(["bash", "-n", str(self.SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)


class SystemdUnitsTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent

    def test_failure_alert_is_wired_and_oneshot(self):
        for name in ("skynet.service.in", "skynet-telegram.service.in"):
            text = (self.ROOT / "deploy" / name).read_text(encoding="utf-8")
            self.assertIn("OnFailure=skynet-alert@%n.service", text)
        alert = (self.ROOT / "deploy" / "skynet-alert@.service.in").read_text(encoding="utf-8")
        self.assertIn("Type=oneshot", alert)
        self.assertIn("scripts/alert.sh %i", alert)


if __name__ == "__main__":
    unittest.main()
