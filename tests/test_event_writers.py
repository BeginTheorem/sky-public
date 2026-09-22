"""Structural guard: every protected/deviation kind must have a durable writer.

Diagnostically important kinds used to be written ONLY to the advanced runtime
projection (``state/runtime.jsonl``) while being listed in
``store.PROTECTED_EVENT_KINDS`` / ``metrics.DEVIATION_KINDS``. The protection was
dead: ``fallback_failure`` and ``provider_error`` were "protected" yet had zero
durable rows, so ``skynet logs`` never showed them and retention could not even
be blamed. This test scans the source for a durable ``append_event`` writer for
every such kind, so a new protected kind that is only logged at runtime fails
immediately instead of being discovered months later from a hole in the
post-mortem.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from skynet.metrics import DEVIATION_KINDS
from skynet.store import PROTECTED_EVENT_KINDS

SOURCE = Path(__file__).resolve().parent.parent / "skynet"

# Kinds that are intentionally runtime-only or historical. Each entry must name
# the durable mechanism or the reason there is none; this is the only place a
# protected/deviation kind is allowed to lack a durable writer.
_ALLOWLIST = {
    "fallback_skipped": "high-volume per-strike cooldown skip; deliberately runtime-only so it cannot flood event_log",
    "gate_protected_approved": "retired gate mechanism; historical rows only, no new writer",
    "gate_protected_soft_denied": "retired gate mechanism; historical rows only, no new writer",
    "recovery_reconciliation": "reserved kind; no emitter in current code (the table of the same name is unrelated)",
}

_KIND_CONSTANT = re.compile(r'^([A-Z][A-Z0-9_]*)\s*=\s*["\']([a-z0-9_]+)["\']', re.MULTILINE)


def _durable_kinds() -> set[str]:
    """Every event kind with a source-level durable ``append_event`` writer.

    Recognizes the literal call, the shared ``_emit`` wrapper, module-level kind
    constants passed to ``append_event``, the promoted provider-chain frozenset
    and the planner status->kind mapping table. Deterministic and read-only.
    """
    durable: set[str] = set()
    constants: dict[str, str] = {}
    for path in SOURCE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        durable.update(re.findall(r'append_event\(\s*["\']([a-z0-9_]+)["\']', text))
        durable.update(re.findall(r'\._emit\(\s*["\']([a-z0-9_]+)["\']', text))
        constants.update(_KIND_CONSTANT.findall(text))
    for path in SOURCE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name, kind in constants.items():
            if re.search(r'append_event\(\s*' + re.escape(name) + r"\b", text):
                durable.add(kind)
    store_text = (SOURCE / "store.py").read_text(encoding="utf-8")
    promoted = re.search(r"DURABLE_PROVIDER_EVENT_KINDS\s*=\s*frozenset\(\{([^}]*)\}\)", store_text, re.DOTALL)
    if promoted:
        durable.update(re.findall(r'["\']([a-z0-9_]+)["\']', promoted.group(1)))
    planner_text = (SOURCE / "autonomous_planner.py").read_text(encoding="utf-8")
    mapping = re.search(r"=\s*\{([^}]*)\}\.get\(status,\s*[\"']([a-z0-9_]+)[\"']\)", planner_text)
    if mapping:
        durable.update(re.findall(r'["\']([a-z0-9_]+)["\']', mapping.group(1)))
        durable.add(mapping.group(2))
    return durable


class EventWriterTests(unittest.TestCase):
    def test_every_protected_and_deviation_kind_has_a_durable_writer(self) -> None:
        durable = _durable_kinds()
        required = PROTECTED_EVENT_KINDS | set(DEVIATION_KINDS)
        missing = sorted(kind for kind in required if kind not in durable and kind not in _ALLOWLIST)
        self.assertEqual(
            missing,
            [],
            "protected/deviation kinds without a durable append_event writer "
            "(route them through store.append_event, or add them to _ALLOWLIST with a reason): "
            f"{missing}",
        )

    def test_original_blind_spot_kinds_are_durable(self) -> None:
        # fallback_failure and provider_error were protected with zero durable
        # rows. Guard them explicitly so a refactor cannot quietly move either
        # back to runtime-only without failing this file.
        durable = _durable_kinds()
        self.assertIn("fallback_failure", durable)
        self.assertIn("provider_error", durable)

    def test_allowlist_is_disjoint_from_durable_kinds(self) -> None:
        # A stale allowlist entry would silently keep masking a kind after it
        # gained a writer; force the list to be cleaned up instead.
        durable = _durable_kinds()
        self.assertEqual(sorted(set(_ALLOWLIST) & durable), [])


if __name__ == "__main__":
    unittest.main()
