
import unittest
from datetime import UTC

from skynet.time import display_timestamp, parse_timestamp, utc_now


def _iso(seconds: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()

class TimeTests(unittest.TestCase):
    def test_utc_now_is_explicit_utc(self) -> None:
        value = utc_now()
        self.assertTrue(value.endswith("Z"))
        self.assertEqual(parse_timestamp(value).tzinfo, UTC)

    def test_display_timestamp_shows_requested_zone(self) -> None:
        rendered = display_timestamp(_iso(81889.0), "UTC")
        self.assertEqual(rendered, _iso(81889.0)[:19] + " [UTC]")

    def test_display_timestamp_is_explicitly_utc(self) -> None:
        rendered = display_timestamp(_iso(163545.0), "UTC")
        self.assertEqual(rendered, _iso(163545.0)[:19] + " [UTC]")
