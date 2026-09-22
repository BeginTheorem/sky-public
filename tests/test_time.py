import unittest
from datetime import UTC

from skynet.time import display_timestamp, parse_timestamp, utc_now


class TimeTests(unittest.TestCase):
    def test_utc_now_is_explicit_utc(self) -> None:
        value = utc_now()
        self.assertTrue(value.endswith("Z"))
        self.assertEqual(parse_timestamp(value).tzinfo, UTC)

    def test_display_timestamp_shows_requested_zone(self) -> None:
        rendered = display_timestamp("2026-09-13T22:44:49+00:00", "UTC")
        self.assertEqual(rendered, "2026-09-13T22:44:49 [UTC]")

    def test_display_timestamp_is_explicitly_utc(self) -> None:
        rendered = display_timestamp("2026-09-14T21:25:45Z", "UTC")
        self.assertEqual(rendered, "2026-09-14T21:25:45 [UTC]")
