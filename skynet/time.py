from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

UTC = UTC
DISPLAY_TIMEZONE = "UTC"


def utc_now() -> str:
    """Return an explicit UTC timestamp for durable storage."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def utc_datetime_now() -> datetime:
    """Return an aware UTC datetime for calculations, never for display."""
    return datetime.now(UTC)


def parse_timestamp(value: str) -> datetime:
    """Parse stored ISO timestamps, including legacy ``+00:00`` values."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def display_timestamp(value: str, zone: str | None = None) -> str:
    """Render a timestamp with an explicit local timezone."""
    parsed = parse_timestamp(value)
    zone_name = zone or DISPLAY_TIMEZONE
    local = parsed.astimezone(ZoneInfo(zone_name))
    return f"{local.replace(tzinfo=None).isoformat(timespec='seconds')} [{local.tzname() or zone_name}]"
