"""UTC time helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with second precision.

    The format ends in ``Z`` to make it explicit that the time is in
    UTC. This is the canonical timestamp format used throughout the
    pipeline (``extracted_at``, ``retrieved_at``, ``processed_at``).
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
