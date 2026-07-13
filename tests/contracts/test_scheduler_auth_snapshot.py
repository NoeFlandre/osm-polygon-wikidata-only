"""Freeze the shapes of ``RequestSchedulerSnapshot`` and
``WikimediaAuthSnapshot``.

These dataclasses are part of the operator-facing surface (printed in
the sync heartbeat). Adding, removing, or renaming a field is a
breaking change for downstream log scrapers.
"""

from __future__ import annotations

import dataclasses

from osm_polygon_wikidata_only.enrichment.wikimedia_auth import (
    WikimediaAuthSnapshot,
    WikimediaCredentials,
)
from osm_polygon_wikidata_only.utils.request_scheduler import (
    AdaptiveRequestScheduler,
    RequestSchedulerSnapshot,
)


def test_request_scheduler_snapshot_fields() -> None:
    fields = {field.name for field in dataclasses.fields(RequestSchedulerSnapshot)}
    expected = {
        "requests_last_minute",
        "current_requests_per_minute",
        "maximum_requests_per_minute",
        "utilization_percent",
        "in_flight",
        "max_in_flight",
        "throttle_events",
        "throttled_hosts_last_minute",
        "cooling_down_hosts",
        "cooldown_remaining_s",
    }
    assert fields == expected


def test_request_scheduler_snapshot_field_types() -> None:
    types = {f.name: f.type for f in dataclasses.fields(RequestSchedulerSnapshot)}
    assert types["requests_last_minute"] == "int"
    assert types["in_flight"] == "int"
    assert types["max_in_flight"] == "int"
    assert types["cooldown_remaining_s"] == "float"


def test_request_scheduler_snapshot_construction() -> None:
    snapshot = RequestSchedulerSnapshot(
        requests_last_minute=10,
        current_requests_per_minute=900.0,
        maximum_requests_per_minute=1200.0,
        utilization_percent=5.0,
        in_flight=2,
        max_in_flight=8,
        throttle_events=1,
        throttled_hosts_last_minute=1,
        cooling_down_hosts=1,
        cooldown_remaining_s=2.0,
    )
    assert snapshot.max_in_flight == 8


def test_request_scheduler_snapshot_is_hashable() -> None:
    snapshot = RequestSchedulerSnapshot(
        requests_last_minute=0,
        current_requests_per_minute=180.0,
        maximum_requests_per_minute=180.0,
        utilization_percent=0.0,
        in_flight=0,
        max_in_flight=3,
        throttle_events=0,
        throttled_hosts_last_minute=0,
        cooling_down_hosts=0,
        cooldown_remaining_s=0.0,
    )
    # Slots + frozen dataclasses are hashable.
    assert hash(snapshot) == hash(snapshot)


def test_wikimedia_auth_snapshot_fields() -> None:
    fields = {field.name for field in dataclasses.fields(WikimediaAuthSnapshot)}
    assert fields == {
        "credentials_configured",
        "authenticated_hosts",
        "anonymous_hosts",
        "pending_hosts",
    }


def test_wikimedia_auth_snapshot_defaults() -> None:
    snap = WikimediaAuthSnapshot(
        credentials_configured=True,
        authenticated_hosts=2,
        anonymous_hosts=1,
    )
    assert snap.pending_hosts == 0


def test_wikimedia_auth_snapshot_redacts_password_in_repr() -> None:
    creds = WikimediaCredentials(username="bot", password="hunter2")
    text = repr(creds)
    assert "hunter2" not in text
    assert "<redacted>" in text


def test_wikimedia_auth_snapshot_redacts_password_in_repr_explicit() -> None:
    snap = WikimediaAuthSnapshot(
        credentials_configured=True,
        authenticated_hosts=1,
        anonymous_hosts=0,
    )
    # __repr__ is custom; ensure password is never echoed.
    assert "password" not in repr(snap).lower()


def test_scheduler_reports_max_in_flight_property() -> None:
    s = AdaptiveRequestScheduler(max_in_flight=8, requests_per_minute=1200)
    assert s.max_in_flight == 8
