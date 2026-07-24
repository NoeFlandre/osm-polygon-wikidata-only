"""Phase 2 / Group E: metadata-refresh marker lifecycle.

Red tests for the ``metadata_refresh`` field added to the existing
``pending_migration_publications.json`` envelope. The marker represents
"global metadata may now be stale" and is written after a local regional
transaction commits, before its upload is enqueued.

Strict validation: stems and fingerprint_hashes must share the same key
set, every hash must be a 64-char lowercase SHA-256, every stem must be
safe, and duplicate stems are rejected.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _marker_helpers():
    """Import the new marker helpers (must exist after Phase 2 implementation)."""
    try:
        from osm_polygon_wikidata_only.pipeline import pending_publications as mod
    except ImportError as exc:
        pytest.fail(f"pending_publications import failed: {exc}")
    for name in (
        "load_metadata_refresh_marker",
        "set_metadata_refresh_marker",
        "clear_metadata_refresh_marker",
    ):
        if not hasattr(mod, name):
            pytest.fail(f"pending_publications.{name} must exist (Phase 2 Group E metadata marker)")
    return mod


def _data_root(tmp_path: Path):
    class _DR:
        def __init__(self, p: Path) -> None:
            self.processed = p
            self.processed_manifests = p / "manifests"

    dr = _DR(tmp_path)
    dr.processed_manifests.mkdir(parents=True, exist_ok=True)
    return dr


def _valid_sha(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Payload: no timestamps
# ---------------------------------------------------------------------------


def test_marker_payload_has_no_timestamp(tmp_path: Path) -> None:
    """The marker must not contain any timestamp field."""
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    set_payload = {
        "monaco-latest": _valid_sha("monaco"),
        "italy-latest": _valid_sha("italy"),
    }
    mod.set_metadata_refresh_marker(
        dr,
        stems=["italy-latest", "monaco-latest"],
        fingerprint_hashes=set_payload,
    )
    raw = json.loads((dr.processed_manifests / "pending_migration_publications.json").read_text())
    assert "metadata_refresh" in raw
    mr = raw["metadata_refresh"]
    for forbidden in ("requested_at", "created_at", "updated_at", "timestamp"):
        assert forbidden not in mr, (
            f"Marker payload must not contain timestamp field {forbidden!r}; got {mr}"
        )


def test_marker_serializes_byte_for_byte_for_identical_state(tmp_path: Path) -> None:
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    set_payload = {
        "monaco-latest": _valid_sha("monaco"),
        "italy-latest": _valid_sha("italy"),
    }
    mod.set_metadata_refresh_marker(
        dr,
        stems=["italy-latest", "monaco-latest"],
        fingerprint_hashes=set_payload,
    )
    raw1 = (dr.processed_manifests / "pending_migration_publications.json").read_bytes()
    mod.set_metadata_refresh_marker(
        dr,
        stems=["italy-latest", "monaco-latest"],
        fingerprint_hashes=set_payload,
    )
    raw2 = (dr.processed_manifests / "pending_migration_publications.json").read_bytes()
    assert raw1 == raw2, (
        "Identical marker state must serialize byte-for-byte identically; "
        "found a non-deterministic field (timestamps, dict ordering, etc.)"
    )


def test_marker_orders_stems_and_hashes_deterministically(tmp_path: Path) -> None:
    """Stems and hashes must be sorted identically."""
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    set_payload = {
        "monaco-latest": _valid_sha("monaco"),
        "italy-latest": _valid_sha("italy"),
    }
    mod.set_metadata_refresh_marker(
        dr,
        stems=["monaco-latest", "italy-latest"],
        fingerprint_hashes=set_payload,
    )
    raw = json.loads((dr.processed_manifests / "pending_migration_publications.json").read_text())
    mr = raw["metadata_refresh"]
    assert mr["stems"] == ["italy-latest", "monaco-latest"], (
        f"Stems must be sorted, got {mr['stems']}"
    )
    assert list(mr["fingerprint_hashes"].keys()) == ["italy-latest", "monaco-latest"], (
        f"Hash keys must match stems order, got {list(mr['fingerprint_hashes'].keys())}"
    )


# ---------------------------------------------------------------------------
# Strict validation
# ---------------------------------------------------------------------------


def test_marker_rejects_duplicate_stems(tmp_path: Path) -> None:
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    with pytest.raises(ValueError):
        mod.set_metadata_refresh_marker(
            dr,
            stems=["monaco-latest", "monaco-latest"],
            fingerprint_hashes={
                "monaco-latest": _valid_sha("monaco"),
            },
        )


def test_marker_rejects_missing_hash_for_stem(tmp_path: Path) -> None:
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    with pytest.raises(ValueError):
        mod.set_metadata_refresh_marker(
            dr,
            stems=["monaco-latest", "italy-latest"],
            fingerprint_hashes={
                "monaco-latest": _valid_sha("monaco"),
            },
        )


def test_marker_rejects_extra_hash_for_unknown_stem(tmp_path: Path) -> None:
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    with pytest.raises(ValueError):
        mod.set_metadata_refresh_marker(
            dr,
            stems=["monaco-latest"],
            fingerprint_hashes={
                "monaco-latest": _valid_sha("monaco"),
                "italy-latest": _valid_sha("italy"),
            },
        )


def test_marker_rejects_malformed_hash_not_64_lowercase_hex(tmp_path: Path) -> None:
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    with pytest.raises(ValueError):
        mod.set_metadata_refresh_marker(
            dr,
            stems=["monaco-latest"],
            fingerprint_hashes={"monaco-latest": "aaaa"},
        )


def test_marker_rejects_unsafe_traversal_stem(tmp_path: Path) -> None:
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    with pytest.raises(ValueError):
        mod.set_metadata_refresh_marker(
            dr,
            stems=["../escape"],
            fingerprint_hashes={"../escape": _valid_sha("escape")},
        )


def test_marker_rejects_empty_stem(tmp_path: Path) -> None:
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    with pytest.raises(ValueError):
        mod.set_metadata_refresh_marker(
            dr,
            stems=[""],
            fingerprint_hashes={"": _valid_sha("")},
        )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_marker_persists_through_independent_load(tmp_path: Path) -> None:
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    set_payload = {
        "monaco-latest": _valid_sha("monaco"),
        "italy-latest": _valid_sha("italy"),
    }
    mod.set_metadata_refresh_marker(
        dr,
        stems=["italy-latest", "monaco-latest"],
        fingerprint_hashes=set_payload,
    )
    loaded = mod.load_metadata_refresh_marker(dr)
    assert loaded is not None
    assert loaded["stems"] == ["italy-latest", "monaco-latest"]
    assert loaded["fingerprint_hashes"] == set_payload


def test_marker_clear_removes_only_marker_not_stems(tmp_path: Path) -> None:
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    from osm_polygon_wikidata_only.pipeline import pending_publications as pending_mod

    pending_mod.add_pending_publications(dr, {"alpha", "beta"})
    mod.set_metadata_refresh_marker(
        dr,
        stems=["alpha"],
        fingerprint_hashes={"alpha": _valid_sha("alpha")},
    )
    mod.clear_metadata_refresh_marker(dr)
    raw = json.loads((dr.processed_manifests / "pending_migration_publications.json").read_text())
    assert "metadata_refresh" not in raw, "Clear must remove the metadata_refresh field"
    assert sorted(raw["stems"]) == ["alpha", "beta"], (
        "Clear must NOT remove the original stems field"
    )


def test_marker_load_returns_none_when_absent(tmp_path: Path) -> None:
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    assert mod.load_metadata_refresh_marker(dr) is None


def test_marker_rejects_corrupt_marker_with_valid_stems_field(tmp_path: Path) -> None:
    """A marquee with a malformed metadata_refresh must fail validation,
    but the corruption check must NOT touch the existing stems field."""
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    (dr.processed_manifests / "pending_migration_publications.json").write_text(
        json.dumps(
            {
                "contract_version": "pending-publications-v1",
                "stems": ["alpha", "beta"],
                "metadata_refresh": "not an object",
            }
        )
    )
    with pytest.raises((ValueError, TypeError)):
        mod.load_metadata_refresh_marker(dr)
    # The existing stems field must remain reachable through the
    # existing stems-only API.
    from osm_polygon_wikidata_only.pipeline import pending_publications as pending_mod

    assert pending_mod.load_pending_publications(dr) == {"alpha", "beta"}


def test_no_op_marker_update_preserves_hash_and_mtime(tmp_path: Path) -> None:
    """A redundant set with the same input must not rewrite the file."""
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    set_payload = {
        "monaco-latest": _valid_sha("monaco"),
        "italy-latest": _valid_sha("italy"),
    }
    mod.set_metadata_refresh_marker(
        dr,
        stems=["italy-latest", "monaco-latest"],
        fingerprint_hashes=set_payload,
    )
    marker_path = dr.processed_manifests / "pending_migration_publications.json"
    first_hash = hashlib.sha256(marker_path.read_bytes()).hexdigest()
    import os

    older = marker_path.stat().st_mtime_ns - 1_000_000
    os.utime(marker_path, ns=(older, older))
    mod.set_metadata_refresh_marker(
        dr,
        stems=["italy-latest", "monaco-latest"],
        fingerprint_hashes=set_payload,
    )
    second_hash = hashlib.sha256(marker_path.read_bytes()).hexdigest()
    assert first_hash == second_hash, "No-op marker update must preserve hash"
    assert marker_path.stat().st_mtime_ns == older, (
        "No-op marker update must NOT rewrite the file (mtime preserved)"
    )


def test_marker_rejects_path_separator_in_stem(tmp_path: Path) -> None:
    """Stem must not contain path separators."""
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    with pytest.raises(ValueError):
        mod.set_metadata_refresh_marker(
            dr,
            stems=["alpha/beta"],
            fingerprint_hashes={"alpha/beta": _valid_sha("ab")},
        )


def test_marker_rejects_mixed_case_in_hash(tmp_path: Path) -> None:
    """Hash must be lowercase hex (no mixed-case)."""
    mod = _marker_helpers()
    dr = _data_root(tmp_path)
    lower = _valid_sha("monaco")
    upper = lower.upper()
    with pytest.raises(ValueError):
        mod.set_metadata_refresh_marker(
            dr,
            stems=["monaco-latest"],
            fingerprint_hashes={"monaco-latest": upper},
        )
