"""Rejection branches of the durable pending-publication envelope."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.pipeline import pending_publications as pp

HASH = "a" * 64


def _root(tmp_path: Path) -> DataRoot:
    return DataRoot(tmp_path)


def _write(root: DataRoot, payload: object) -> Path:
    path = root.processed_manifests / pp.FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _envelope(**fields: object) -> dict[str, object]:
    return {"contract_version": pp.CONTRACT_VERSION, **fields}


@pytest.mark.parametrize("stem", ["", ".", "..", "a/b", "a\\b"])
def test_unsafe_stems_cannot_be_saved(tmp_path: Path, stem: str) -> None:
    with pytest.raises(ValueError, match="Invalid pending publication stem"):
        pp.save_pending_publications(_root(tmp_path), {stem})


def test_add_and_remove_round_trip_and_ignore_empty_sets(tmp_path: Path) -> None:
    root = _root(tmp_path)
    pp.add_pending_publications(root, set())
    assert pp.load_pending_publications(root) == set()
    pp.add_pending_publications(root, {"b", "a"})
    pp.remove_pending_publications(root, set())
    pp.remove_pending_publications(root, {"a"})
    assert pp.load_pending_publications(root) == {"b"}


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        ([], TypeError, "must be a JSON object"),
        ({"contract_version": "other"}, ValueError, "Invalid contract version"),
        (_envelope(), TypeError, "missing 'stems'"),
        (_envelope(stems="a"), TypeError, "must be a list"),
        (_envelope(stems=[1]), TypeError, "index 0 is not a string"),
        (_envelope(stems=[".."]), ValueError, "Invalid pending publication stem"),
        (_envelope(stems=["a", "a"]), ValueError, "duplicate stems"),
    ],
)
def test_malformed_stem_envelopes_are_rejected(
    tmp_path: Path, payload: object, error: type[Exception], message: str
) -> None:
    root = _root(tmp_path)
    _write(root, payload)
    with pytest.raises(error, match=message):
        pp.load_pending_publications(root)


def test_undecodable_envelope_is_rejected(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _write(root, {}).write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed pending publication manifest"):
        pp.load_pending_publications(root)


@pytest.mark.parametrize(
    ("stems", "hashes", "message"),
    [
        ("a", {"a": HASH}, "stems must be a list"),
        (["a"], [HASH], "fingerprint_hashes must be a dict"),
        ([], {}, "at least one stem"),
        (["a", "a"], {"a": HASH}, "must not contain duplicates"),
        (["a"], {"b": HASH}, "keys must match the stems list"),
        ([1], {1: HASH}, "Marker stem must be a string"),
        ([""], {"": HASH}, "must be non-empty"),
        (["a/b"], {"a/b": HASH}, "path separators"),
        ([".."], {"..": HASH}, "path component"),
        (["a"], {"a": 5}, "hash must be a string"),
        (["a"], {"a": "A" * 64}, "64 lowercase hex"),
    ],
)
def test_invalid_marker_inputs_are_rejected(
    tmp_path: Path, stems: object, hashes: object, message: str
) -> None:
    root = _root(tmp_path)
    with pytest.raises(ValueError, match=message):
        pp.set_metadata_refresh_marker(root, stems, hashes)  # ty: ignore[invalid-argument-type]
    assert not (root.processed_manifests / pp.FILENAME).exists()


def test_marker_lifecycle_preserves_stems_and_skips_identical_rewrites(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert pp.load_metadata_refresh_marker(root) is None
    pp.clear_metadata_refresh_marker(root)
    pp.save_pending_publications(root, {"keep"})
    assert pp.load_metadata_refresh_marker(root) is None
    pp.clear_metadata_refresh_marker(root)

    pp.set_metadata_refresh_marker(root, ["b", "a"], {"a": HASH, "b": HASH})
    path = root.processed_manifests / pp.FILENAME
    before = path.stat().st_mtime_ns
    pp.set_metadata_refresh_marker(root, ["a", "b"], {"b": HASH, "a": HASH})
    assert path.stat().st_mtime_ns == before
    assert pp.load_metadata_refresh_marker(root) == {
        "stems": ["a", "b"],
        "fingerprint_hashes": {"a": HASH, "b": HASH},
    }

    pp.clear_metadata_refresh_marker(root)
    assert pp.load_metadata_refresh_marker(root) is None
    assert pp.load_pending_publications(root) == {"keep"}


def test_marker_on_fresh_root_creates_a_valid_envelope(tmp_path: Path) -> None:
    root = _root(tmp_path)
    pp.set_metadata_refresh_marker(root, ["a"], {"a": HASH})
    assert pp.load_pending_publications(root) == set()


@pytest.mark.parametrize(
    ("marker", "error", "message"),
    [
        ([], TypeError, "must be a JSON object"),
        ({"stems": []}, ValueError, "exactly 'stems' and 'fingerprint_hashes'"),
        ({"stems": "a", "fingerprint_hashes": {}}, TypeError, "stems must be a list"),
        ({"stems": ["a", "a"], "fingerprint_hashes": {}}, ValueError, "duplicates"),
        ({"stems": [1], "fingerprint_hashes": {}}, TypeError, "entries must be strings"),
        ({"stems": ["a/b"], "fingerprint_hashes": {}}, ValueError, "path separators"),
        ({"stems": ["a"], "fingerprint_hashes": []}, TypeError, "must be a dict"),
        ({"stems": ["a"], "fingerprint_hashes": {"b": HASH}}, ValueError, "keys must match"),
        ({"stems": ["a"], "fingerprint_hashes": {"a": "nope"}}, ValueError, "64 lowercase"),
    ],
)
def test_malformed_persisted_markers_are_rejected(
    tmp_path: Path, marker: object, error: type[Exception], message: str
) -> None:
    root = _root(tmp_path)
    _write(root, _envelope(stems=[], metadata_refresh=marker))
    with pytest.raises(error, match=message):
        pp.load_metadata_refresh_marker(root)
