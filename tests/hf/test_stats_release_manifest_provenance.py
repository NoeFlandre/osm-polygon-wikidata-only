"""Processed-manifest validation and provenance for statistics releases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.hf._stats_release.manifest import build_provenance
from osm_polygon_wikidata_only.hf._stats_release.models import StatsReleaseError


def _polygons(processed: Path, stem: str, rows: int) -> Path:
    path = processed / "polygons" / f"{stem}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"id": list(range(rows))}), path)
    return path


def _manifest(processed: Path, payload: object) -> bytes:
    path = processed / "manifests" / "processed_pbfs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    return raw


def _entry(stem: str, rows: int, **extra: object) -> dict[str, object]:
    return {"polygons_path": f"polygons/{stem}.parquet", "row_counts": {"polygons": rows}, **extra}


def _provenance(processed: Path, **kwargs: str | None) -> dict[str, object]:
    return build_provenance(
        processed,
        source_revision=kwargs.get("source_revision"),
        data_revision=kwargs.get("data_revision"),
    )


def test_valid_manifest_produces_full_provenance(tmp_path: Path) -> None:
    _polygons(tmp_path, "alpha-latest", 2)
    _polygons(tmp_path, "beta-latest", 3)
    raw = _manifest(
        tmp_path,
        {
            "source_revision": "src-rev",
            "data": {"revision": "data-rev"},
            "regions": {
                "beta-latest.osm.pbf": _entry("beta-latest", 3, contract_version="v1"),
                "alpha-latest.osm.pbf": {
                    "polygons_path": "polygons/alpha-latest.parquet",
                    "polygon_count": 2,
                    "source_pbf": "alpha.osm.pbf",
                    "contract_version": "v1",
                },
            },
        },
    )

    provenance = _provenance(tmp_path)

    assert provenance["contract_version"] == "v1"
    assert provenance["source_revision"] == "src-rev"
    assert provenance["data_revision"] == "data-rev"
    assert provenance["source_pbf_files"] == ["alpha.osm.pbf", "beta-latest.osm.pbf"]
    assert provenance["manifest"] == {
        "entry_count": 2,
        "path": "manifests/processed_pbfs.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    polygons = provenance["polygons"]
    assert isinstance(polygons, list)
    assert [(row["path"], row["row_count"]) for row in polygons] == [
        ("polygons/alpha-latest.parquet", 2),
        ("polygons/beta-latest.parquet", 3),
    ]


def test_explicit_revisions_override_and_mixed_contracts_are_listed(tmp_path: Path) -> None:
    _polygons(tmp_path, "a", 1)
    _polygons(tmp_path, "b", 1)
    _manifest(
        tmp_path,
        {
            "a.pbf": _entry("a", 1, contract_version="v1"),
            "b.pbf": _entry("b", 1, contract_version="v2"),
        },
    )
    provenance = _provenance(tmp_path, source_revision="s", data_revision="d")
    assert provenance["contract_version"] == ["v1", "v2"]
    assert provenance["source"] == {"pbf_files": ["a.pbf", "b.pbf"], "revision": "s"}
    assert provenance["data"] == {"revision": "d"}


def test_revisions_absent_from_manifest_resolve_to_none(tmp_path: Path) -> None:
    _polygons(tmp_path, "a", 1)
    _manifest(tmp_path, {"regions": {"a.pbf": _entry("a", 1)}, "source": "not-a-mapping"})
    provenance = _provenance(tmp_path)
    assert provenance["source_revision"] is None
    assert provenance["data_revision"] is None
    assert provenance["contract_version"] == []


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(StatsReleaseError, match="complete processed manifest is required"):
        _provenance(tmp_path)


def test_unparseable_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifests" / "processed_pbfs.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StatsReleaseError, match="cannot read processed manifest"):
        _provenance(tmp_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "not an object"),
        ({}, "no region entries"),
        ({"regions": []}, "no region entries"),
        ({"a.pbf": "entry"}, "entry is not an object"),
        ({"a.pbf": {"polygons_path": "elsewhere.parquet"}}, "must point to"),
        (
            {"a.pbf": {"polygons_path": "polygons/a.parquet", "source_pbf": 5}},
            "lacks source_pbf",
        ),
        (
            {"a.pbf": {"polygons_path": "polygons/a.parquet", "polygon_count": True}},
            "lacks polygon row count",
        ),
        (
            {"a.pbf": {"polygons_path": "polygons/a.parquet", "row_counts": {"polygons": "1"}}},
            "lacks polygon row count",
        ),
    ],
)
def test_malformed_manifest_entries_are_rejected(
    tmp_path: Path, payload: object, message: str
) -> None:
    _manifest(tmp_path, payload)
    with pytest.raises(StatsReleaseError, match=message):
        _provenance(tmp_path)


def test_manifest_listed_file_must_exist(tmp_path: Path) -> None:
    _manifest(tmp_path, {"a.pbf": _entry("a", 1)})
    with pytest.raises(StatsReleaseError, match="manifest-listed polygon file is missing"):
        _provenance(tmp_path)


def test_row_count_mismatch_is_rejected(tmp_path: Path) -> None:
    _polygons(tmp_path, "a", 2)
    _manifest(tmp_path, {"a.pbf": _entry("a", 5)})
    with pytest.raises(StatsReleaseError, match="manifest=5, file=2"):
        _provenance(tmp_path)


def test_unreadable_parquet_metadata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "polygons" / "a.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not parquet")
    _manifest(tmp_path, {"a.pbf": _entry("a", 1)})
    with pytest.raises(StatsReleaseError, match="cannot read Parquet metadata"):
        _provenance(tmp_path)


def test_unlisted_polygon_files_are_rejected(tmp_path: Path) -> None:
    _polygons(tmp_path, "a", 1)
    _polygons(tmp_path, "stray", 1)
    _manifest(tmp_path, {"a.pbf": _entry("a", 1)})
    with pytest.raises(StatsReleaseError, match="absent from the manifest: polygons/stray.parquet"):
        _provenance(tmp_path)
