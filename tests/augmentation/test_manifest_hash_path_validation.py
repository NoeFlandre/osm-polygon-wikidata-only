"""Tests verifying that ``augmentation_is_current`` validates ``core_hashes`` strictly.

For each stem the manifest must contain exactly two entries:

* ``processed/polygons/<stem>.parquet``
* exactly one of:
  - ``processed/articles/<stem>.parquet``
  - ``processed/wikipedia/documents/<stem>.parquet``

Arbitrary external paths, missing entries, extra entries, malformed
values, wrong-stem paths, and traversal-like paths are rejected by
returning ``False`` (not by raising).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.orchestrator import augmentation_is_current
from osm_polygon_wikidata_only.augmentation.schema import (
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.steps import (
    sha256_file,
    update_augmentation_manifest,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_document_migration import (
    apply_migration,
    plan_migration,
)
from osm_polygon_wikidata_only.config.paths import DataRoot

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "processed"
STEM = "monaco-latest"


def _ensure_migrated_canonical(data_root: DataRoot) -> None:
    apply_migration(plan_migration(data_root.processed, stems={STEM}))


def _ensure_all_sidecars(data_root: DataRoot) -> None:
    for relative, schema in (
        (f"wikivoyage/documents/{STEM}.parquet", document_schema()),
        (f"wikivoyage/sections/{STEM}.parquet", section_schema()),
        (f"wikidata/facts/{STEM}.parquet", fact_schema()),
    ):
        path = data_root.processed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([], schema=schema), path)


def _seed_data_root(tmp_path: Path) -> DataRoot:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    for relative in (
        f"articles/{STEM}.parquet",
        f"polygon_articles/{STEM}.parquet",
        f"polygons/{STEM}.parquet",
        f"wikipedia/documents/{STEM}.parquet",
        f"wikipedia/sections/{STEM}.parquet",
    ):
        src = FIXTURES / relative
        dest = data_root.processed / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    _ensure_migrated_canonical(data_root)
    _ensure_all_sidecars(data_root)
    return data_root


def _write_manifest_with_hashes(data_root: DataRoot, core_hashes: object) -> None:
    update_augmentation_manifest(
        data_root,
        stem=STEM,
        paths=(
            data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet",
            data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet",
            data_root.processed / "wikivoyage" / "documents" / f"{STEM}.parquet",
            data_root.processed / "wikivoyage" / "sections" / f"{STEM}.parquet",
            data_root.processed / "wikidata" / "facts" / f"{STEM}.parquet",
        ),
        core_hashes=core_hashes,  # type: ignore[arg-type]
        counts={
            "wikipedia_documents": 1,
            "wikipedia_sections": 0,
            "wikivoyage_documents": 0,
            "wikivoyage_sections": 0,
            "wikidata_facts": 0,
        },
        completed_at="2026-07-15T00:00:00Z",
    )


def _valid_canonical_hashes(data_root: DataRoot) -> dict[str, str]:
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    polygon = data_root.processed_polygons / f"{STEM}.parquet"
    return {
        str(canonical): sha256_file(canonical),
        str(polygon): sha256_file(polygon),
    }


def _valid_legacy_hashes(data_root: DataRoot) -> dict[str, str]:
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    polygon = data_root.processed_polygons / f"{STEM}.parquet"
    return {
        str(legacy): sha256_file(legacy),
        str(polygon): sha256_file(polygon),
    }


# ---------------------------------------------------------------------------
# Positive cases — must still return True
# ---------------------------------------------------------------------------


def test_canonical_hashes_validate(tmp_path: Path) -> None:
    data_root = _seed_data_root(tmp_path)
    _write_manifest_with_hashes(data_root, _valid_canonical_hashes(data_root))
    assert augmentation_is_current(data_root, STEM)


def test_legacy_hashes_validate(tmp_path: Path) -> None:
    data_root = _seed_data_root(tmp_path)
    _write_manifest_with_hashes(data_root, _valid_legacy_hashes(data_root))
    assert augmentation_is_current(data_root, STEM)


# ---------------------------------------------------------------------------
# Negative cases — must return False without raising
# ---------------------------------------------------------------------------


def test_arbitrary_external_path_rejected(tmp_path: Path) -> None:
    data_root = _seed_data_root(tmp_path)
    polygon = data_root.processed_polygons / f"{STEM}.parquet"
    bogus = Path("/etc/passwd")
    _write_manifest_with_hashes(
        data_root,
        {str(polygon): sha256_file(polygon), str(bogus): "0" * 64},
    )
    assert not augmentation_is_current(data_root, STEM)


def test_polygon_only_hashes_rejected(tmp_path: Path) -> None:
    data_root = _seed_data_root(tmp_path)
    polygon = data_root.processed_polygons / f"{STEM}.parquet"
    _write_manifest_with_hashes(data_root, {str(polygon): sha256_file(polygon)})
    assert not augmentation_is_current(data_root, STEM)


def test_extra_third_path_rejected(tmp_path: Path) -> None:
    data_root = _seed_data_root(tmp_path)
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    polygon = data_root.processed_polygons / f"{STEM}.parquet"
    bogus = data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet"
    _write_manifest_with_hashes(
        data_root,
        {
            str(canonical): sha256_file(canonical),
            str(polygon): sha256_file(polygon),
            str(bogus): sha256_file(bogus),
        },
    )
    assert not augmentation_is_current(data_root, STEM)


def test_wrong_stem_wikipedia_path_rejected(tmp_path: Path) -> None:
    data_root = _seed_data_root(tmp_path)
    wrong_stem_doc = data_root.processed / "wikipedia" / "documents" / "other-region.parquet"
    wrong_stem_doc.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([], schema=document_schema()), wrong_stem_doc)
    polygon = data_root.processed_polygons / f"{STEM}.parquet"
    _write_manifest_with_hashes(
        data_root,
        {str(wrong_stem_doc): sha256_file(wrong_stem_doc), str(polygon): sha256_file(polygon)},
    )
    assert not augmentation_is_current(data_root, STEM)


def test_wrong_stem_articles_path_rejected(tmp_path: Path) -> None:
    data_root = _seed_data_root(tmp_path)
    wrong_legacy = data_root.processed_articles / "other-region.parquet"
    shutil.copy2(FIXTURES / f"articles/{STEM}.parquet", wrong_legacy)
    polygon = data_root.processed_polygons / f"{STEM}.parquet"
    _write_manifest_with_hashes(
        data_root,
        {str(wrong_legacy): sha256_file(wrong_legacy), str(polygon): sha256_file(polygon)},
    )
    assert not augmentation_is_current(data_root, STEM)


def test_traversal_like_path_rejected(tmp_path: Path) -> None:
    data_root = _seed_data_root(tmp_path)
    polygon = data_root.processed_polygons / f"{STEM}.parquet"
    traversal = data_root.processed / "articles" / ".." / f"{STEM}.parquet"
    _write_manifest_with_hashes(
        data_root,
        {str(traversal): sha256_file(polygon), str(polygon): sha256_file(polygon)},
    )
    assert not augmentation_is_current(data_root, STEM)


def test_both_legacy_and_canonical_rejected(tmp_path: Path) -> None:
    data_root = _seed_data_root(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    canonical = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    polygon = data_root.processed_polygons / f"{STEM}.parquet"
    _write_manifest_with_hashes(
        data_root,
        {
            str(legacy): sha256_file(legacy),
            str(canonical): sha256_file(canonical),
            str(polygon): sha256_file(polygon),
        },
    )
    assert not augmentation_is_current(data_root, STEM)


# ---------------------------------------------------------------------------
# Malformed values — must return False without raising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_hashes",
    [
        None,
        [],
        "",
        42,
        {f"processed/polygons/{STEM}.parquet": None},
        {f"processed/polygons/{STEM}.parquet": 12345},
        {f"processed/polygons/{STEM}.parquet": ""},
        {f"processed/polygons/{STEM}.parquet": "not-a-hash"},
        {123: "a" * 64},
        {f"processed/polygons/{STEM}.parquet": "a" * 63},
        {f"processed/polygons/{STEM}.parquet": "a" * 65},
        {f"processed/polygons/{STEM}.parquet": "g" * 64},
        {f"processed/articles/{STEM}.parquet": "a" * 64},
    ],
)
def test_malformed_core_hashes_return_false(tmp_path: Path, bad_hashes: object) -> None:
    """Write the manifest directly so non-JSON-serialisable values are exercised."""
    data_root = _seed_data_root(tmp_path)
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                STEM: {
                    "contract_version": "text-sidecars-v1",
                    "core_hashes": bad_hashes,  # type: ignore[dict-item]
                    "counts": {
                        "wikipedia_documents": 1,
                        "wikipedia_sections": 0,
                        "wikivoyage_documents": 0,
                        "wikivoyage_sections": 0,
                        "wikidata_facts": 0,
                    },
                    "completed_at": "2026-07-15T00:00:00Z",
                }
            }
        ),
        encoding="utf-8",
    )
    assert not augmentation_is_current(data_root, STEM)
