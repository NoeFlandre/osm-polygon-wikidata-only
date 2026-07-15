"""Safety contracts for post-publication legacy article retirement."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_document_migration import (
    MigrationError,
    apply_migration,
    plan_migration,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_retirement import (
    finalize_local_retirement,
    prepare_local_retirement,
)
from osm_polygon_wikidata_only.config.paths import DataRoot

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "processed"
STEM = "monaco-latest"


def _seed(tmp_path: Path) -> DataRoot:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    for relative in (
        f"articles/{STEM}.parquet",
        f"polygon_articles/{STEM}.parquet",
        f"wikipedia/documents/{STEM}.parquet",
        f"wikipedia/sections/{STEM}.parquet",
    ):
        destination = data_root.processed / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIXTURES / relative, destination)
    (data_root.processed_manifests / "processed_pbfs.json").write_text(
        json.dumps({f"{STEM}.osm.pbf": {"articles_path": f"articles/{STEM}.parquet"}}),
        encoding="utf-8",
    )
    apply_migration(plan_migration(data_root.processed, stems={STEM}))
    return data_root


def test_prepare_repoints_manifest_without_deleting_legacy(tmp_path: Path) -> None:
    data_root = _seed(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"

    prepare_local_retirement(data_root, STEM)

    assert legacy.exists()
    entry = json.loads((data_root.processed_manifests / "processed_pbfs.json").read_text())[
        f"{STEM}.osm.pbf"
    ]
    assert "articles_path" not in entry
    assert entry["wikipedia_documents_path"] == f"wikipedia/documents/{STEM}.parquet"


def test_finalize_deletes_only_after_lossless_checks(tmp_path: Path) -> None:
    data_root = _seed(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"

    finalize_local_retirement(data_root, STEM)

    assert not legacy.exists()
    prepare_local_retirement(data_root, STEM)  # crash-safe retry


def test_conflicting_document_keeps_legacy(tmp_path: Path) -> None:
    data_root = _seed(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    document = data_root.processed / "wikipedia" / "documents" / f"{STEM}.parquet"
    table = pq.read_table(document)  # type: ignore[no-untyped-call]
    values = table.to_pylist()
    values[0]["title"] = "conflict"
    pq.write_table(pa.Table.from_pylist(values, schema=table.schema), document)

    with pytest.raises(MigrationError, match="safe to retire"):
        finalize_local_retirement(data_root, STEM)

    assert legacy.exists()


def test_unresolved_sections_keep_legacy(tmp_path: Path) -> None:
    data_root = _seed(tmp_path)
    legacy = data_root.processed_articles / f"{STEM}.parquet"
    sections = data_root.processed / "wikipedia" / "sections" / f"{STEM}.parquet"
    table = pq.read_table(sections)  # type: ignore[no-untyped-call]
    values = table.to_pylist()
    values[0]["document_id"] = "Q999:wikipedia:en:1:1"
    pq.write_table(pa.Table.from_pylist(values, schema=table.schema), sections)

    with pytest.raises(MigrationError, match="sections unresolved"):
        finalize_local_retirement(data_root, STEM)

    assert legacy.exists()
