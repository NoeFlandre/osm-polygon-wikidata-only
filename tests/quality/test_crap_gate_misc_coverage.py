from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation import integrity
from osm_polygon_wikidata_only.augmentation.schema import document_schema
from osm_polygon_wikidata_only.cli.sync_application import (
    SyncApplication,
    SyncApplicationContext,
    SyncApplicationServices,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.grid5000.sentence_controller_policy import (
    is_source_commit_migration_safe,
)
from osm_polygon_wikidata_only.hf import language_splits as hf_language_splits
from osm_polygon_wikidata_only.hf._geographic.h3_geometry import _boundary_points
from osm_polygon_wikidata_only.hf._publication.readme_snapshot import (
    integrity_audit as _integrity_audit,
)
from osm_polygon_wikidata_only.hf.language_splits import LanguageTable, LanguageTableSpec
from osm_polygon_wikidata_only.pipeline._link_migration import planning as link_planning
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    StemClassification,
    StemPlan,
)
from scripts import assemble_docs_site


def test_language_file_scan_checks_row_count_and_wraps_stream_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Batch:
        num_rows = 2

        def column(self, _index: int) -> object:
            return object()

    class Parquet:
        def __enter__(self) -> Parquet:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(hf_language_splits, "open_parquet", lambda _path: Parquet())
    monkeypatch.setattr(
        hf_language_splits,
        "iter_record_batches",
        lambda *_args, **_kwargs: [Batch()],
    )
    monkeypatch.setattr(hf_language_splits, "_observe_language_batch", lambda *_args: None)
    spec = LanguageTableSpec(
        table=LanguageTable.WIKIPEDIA_DOCUMENTS,
        relative_dir="wikipedia/documents",
        language_column="language",
        identity_columns=("document_id",),
        schema_factory=lambda: pa.schema([]),
        configuration="test",
    )
    path = tmp_path / "table.parquet"

    hf_language_splits._scan_language_file(path, spec, {}, expected_rows=2)
    with pytest.raises(hf_language_splits.LanguageInventoryError, match="row count changed"):
        hf_language_splits._scan_language_file(path, spec, {}, expected_rows=3)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("stream failed")

    monkeypatch.setattr(hf_language_splits, "iter_record_batches", fail)
    with pytest.raises(
        hf_language_splits.LanguageInventoryError, match="Could not scan language column"
    ):
        hf_language_splits._scan_language_file(path, spec, {}, expected_rows=0)


def test_integrity_audit_handles_missing_corrupt_non_object_and_valid_payloads(
    tmp_path: Path,
) -> None:
    data_root = DataRoot(tmp_path)
    path = data_root.processed / "integrity" / "integrity_audit.json"
    assert _integrity_audit(data_root) is None

    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    assert _integrity_audit(data_root) is None
    path.write_text("[]", encoding="utf-8")
    assert _integrity_audit(data_root) is None
    path.write_text(json.dumps({"rejected": 1}), encoding="utf-8")
    payload = _integrity_audit(data_root)
    assert payload is not None
    assert payload["rejected"] == 1
    assert payload["contract_version"]


def test_integrity_helpers_cover_missing_polygons_and_empty_rows(tmp_path: Path) -> None:
    path = tmp_path / "polygons.parquet"
    with pytest.raises(FileNotFoundError, match="Polygons parquet missing"):
        integrity._read_polygon_wikidata_set(path)

    pq.write_table(pa.table({"wikidata": ["Q1", "", None]}), path)
    assert integrity._read_polygon_wikidata_set(path) == {"Q1"}

    schema = document_schema()
    empty = integrity._table_from_rows([], tuple(schema.names), schema)
    assert empty.num_rows == 0
    assert empty.schema.equals(schema, check_metadata=True)


def test_source_commit_migration_requires_safe_nonempty_batches() -> None:
    assert not is_source_commit_migration_safe({})
    assert not is_source_commit_migration_safe({"batches": []})
    assert is_source_commit_migration_safe(
        {"batches": [{"state": "planned", "oar_job_id": "", "hf_commit": None}]}
    )
    assert not is_source_commit_migration_safe(
        {"batches": [{"state": "planned", "oar_job_id": "123", "hf_commit": None}]}
    )
    assert not is_source_commit_migration_safe({"batches": [None]})


def test_boundary_points_handles_empty_and_short_coordinate_pairs() -> None:
    assert _boundary_points([]) == []
    assert _boundary_points([(50, 2), (8,), (60, 4)]) == [(2.0, 50.0), (4.0, 60.0)]


def test_restore_site_removes_partial_output_and_restores_backup(tmp_path: Path) -> None:
    site = tmp_path / "site"
    backup = tmp_path / "backup"
    site.mkdir()
    backup.mkdir()
    (site / "partial.html").write_text("partial", encoding="utf-8")
    (backup / "index.html").write_text("previous", encoding="utf-8")

    assemble_docs_site._restore_site(site, backup)

    assert not (site / "partial.html").exists()
    assert (site / "index.html").read_text(encoding="utf-8") == "previous"
    assemble_docs_site._restore_site(site, None)
    assert not site.exists()


def test_metadata_repair_needed_accepts_a_clean_repository_refresh() -> None:
    context = SimpleNamespace(
        push_enabled=True,
        reconciliation_plan=SimpleNamespace(repository_refresh=True),
        core_will_be_repaired=False,
        containment_enqueued=False,
    )
    application = SyncApplication(
        context=cast(SyncApplicationContext, context),
        services=cast(SyncApplicationServices, SimpleNamespace()),
    )

    assert application._metadata_repair_needed(0)


def test_link_migration_legacy_rejection_table_reader_handles_missing_files(
    tmp_path: Path,
) -> None:
    plan = StemPlan(
        "region-latest",
        StemClassification.MIGRATABLE,
        "",
        "",
        "",
        "",
        0,
        None,
    )
    assert link_planning._read_legacy_rejection_tables(tmp_path, plan) is None
