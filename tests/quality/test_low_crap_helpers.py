"""Characterization contracts for small deterministic quality-gate helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from osm_polygon_wikidata_only.augmentation.integrity import (
    INTEGRITY_CONTRACT_VERSION,
    IntegrityReport,
    PolygonArticlesIntegrityResult,
    WikivoyageIntegrityResult,
)
from osm_polygon_wikidata_only.augmentation.orchestrator import _completed_region_stems
from osm_polygon_wikidata_only.cli.parser import parse_languages
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.models import ManifestStats
from osm_polygon_wikidata_only.hf._dataset_stats.combined_languages import _parse_cached_stats
from osm_polygon_wikidata_only.hf._dataset_stats.models import (
    AugmentationStats,
    CombinedLanguageStats,
    ProjectTextStats,
    WikidataFactStats,
)
from osm_polygon_wikidata_only.hf._dataset_stats.rendering import (
    _render_augmentation_coverage_table,
)
from osm_polygon_wikidata_only.hf.upload_queue import (
    QUEUE_CONTRACT_VERSION,
    _read_envelope,
)
from osm_polygon_wikidata_only.io.manifest import make_entry, update_entry
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    MigrationPlan,
    StemClassification,
    StemPlan,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.checkpoints import (
    RecoveryBatchArtifacts,
    RecoveryCheckpointStore,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.models import RecoveryRepairError
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.progress import (
    RecoveryHeartbeat,
    RecoveryProgress,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.validation import (
    _unique_mapping,
    _unique_rows,
)
from osm_polygon_wikidata_only.pipeline.containment_migration import _remap_link
from osm_polygon_wikidata_only.pipeline.link_migration import (
    _ensure_migration_plan_safe,
    _polygon_qid_set,
)
from osm_polygon_wikidata_only.pipeline.persistence import (
    _enforce_integrity_if_needed,
    _integrity_metadata,
)
from osm_polygon_wikidata_only.v2.card import V2CardStats
from osm_polygon_wikidata_only.v2.reuse import _normalize_link


def test_completed_region_stems_intersects_polygon_and_article_shards(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    root.processed_articles.mkdir(parents=True)
    root.processed_polygons.mkdir(parents=True)
    (root.processed_articles / "z-latest.parquet").touch()
    (root.processed_articles / "shared-latest.parquet").touch()
    (root.processed / "wikipedia" / "documents").mkdir(parents=True)
    (root.processed / "wikipedia" / "documents" / "wiki-only.parquet").touch()
    (root.processed / "wikipedia" / "documents" / "shared-latest.parquet").touch()
    (root.processed_polygons / "shared-latest.parquet").touch()
    (root.processed_polygons / "polygon-only.parquet").touch()

    assert _completed_region_stems(root) == ["shared-latest"]


def test_integrity_report_serializes_nested_results() -> None:
    report = IntegrityReport(
        contract_version=INTEGRITY_CONTRACT_VERSION,
        polygon_articles=(PolygonArticlesIntegrityResult("demo", 2, 1, 1, True),),
        wikivoyage=(WikivoyageIntegrityResult("demo", 3, 2, 1, 4, 3, 1, True, True),),
        audit_path=Path("audit.json"),
    )

    payload = report.to_dict()

    assert payload["contract_version"] == INTEGRITY_CONTRACT_VERSION
    assert payload["polygon_articles"][0]["rejected_row_count"] == 1
    assert payload["wikivoyage"][0]["cascaded_section_count"] == 1


def test_parse_languages_trims_deduplicates_and_sorts() -> None:
    assert parse_languages(" fr, en,fr ,, de ") == ("de", "en", "fr")


def test_render_augmentation_coverage_table_includes_orphans() -> None:
    stats = AugmentationStats(
        core_region_count=4,
        fully_augmented_count=2,
        partial_augmented_count=1,
        not_augmented_count=1,
        orphan_sidecar_stems=("orphan-latest",),
        wikipedia_documents=ProjectTextStats(),
        wikipedia_sections=ProjectTextStats(),
        wikivoyage_documents=ProjectTextStats(),
        wikivoyage_sections=ProjectTextStats(),
        wikidata_facts=WikidataFactStats(),
        core_parquet_bytes=0,
        augmentation_parquet_bytes=0,
        total_parquet_bytes=0,
        unreadable_file_count=0,
        combined_languages=CombinedLanguageStats(),
    )

    rendered = _render_augmentation_coverage_table(stats)

    assert "| Core regions | 4 | 100.0% |" in rendered
    assert "| Fully augmented | 2 | 50.0% |" in rendered
    assert "Orphan stems: orphan-latest" in rendered


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"contract_version": QUEUE_CONTRACT_VERSION, "sequence": 3}, True),
        ({"contract_version": "legacy"}, False),
        (["not-an-object"], False),
    ],
)
def test_read_envelope_accepts_only_current_json_objects(
    tmp_path: Path, payload: object, expected: bool
) -> None:
    path = tmp_path / "envelope.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert (_read_envelope(path) is not None) is expected


def test_unique_recovery_helpers_keep_rows_and_reject_duplicates() -> None:
    rows = [{"id": "a", "value": "Q1"}, {"id": "b", "value": "Q2"}]

    assert _unique_mapping(rows, "id", "value", "id") == {"a": "Q1", "b": "Q2"}
    assert _unique_rows(rows, "id", "id") == {"a": rows[0], "b": rows[1]}
    with pytest.raises(RecoveryRepairError):
        _unique_mapping([rows[0], rows[0]], "id", "value", "id")
    with pytest.raises(RecoveryRepairError):
        _unique_rows([rows[0], rows[0]], "id", "id")


def test_polygon_qid_set_parses_distinct_osm_tag_qids() -> None:
    table = pa.table({"wikidata": ["Q1; Q2", "Q2", ""]})

    assert _polygon_qid_set(table) == {"Q1", "Q2"}


def test_integrity_metadata_serializes_rejections() -> None:
    result = PolygonArticlesIntegrityResult("demo", 1, 0, 1, True)

    assert _integrity_metadata(result) == {
        "contract_version": INTEGRITY_CONTRACT_VERSION,
        "shard": "demo",
        "original_row_count": 1,
        "retained_row_count": 0,
        "rejected_row_count": 1,
        "rewritten": True,
        "rejections": [],
    }


def test_manifest_update_entry_preserves_existing_fields(tmp_path: Path) -> None:
    path = tmp_path / "processed_pbfs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = make_entry(
        source_pbf="demo.osm.pbf",
        region="demo",
        polygons_path="polygons/demo.parquet",
        articles_path="articles/demo.parquet",
        polygon_articles_path="polygon_articles/demo.parquet",
        stats=ManifestStats(polygon_count=1),
        extraction_version="test",
        processed_at="2026-01-01T00:00:00Z",
    )
    path.write_text(json.dumps({"demo.osm.pbf": entry}), encoding="utf-8")

    updated = update_entry(path, source_pbf="demo.osm.pbf", polygon_count=2)

    assert updated["source_pbf"] == "demo.osm.pbf"
    assert updated["polygon_count"] == 2
    assert updated["region"] == "demo"


def test_v2_card_counts_documents_in_other_languages() -> None:
    stats = V2CardStats(
        regions=1,
        polygons=1,
        unique_wikidata_entities=1,
        wikipedia_documents=10,
        wikipedia_sections=1,
        wikivoyage_documents=0,
        wikivoyage_sections=0,
        wikidata_facts=0,
        polygon_document_links=0,
        wikipedia_tag_only_polygons=0,
        document_words=0,
        languages=2,
        new_polygons_vs_v1=None,
        new_wikipedia_documents_vs_v1=None,
        text_coverage_funnel=(),
        top_wikipedia_languages=(("en", 7), ("fr", 1)),
        polygon_link_storage_bytes=0,
        total_parquet_storage_bytes=0,
    )

    assert stats.other_wikipedia_languages == 2


def test_parse_cached_stats_accepts_valid_payload_and_rejects_malformed() -> None:
    payload = {
        "document_count": "3",
        "language_count": 2,
        "documents_per_language": [["en", "2"], ["fr", 1]],
        "polygons_per_language": [["en", 1]],
    }

    parsed = _parse_cached_stats(payload)

    assert parsed == CombinedLanguageStats(
        document_count=3,
        language_count=2,
        documents_per_language=(("en", 2), ("fr", 1)),
        polygons_per_language=(("en", 1),),
    )
    assert _parse_cached_stats({"document_count": "bad"}) is None


def test_remap_link_updates_canonical_identity_and_preserves_unknown_rows() -> None:
    parent_polygons = {
        ("way", 7): {
            "polygon_id": "parent:way:7",
            "source_pbf": "parent.osm.pbf",
            "region": "parent",
        }
    }
    row = {
        "polygon_id": "child:way:7",
        "osm_type": "way",
        "osm_id": 7,
        "source_pbf": "child.osm.pbf",
        "region": "child",
    }

    assert _remap_link(row, parent_stem="parent-latest", parent_polygons=parent_polygons) == {
        **row,
        "polygon_id": "parent:way:7",
        "source_pbf": "parent.osm.pbf",
        "region": "parent",
    }
    no_polygon_id = {"article_id": "article"}
    assert (
        _remap_link(no_polygon_id, parent_stem="parent-latest", parent_polygons=parent_polygons)
        is no_polygon_id
    )


def test_ensure_migration_plan_safe_accepts_clean_and_rejects_blocked() -> None:
    def stem(classification: StemClassification) -> StemPlan:
        return StemPlan("demo", classification, "reason", "p", "l", "d", 0, None)

    _ensure_migration_plan_safe(
        MigrationPlan(Path("processed"), (stem(StemClassification.CANONICAL),))
    )
    with pytest.raises(ValueError, match="blocked stems"):
        _ensure_migration_plan_safe(
            MigrationPlan(Path("processed"), (stem(StemClassification.BLOCKED),))
        )


def test_normalize_link_resolves_article_identity_and_rejects_missing_document() -> None:
    base = {"article_id": "article", "wikidata": "Q1", "language": "en"}
    normalized = _normalize_link(base, {"article": {"document_id": "doc-1"}})

    assert normalized["document_id"] == "doc-1"
    assert json.loads(normalized["link_sources"]) == ["wikidata_sitelink"]
    with pytest.raises(ValueError, match="no resolvable document identity"):
        _normalize_link(base, {})


def test_recovery_heartbeat_logs_and_stops_when_logging_fails() -> None:
    class StopAfterOneWait:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, _interval: float) -> bool:
            self.calls += 1
            return self.calls > 1

    progress = RecoveryProgress("demo-latest", 1, clock=lambda: 0.0)
    messages: list[str] = []
    heartbeat = RecoveryHeartbeat(progress, messages.append, interval_s=0.0)
    heartbeat._stop = StopAfterOneWait()  # type: ignore[assignment]
    heartbeat._run()
    assert len(messages) == 1

    failing = RecoveryHeartbeat(
        progress,
        lambda _message: (_ for _ in ()).throw(RuntimeError()),
    )
    failing._stop = StopAfterOneWait()  # type: ignore[assignment]
    failing._run()


def test_recovery_checkpoint_save_is_idempotent(tmp_path: Path) -> None:
    store = RecoveryCheckpointStore(tmp_path, "demo-latest", "plan")
    artifacts = RecoveryBatchArtifacts(qids=("Q1",), documents=(), sections=(), facts=())

    first = store.save(0, artifacts)
    second = store.save(0, artifacts)

    assert second == first


def test_enforce_integrity_helper_skips_missing_inputs_and_clean_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = {"source": "fixture"}
    missing_result, missing_entry = _enforce_integrity_if_needed(
        DataRoot(tmp_path),
        "demo-latest",
        "demo.osm.pbf",
        tmp_path / "missing-links.parquet",
        tmp_path / "missing-polygons.parquet",
        tmp_path / "manifest.json",
        entry,
    )
    assert missing_result is None
    assert missing_entry is entry

    links_path = tmp_path / "links.parquet"
    polygons_path = tmp_path / "polygons.parquet"
    links_path.touch()
    polygons_path.touch()
    clean = PolygonArticlesIntegrityResult("demo-latest", 1, 1, 0, False)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.pipeline.persistence.enforce_polygon_articles_integrity",
        lambda *_args: clean,
    )
    clean_result, clean_entry = _enforce_integrity_if_needed(
        DataRoot(tmp_path),
        "demo-latest",
        "demo.osm.pbf",
        links_path,
        polygons_path,
        tmp_path / "manifest.json",
        entry,
    )
    assert clean_result == clean
    assert clean_entry is entry
