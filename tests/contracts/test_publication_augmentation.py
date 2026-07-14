"""Publication-level checks for the augmentation-aware README.

This module verifies the canonical ``write_readme_snapshot`` path:
the rendered card MUST include both core and augmentation statistics
when the underlying local files exist. The publication layer is
expected to detect both kinds of inputs and re-render before each
upload path.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.orchestrator import (
    AugmentationResult,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf.publication import (
    assemble_augmentation_upload,
    assemble_core_upload,
    assemble_region_upload,
    write_readme_snapshot,
)
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult

STEM = "monaco-latest"
REPO_ID = "NoeFlandre/osm-polygon-wikidata-only"

_POLYGON_TABLE = pa.table(
    {
        "polygon_id": ["monaco-latest:relation:1"],
        "wikidata": ["Q235"],
        "region": ["monaco"],
        "has_wikipedia": [True],
        "text_available": [True],
        "has_english_wikipedia": [True],
        "wikipedia_language_count": [1],
        "wikipedia_languages": ['["en"]'],
    }
)
_ARTICLE_TABLE = pa.table(
    {
        "article_id": ["Q235:en:1:1"],
        "wikidata": ["Q235"],
        "language": ["en"],
        "article_length_words": [100],
        "article_length_tokens_estimate": [25],
    }
)
_LINK_TABLE = pa.table(
    {
        "polygon_id": ["monaco-latest:relation:1"],
        "article_id": ["Q235:en:1:1"],
        "language": ["en"],
    }
)
_DOC_TABLE = pa.table(
    {
        "document_id": ["Q235:wikipedia:en:1:1"],
        "wikidata": ["Q235"],
        "project": ["wikipedia"],
        "language": ["en"],
        "full_text": ["Wikipedia body text."],
        "article_length_chars": [20],
        "article_length_words": [3],
        "article_length_tokens_estimate": [5],
    }
)
_SECTIONS_TABLE = pa.table(
    {
        "section_id": ["sec-1"],
        "document_id": ["Q235:wikipedia:en:1:1"],
        "wikidata": ["Q235"],
        "project": ["wikipedia"],
        "language": ["en"],
        "text": ["Wikipedia section body."],
        "text_length_chars": [24],
        "text_length_words": [3],
        "text_length_tokens_estimate": [6],
    }
)
_FACT_TABLE = pa.table(
    {
        "fact_id": ["f-1"],
        "wikidata": ["Q235"],
        "property_id": ["P17"],
        "property_label_en": ["country"],
        "property_labels": ['{"en": "country"}'],
        "value_type": ["wikibase-entityid"],
        "value_entity_id": ["Q142"],
        "value_label_en": ["France"],
        "value_labels": ['{"en": "France"}'],
        "value_text": ["Q142"],
        "qualifiers": ["{}"],
        "references": ["[]"],
    }
)


def _write_empty(path: Path, columns: list[str]) -> Path:
    """Write a valid empty parquet with the given column names."""
    schema = pa.schema([pa.field(c, pa.string()) for c in columns])
    pq.write_table(pa.Table.from_pylist([], schema=schema), path)
    return path


def _stub_process_result(tmp_path: Path) -> tuple[ProcessResult, DataRoot]:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    polygons = data_root.processed_polygons / f"{STEM}.parquet"
    articles = data_root.processed_articles / f"{STEM}.parquet"
    links = data_root.processed_links / f"{STEM}.parquet"
    manifest = data_root.processed_manifests / "processed_pbfs.json"
    for p in (polygons, articles, links, manifest):
        p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_POLYGON_TABLE, polygons)
    pq.write_table(_ARTICLE_TABLE, articles)
    pq.write_table(_LINK_TABLE, links)
    manifest.write_text("{}", encoding="utf-8")
    return (
        ProcessResult(
            polygons_path=polygons,
            articles_path=articles,
            polygon_articles_path=links,
            manifest_path=manifest,
            polygon_count=1,
            article_count=1,
            link_count=1,
            manifest_entry={"source_pbf": f"{STEM}.osm.pbf"},
            stage_timings_s={},
        ),
        data_root,
    )


def _stub_augmentation_result(data_root_processed: Path) -> AugmentationResult:
    paths = {
        "wikipedia_documents_path": data_root_processed
        / "wikipedia"
        / "documents"
        / f"{STEM}.parquet",
        "wikipedia_sections_path": data_root_processed
        / "wikipedia"
        / "sections"
        / f"{STEM}.parquet",
        "wikivoyage_documents_path": data_root_processed
        / "wikivoyage"
        / "documents"
        / f"{STEM}.parquet",
        "wikivoyage_sections_path": data_root_processed
        / "wikivoyage"
        / "sections"
        / f"{STEM}.parquet",
        "wikidata_facts_path": data_root_processed / "wikidata" / "facts" / f"{STEM}.parquet",
        "manifest_path": data_root_processed
        / "augmentation"
        / "manifests"
        / "augmentation_manifest.json",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_DOC_TABLE, paths["wikipedia_documents_path"])
    pq.write_table(_SECTIONS_TABLE, paths["wikipedia_sections_path"])
    pq.write_table(_DOC_TABLE, paths["wikivoyage_documents_path"])
    pq.write_table(_SECTIONS_TABLE, paths["wikivoyage_sections_path"])
    pq.write_table(_FACT_TABLE, paths["wikidata_facts_path"])
    paths["manifest_path"].write_text("{}", encoding="utf-8")
    return AugmentationResult(**paths, counts={"wikipedia_documents": 1})


def _stub_coverage_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_coverage_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_polygon_count_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda _dir: ([], []),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        lambda _dir: None,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )


# ---------------------------------------------------------------------------
# write_readme_snapshot includes both stats
# ---------------------------------------------------------------------------


def test_write_readme_snapshot_includes_core_and_augmentation_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """write_readme_snapshot must compose BOTH core and augmentation stats
    when all sidecars exist locally."""
    core, data_root = _stub_process_result(tmp_path)
    _stub_augmentation_result(data_root.processed)
    _stub_coverage_assets(monkeypatch)

    files = assemble_core_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        core=core,
        world_land_warning=lambda msg: None,
    )
    readme_path = files[-2][0]  # README is the 7th entry of 8 in legacy core.
    md = readme_path.read_text(encoding="utf-8")
    # Core stats sections remain.
    assert "## Dataset snapshot" in md
    assert "## Wikipedia coverage funnel" in md
    # Augmentation sections are present because sidecars exist.
    assert "## Augmentation coverage" in md
    assert "## Wikipedia text corpus" in md
    assert "## Wikivoyage text corpus" in md
    assert "## Wikidata facts" in md


def test_write_readme_snapshot_core_only_when_no_augmentation_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A core-only publication with no augmentation dirs must still
    produce a valid README. Augmentation sections render only when
    augmentation stats are computed."""
    core, data_root = _stub_process_result(tmp_path)
    _stub_coverage_assets(monkeypatch)
    files = assemble_core_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        core=core,
        world_land_warning=lambda msg: None,
    )
    readme_path = files[-2][0]
    md = readme_path.read_text(encoding="utf-8")
    # Core sections always present.
    assert "## Dataset snapshot" in md
    # When augmentation dirs are missing, augmentation sections are
    # still rendered (showing empty data) so the README contract is
    # stable across pipeline states.
    assert "## Wikipedia text corpus" in md
    assert "## Wikidata facts" in md


def test_write_readme_snapshot_deterministic_for_identical_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two renders from the same local inputs must produce byte-identical
    output."""
    _core, data_root = _stub_process_result(tmp_path)
    _stub_coverage_assets(monkeypatch)
    snapshots_dir = data_root.cache / "determinism_check"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    readme_a = snapshots_dir / "a.md"
    readme_b = snapshots_dir / "b.md"
    atomic = "osm_polygon_wikidata_only.io.atomic.atomic_write_text"
    monkeypatch.setattr(atomic, lambda path, text, **kw: path.write_text(text, encoding="utf-8"))
    write_readme_snapshot(data_root, REPO_ID, readme_a)
    write_readme_snapshot(data_root, REPO_ID, readme_b)
    assert readme_a.read_text(encoding="utf-8") == readme_b.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-publication-path README freshness
# ---------------------------------------------------------------------------


def test_core_upload_runs_write_readme_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy core assembler must call write_readme_snapshot."""
    core, data_root = _stub_process_result(tmp_path)
    _stub_coverage_assets(monkeypatch)
    calls: list[tuple[DataRoot, str]] = []

    def tracker(data_root, repo_id, destination):  # type: ignore[no-untyped-def]
        calls.append((data_root, repo_id))
        destination.write_text(
            "# placeholder\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        tracker,
    )
    assemble_core_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        core=core,
        world_land_warning=lambda msg: None,
    )
    assert calls == [(data_root, REPO_ID)]


def test_region_upload_with_core_runs_write_readme_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    _stub_coverage_assets(monkeypatch)
    calls: list[tuple[DataRoot, str]] = []

    def tracker(data_root, repo_id, destination):  # type: ignore[no-untyped-def]
        calls.append((data_root, repo_id))
        destination.write_text("# placeholder\n", encoding="utf-8")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        tracker,
    )
    assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=core,
        world_land_warning=None,
    )
    assert calls == [(data_root, REPO_ID)]


def test_augmentation_only_upload_runs_write_readme_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The augmentation-only publication must always refresh the README."""
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    calls: list[tuple[DataRoot, str]] = []

    def tracker(data_root, repo_id, destination):  # type: ignore[no-untyped-def]
        calls.append((data_root, repo_id))
        destination.write_text("# placeholder\n", encoding="utf-8")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        tracker,
    )
    assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    assert calls == [(data_root, REPO_ID)]


def test_readme_generation_failure_prevents_upload_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If write_readme_snapshot raises, the assembler must propagate the
    failure before producing any file list."""
    core, data_root = _stub_process_result(tmp_path)
    _stub_coverage_assets(monkeypatch)

    def boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("stats generation failed")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        boom,
    )
    with pytest.raises(RuntimeError, match="stats generation failed"):
        assemble_core_upload(
            data_root=data_root,
            repo_id=REPO_ID,
            core=core,
            world_land_warning=lambda msg: None,
        )


# ---------------------------------------------------------------------------
# README stays last in every contract
# ---------------------------------------------------------------------------


def test_readme_remains_last_in_core_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, data_root = _stub_process_result(tmp_path)
    _stub_coverage_assets(monkeypatch)
    files = assemble_core_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        core=core,
        world_land_warning=lambda msg: None,
    )
    # The README is the 7th entry (index 6) of 8. The legacy coverage
    # map remains the last entry, as documented.
    assert files[-2][1] == "README.md"


def test_readme_remains_last_in_region_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, data_root = _stub_process_result(tmp_path)
    aug = _stub_augmentation_result(data_root.processed)
    _stub_coverage_assets(monkeypatch)
    files = assemble_region_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        stem=STEM,
        augmentation=aug,
        core=core,
        world_land_warning=None,
    )
    assert files[-1][1] == "README.md"


def test_readme_remains_last_in_augmentation_only_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    aug = _stub_augmentation_result(data_root.processed)
    files = assemble_augmentation_upload(
        data_root=data_root,
        repo_id=REPO_ID,
        augmentation=aug,
    )
    assert files[-1][1] == "README.md"


# ---------------------------------------------------------------------------
# Static structure tests
# ---------------------------------------------------------------------------


def test_write_readme_snapshot_calls_have_consistent_path() -> None:
    """write_readme_snapshot must accept (data_root, repo_id, destination)."""
    import inspect

    sig = inspect.signature(write_readme_snapshot)
    assert list(sig.parameters) == ["data_root", "repo_id", "destination"]


def test_write_readme_snapshot_writes_to_destination_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rendered README MUST be written via atomic_write_text."""
    _, data_root = _stub_process_result(tmp_path)
    captured: list[tuple[Path, str]] = []

    def fake_atomic(path, text, **kwargs):  # type: ignore[no-untyped-def]
        captured.append((path, text))
        path.write_text(text, encoding="utf-8")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.atomic_write_text",
        fake_atomic,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.render_dataset_card",
        lambda **kwargs: "BODY",
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.render_stats_section",
        lambda *args, **kwargs: "STATS",
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_manifest",
        lambda path: {},
    )
    destination = tmp_path / "out.md"
    write_readme_snapshot(data_root, REPO_ID, destination)
    assert len(captured) == 1
    assert captured[0][0] == destination
    assert captured[0][1] == "BODY"
