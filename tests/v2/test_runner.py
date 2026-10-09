import threading
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikipedia.transport import InMemoryWikipediaClient
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.v2.card_models import V2CardStats
from osm_polygon_wikidata_only.v2.config import V2_ASSET_PATHS
from osm_polygon_wikidata_only.v2.extractor import V2ExtractedPbf, V2PbfStem
from osm_polygon_wikidata_only.v2.runner import V2Publication, run_v2_sync
from osm_polygon_wikidata_only.v2.storage import write_v2_region

pytestmark = pytest.mark.map_orchestration


class _ReadyIndex:
    def wait_until_ready(self) -> None:
        return None

    def close(self) -> None:
        return None


def _stub_fast_v2_orchestration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub storage and release rendering outside runner orchestration contracts."""

    def write_map_placeholders(
        processed_v2: Path,
        output_dir: Path,
        **_kwargs: object,
    ) -> tuple[Path, Path, Path]:
        del output_dir
        paths = tuple(processed_v2 / asset for asset in V2_ASSET_PATHS)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test map")
        return paths[0], paths[1], paths[2]

    def write_card_placeholder(processed_v2: Path, **_kwargs: object) -> Path:
        card = processed_v2 / "README.md"
        card.write_text("# Test V2 card\n", encoding="utf-8")
        return card

    card_stats = V2CardStats(
        regions=0,
        polygons=0,
        unique_wikidata_entities=0,
        wikipedia_documents=0,
        wikipedia_sections=0,
        wikivoyage_documents=0,
        wikivoyage_sections=0,
        wikidata_facts=0,
        polygon_document_links=0,
        wikipedia_tag_only_polygons=0,
        document_words=0,
        languages=0,
        new_polygons_vs_v1=None,
        new_wikipedia_documents_vs_v1=None,
        text_coverage_funnel=(),
        top_wikipedia_languages=(),
        polygon_link_storage_bytes=0,
        total_parquet_storage_bytes=0,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.start_v1_reuse_index",
        lambda *_args, **_kwargs: _ReadyIndex(),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.generate_v2_map_assets",
        write_map_placeholders,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.compute_v2_card_stats",
        lambda *_args, **_kwargs: card_stats,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.write_v2_card",
        write_card_placeholder,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.storage._write_table",
        lambda path, *_args: path.write_bytes(b"orchestration test artifact"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.storage._validate_written_table",
        lambda *_args: None,
    )


def test_v2_runner_writes_only_inside_v2_storage_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    pbf = root.raw / "region-latest.osm.pbf"
    pbf.touch()
    extracted = V2ExtractedPbf(V2PbfStem(pbf, "region-latest", "region"), (), 0.0)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.extract_v2_pbf",
        lambda *_args, **_kwargs: extracted,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.merge_v2_region",
        lambda data_root, extracted, **_kwargs: write_v2_region(
            data_root.processed_v2,
            extracted.stem.stem,
            polygons=[],
            documents=[],
            links=[],
        ),
    )

    def cache_map_data_in_v2(
        processed_v2: Path,
        assets_dir: Path,
        *,
        v1_processed: Path,
        land_geojson_path: Path | None,
        land_cache_dir: Path,
    ) -> None:
        assert processed_v2 == root.processed_v2
        assert assets_dir == root.processed_v2 / "assets"
        assert v1_processed == root.processed
        assert land_geojson_path is None
        assert land_cache_dir == root.v2_cache
        (land_cache_dir / "ne_110m_land.geojson").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.generate_v2_map_assets",
        cache_map_data_in_v2,
    )
    files_before = {path for path in root.path.rglob("*") if path.is_file()}

    assert (
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=InMemoryWikipediaClient({}),
        )
        == 0
    )

    written_files = {path for path in root.path.rglob("*") if path.is_file()} - files_before
    assert written_files
    assert all(
        path.is_relative_to(root.processed_v2) or path.is_relative_to(root.v2_cache)
        for path in written_files
    ), f"V2 wrote files outside its owned roots: {sorted(written_files)}"


def test_v2_runner_is_resumable_and_publishes_metadata_last(tmp_path: Path, monkeypatch) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    _stub_fast_v2_orchestration(monkeypatch)

    pbf = root.raw / "region-latest.osm.pbf"
    pbf.touch()
    extracted = V2ExtractedPbf(
        V2PbfStem(pbf, "region-latest", "region"),
        (),
        0.0,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.extract_v2_pbf",
        lambda *_args, **_kwargs: extracted,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.merge_v2_region",
        lambda data_root, extracted, **_kwargs: write_v2_region(
            data_root.processed_v2,
            extracted.stem.stem,
            polygons=[],
            documents=[],
            links=[],
        ),
    )
    uploads: list[tuple[list, str]] = []
    client = InMemoryWikipediaClient({})
    assert (
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=client,
            publication=V2Publication(
                push=True,
                upload=lambda ops, message: uploads.append((ops, message)),
            ),
        )
        == 0
    )
    assert [message for _, message in uploads] == [
        "Add V2 region region-latest",
        "Update V2 dataset card and manifest",
    ]
    uploads.clear()
    assert (
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=client,
            publication=V2Publication(
                push=True,
                upload=lambda ops, message: uploads.append((ops, message)),
                remote_inventory=RemoteInventory(
                    {
                        "polygons/region-latest.parquet",
                        "wikipedia/documents/region-latest.parquet",
                        "wikipedia/sections/region-latest.parquet",
                        "polygon_document_links/region-latest.parquet",
                    }
                ),
            ),
        )
        == 0
    )
    assert [message for _, message in uploads] == ["Update V2 dataset card and manifest"]

    uploads.clear()
    assert (
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=client,
            publication=V2Publication(
                push=True,
                upload=lambda ops, message: uploads.append((ops, message)),
                remote_inventory=RemoteInventory({"polygons/region-latest.parquet"}),
            ),
        )
        == 0
    )
    assert [message for _, message in uploads] == [
        "Repair V2 region region-latest",
        "Update V2 dataset card and manifest",
    ]


def test_v2_runner_resumes_provisional_region_without_reextracting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A persisted provisional region is finalized from disk on restart."""
    _stub_fast_v2_orchestration(monkeypatch)
    root = DataRoot(tmp_path)
    root.ensure()
    pbf = root.raw / "region-latest.osm.pbf"
    pbf.touch()
    write_v2_region(
        root.processed_v2,
        "region-latest",
        polygons=[],
        documents=[],
        links=[],
        v1_index_reconciled=False,
    )
    reconciled: list[str] = []

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.extract_v2_pbf",
        lambda *_args, **_kwargs: pytest.fail("a persisted provisional region was re-extracted"),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.reconcile_v2_region",
        lambda _data_root, stem, **_kwargs: reconciled.append(stem),
    )

    assert (
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=InMemoryWikipediaClient({}),
        )
        == 0
    )
    assert reconciled == ["region-latest"]


def test_v2_runner_groups_region_publications_into_bounded_commits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Many completed regions are published in batches, not one commit each."""
    _stub_fast_v2_orchestration(monkeypatch)
    root = DataRoot(tmp_path)
    root.ensure()
    pbfs = []
    for index in range(17):
        pbf = root.raw / f"region-{index:02d}-latest.osm.pbf"
        pbf.touch()
        pbfs.append(pbf)

    def extract(path: Path, **_kwargs: object) -> V2ExtractedPbf:
        return V2ExtractedPbf(
            V2PbfStem(path, path.name.removesuffix(".osm.pbf"), "region"),
            (),
            0.0,
        )

    monkeypatch.setattr("osm_polygon_wikidata_only.v2.runner.extract_v2_pbf", extract)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.merge_v2_region",
        lambda data_root, extracted, **_kwargs: write_v2_region(
            data_root.processed_v2,
            extracted.stem.stem,
            polygons=[],
            documents=[],
            links=[],
        ),
    )
    uploads: list[tuple[list, str]] = []

    assert (
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=InMemoryWikipediaClient({}),
            publication=V2Publication(
                push=True,
                upload=lambda ops, message: uploads.append((ops, message)),
            ),
        )
        == 0
    )

    region_uploads = uploads[:-1]
    assert len(region_uploads) == 2
    assert [len(ops) for ops, _ in region_uploads] == [64, 4]
    assert uploads[-1][1] == "Update V2 dataset card and manifest"


def test_v2_runner_rebuilds_a_region_when_a_manifest_file_is_tampered(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_fast_v2_orchestration(monkeypatch)
    root = DataRoot(tmp_path)
    root.ensure()
    pbf = root.raw / "region-latest.osm.pbf"
    pbf.touch()
    extracted = V2ExtractedPbf(V2PbfStem(pbf, "region-latest", "region"), (), 0.0)
    calls = 0

    def extract(*_args: object, **_kwargs: object) -> V2ExtractedPbf:
        nonlocal calls
        calls += 1
        return extracted

    monkeypatch.setattr("osm_polygon_wikidata_only.v2.runner.extract_v2_pbf", extract)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.merge_v2_region",
        lambda data_root, extracted, **_kwargs: write_v2_region(
            data_root.processed_v2,
            extracted.stem.stem,
            polygons=[],
            documents=[],
            links=[],
        ),
    )
    client = InMemoryWikipediaClient({})
    for _ in range(2):
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=client,
        )
        if calls == 1:
            path = root.processed_v2 / "polygons" / "region-latest.parquet"
            path.write_bytes(path.read_bytes() + b"tampered")
    assert calls == 2


def test_v2_runner_extracts_while_v1_index_is_still_building(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    pbf = root.raw / "region-latest.osm.pbf"
    pbf.touch()
    events: list[str] = []

    class InFlightIndex:
        is_ready = False

        def wait_until_ready(self) -> None:
            events.append("index-ready")
            self.is_ready = True

        def close(self) -> None:
            events.append("index-closed")

    index = InFlightIndex()
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.start_v1_reuse_index",
        lambda *_args, **_kwargs: events.append("index-started") or index,
    )

    def extract(*_args: object, **_kwargs: object) -> V2ExtractedPbf:
        events.append("extracted")
        assert not index.is_ready
        return V2ExtractedPbf(V2PbfStem(pbf, "region-latest", "region"), (), 0.0)

    monkeypatch.setattr("osm_polygon_wikidata_only.v2.runner.extract_v2_pbf", extract)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.merge_v2_region",
        lambda data_root, extracted, **_kwargs: write_v2_region(
            data_root.processed_v2,
            extracted.stem.stem,
            polygons=[],
            documents=[],
            links=[],
        ),
    )

    assert (
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=InMemoryWikipediaClient({}),
        )
        == 0
    )
    assert events == ["index-started", "extracted", "index-ready", "index-closed"]


def test_v2_runner_prefetches_next_extraction_before_current_merge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    first = root.raw / "first-latest.osm.pbf"
    second = root.raw / "second-latest.osm.pbf"
    first.touch()
    second.touch()
    first_extracted = V2ExtractedPbf(V2PbfStem(first, "first-latest", "first"), (), 0.0)
    second_extracted = V2ExtractedPbf(V2PbfStem(second, "second-latest", "second"), (), 0.0)
    second_started = threading.Event()
    merge_finished = threading.Event()

    class ReadyIndex:
        is_ready = True

        def wait_until_ready(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.start_v1_reuse_index",
        lambda *_args, **_kwargs: ReadyIndex(),
    )

    def extract(path: Path, **_kwargs: object) -> V2ExtractedPbf:
        if path == second:
            second_started.set()
            return second_extracted
        return first_extracted

    monkeypatch.setattr("osm_polygon_wikidata_only.v2.runner.extract_v2_pbf", extract)

    def merge(data_root: DataRoot, extracted: V2ExtractedPbf, **_kwargs: object) -> None:
        if extracted.stem.stem == "first-latest":
            assert second_started.wait(timeout=2)
            merge_finished.set()
        write_v2_region(
            data_root.processed_v2,
            extracted.stem.stem,
            polygons=[],
            documents=[],
            links=[],
        )

    monkeypatch.setattr("osm_polygon_wikidata_only.v2.runner.merge_v2_region", merge)
    assert (
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=InMemoryWikipediaClient({}),
        )
        == 0
    )
    assert merge_finished.is_set()


def test_v2_runner_processes_regions_before_final_index_reconciliation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    first = root.raw / "first-latest.osm.pbf"
    second = root.raw / "second-latest.osm.pbf"
    first.touch()
    second.touch()
    first_extracted = V2ExtractedPbf(
        V2PbfStem(first, "first-latest", "first"),
        ({"polygon_id": "first", "wikipedia_tag_refs": '[{"language":"en"}]'},),
        0.0,
    )
    second_extracted = V2ExtractedPbf(
        V2PbfStem(second, "second-latest", "second"),
        ({"polygon_id": "second", "wikipedia_tag_refs": '[{"language":"en"}]'},),
        0.0,
    )
    merged: list[str] = []
    reconciled: list[str] = []

    class InFlightIndex:
        is_ready = False

        def wait_until_ready(self) -> None:
            assert merged == ["first-latest", "second-latest"]
            self.is_ready = True

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.start_v1_reuse_index",
        lambda *_args, **_kwargs: InFlightIndex(),
    )

    def extract(path: Path, **_kwargs: object) -> V2ExtractedPbf:
        return first_extracted if path == first else second_extracted

    monkeypatch.setattr("osm_polygon_wikidata_only.v2.runner.extract_v2_pbf", extract)

    def merge(data_root: DataRoot, extracted: V2ExtractedPbf, **_kwargs: object) -> None:
        merged.append(extracted.stem.stem)
        write_v2_region(
            data_root.processed_v2,
            extracted.stem.stem,
            polygons=[],
            documents=[],
            links=[],
        )

    monkeypatch.setattr("osm_polygon_wikidata_only.v2.runner.merge_v2_region", merge)
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.reconcile_v2_region",
        lambda _data_root, stem, **_kwargs: reconciled.append(stem),
    )
    assert (
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=InMemoryWikipediaClient({}),
        )
        == 0
    )
    assert reconciled == ["first-latest", "second-latest"]


def test_v2_runner_does_not_start_later_extraction_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
    first = root.raw / "first-latest.osm.pbf"
    second = root.raw / "second-latest.osm.pbf"
    first.touch()
    second.touch()
    calls: list[str] = []

    class ReadyIndex:
        is_ready = True

        def wait_until_ready(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.v2.runner.start_v1_reuse_index",
        lambda *_args, **_kwargs: ReadyIndex(),
    )

    def extract(path: Path, **_kwargs: object) -> V2ExtractedPbf:
        calls.append(path.name)
        if path == first:
            raise RuntimeError("first extraction failed")
        raise AssertionError("later extraction started after a failure")

    monkeypatch.setattr("osm_polygon_wikidata_only.v2.runner.extract_v2_pbf", extract)
    with pytest.raises(RuntimeError, match="first extraction failed"):
        run_v2_sync(
            root.raw,
            data_root=root,
            settings=Settings(skip_existing=True),
            wikipedia_client=InMemoryWikipediaClient({}),
        )
    assert calls == [first.name]
