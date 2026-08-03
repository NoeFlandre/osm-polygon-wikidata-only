from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikipedia.transport import InMemoryWikipediaClient
from osm_polygon_wikidata_only.v2.extractor import V2ExtractedPbf, V2PbfStem
from osm_polygon_wikidata_only.v2.runner import run_v2_sync
from osm_polygon_wikidata_only.v2.storage import write_v2_region


def test_v2_runner_is_resumable_and_publishes_metadata_last(tmp_path: Path, monkeypatch) -> None:
    root = DataRoot(tmp_path)
    root.ensure()
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
            push=True,
            upload=lambda ops, message: uploads.append((ops, message)),
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
            push=True,
            upload=lambda ops, message: uploads.append((ops, message)),
        )
        == 0
    )
    assert [message for _, message in uploads] == ["Update V2 dataset card and manifest"]
