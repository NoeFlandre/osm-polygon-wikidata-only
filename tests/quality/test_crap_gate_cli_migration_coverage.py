from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pytest

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.models import Polygon
from osm_polygon_wikidata_only.enrichment.wikidata_client import InMemoryWikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia_client import InMemoryWikipediaClient
from osm_polygon_wikidata_only.hf.uploader import UploadError
from osm_polygon_wikidata_only.pipeline import row_construction
from osm_polygon_wikidata_only.pipeline._link_migration import artifacts as link_artifacts
from osm_polygon_wikidata_only.pipeline._link_migration import planning as link_planning
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    StemClassification,
    StemPlan,
)
from osm_polygon_wikidata_only.v2 import cli as v2_cli
from osm_polygon_wikidata_only.v2 import index_scanning, language_split_manifest


def test_v2_inventory_handles_disabled_success_and_unavailable_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(push=False)
    settings = Settings(hf_token="token")
    assert v2_cli._fetch_inventory(args, "owner/repo", settings, None) is None

    inventory = object()
    monkeypatch.setattr(
        v2_cli.RemoteInventory,
        "fetch",
        lambda repo_id, *, hub, token: inventory,
    )
    args.push = True
    assert v2_cli._fetch_inventory(args, "owner/repo", settings, None) is inventory

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise UploadError("not listable")

    monkeypatch.setattr(v2_cli.RemoteInventory, "fetch", unavailable)
    assert v2_cli._fetch_inventory(args, "owner/repo", settings, None) is None


def test_v2_trackio_publisher_is_only_built_for_live_pushes(tmp_path: Path) -> None:
    assert (
        v2_cli._build_trackio_publisher(
            argparse.Namespace(push=False, dry_run=False), DataRoot(tmp_path)
        )
        is None
    )
    assert (
        v2_cli._build_trackio_publisher(
            argparse.Namespace(push=True, dry_run=True), DataRoot(tmp_path)
        )
        is None
    )
    assert callable(
        v2_cli._build_trackio_publisher(
            argparse.Namespace(push=True, dry_run=False), DataRoot(tmp_path)
        )
    )


@pytest.mark.parametrize("error", [ValueError("bad value"), RuntimeError("bad table")])
def test_v1_index_row_group_errors_keep_their_contract(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(
        index_scanning, "_read_row_group", lambda *_args: (_ for _ in ()).throw(error)
    )

    if isinstance(error, ValueError):
        with pytest.raises(ValueError, match="bad value"):
            cast(Any, index_scanning._read_index_rows)(object(), Path("shard.parquet"), False, 0)
    else:
        with pytest.raises(ValueError, match="V1 document shard is unreadable"):
            cast(Any, index_scanning._read_index_rows)(object(), Path("shard.parquet"), False, 0)


def test_v1_legacy_row_conversion_wraps_bad_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        index_scanning,
        "wikipedia_document_from_article_row",
        lambda _row: (_ for _ in ()).throw(TypeError("invalid legacy row")),
    )
    table = SimpleNamespace(to_pylist=lambda: [{"article_id": "bad"}])

    with pytest.raises(ValueError, match="V1 legacy article shard is invalid"):
        index_scanning._legacy_rows(table, Path("legacy.parquet"))


def test_manifest_record_readers_reject_wrong_shapes_and_escape_roots(tmp_path: Path) -> None:
    assert language_split_manifest._manifest_records("not-a-list") == ()
    assert language_split_manifest._manifest_records([{"path": "ok"}, None, 42]) == (
        {"path": "ok"},
    )

    missing = tmp_path / "missing.json"
    assert language_split_manifest._read_previous_manifest(missing) is None
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    assert language_split_manifest._read_previous_manifest(manifest) is None
    manifest.write_text("{broken", encoding="utf-8")
    assert language_split_manifest._read_previous_manifest(manifest) is None

    root = tmp_path.resolve()
    assert language_split_manifest._manifest_output_root(None, root) is None
    assert language_split_manifest._manifest_output_root({"output_root": 42}, root) is None
    assert (
        language_split_manifest._manifest_output_root({"output_root": "../outside"}, root) is None
    )
    assert language_split_manifest._manifest_output_root(
        {"output_root": "language_splits"}, root
    ) == (root / "language_splits")


@pytest.mark.parametrize("stem", ["", ".", "..", "nested/path", r"nested\path"])
def test_link_migration_stem_validation_rejects_unsafe_values(stem: str) -> None:
    assert not link_planning._is_valid_stem(stem)


@pytest.mark.parametrize("error", [KeyError("missing column"), pa.ArrowInvalid("bad schema")])
def test_legacy_rejection_table_reader_returns_none_for_invalid_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    stem_plan = StemPlan(
        "region-latest",
        StemClassification.MIGRATABLE,
        "",
        "",
        "",
        "",
        0,
        None,
    )
    for path in link_planning._stem_paths(stem_plan.stem, tmp_path)[:2]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(
        link_planning.pq, "read_table", lambda *_args, **_kwargs: (_ for _ in ()).throw(error)
    )

    assert link_planning._read_legacy_rejection_tables(tmp_path, stem_plan) is None


def test_source_pbf_helper_rejects_zero_or_multiple_values() -> None:
    class Column:
        def __init__(self, values: list[str]) -> None:
            self.values = values

        def to_pylist(self) -> list[str]:
            return self.values

    class Table:
        def __init__(self, values: list[str]) -> None:
            self.values = values

        def column(self, name: str) -> Column:
            assert name == "source_pbf"
            return Column(self.values)

    for values, count in [([], 0), (["a.pbf", "b.pbf"], 2)]:
        inputs = SimpleNamespace(
            stem_plan=SimpleNamespace(stem="region-latest"),
            polygons_table=Table(values),
        )
        with pytest.raises(RuntimeError, match=f"{count} distinct source_pbf"):
            link_artifacts.source_pbf_for_stem(cast(Any, inputs))


def test_enrich_polygon_fetches_and_summarizes_missing_qid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon = Polygon.make(
        source_pbf_stem="region-latest",
        region="region",
        source_pbf="region-latest.osm.pbf",
        osm_type="relation",
        osm_id=1,
        wikidata="Q1",
        name="Place",
        tags="{}",
        tag_keys="[]",
        tag_count=0,
        osm_primary_tag="",
        centroid='{"type":"Point","coordinates":[0,0]}',
        lat=0.0,
        lon=0.0,
        bbox="[0,0,0,0]",
        area_m2=1.0,
        area_km2=0.000001,
        area_bucket="small",
        has_name=True,
        has_wikidata=True,
        extraction_version="v1",
        extracted_at="now",
    )
    summary = SimpleNamespace(
        articles=[
            SimpleNamespace(language="en", full_text="English text"),
            SimpleNamespace(language="fr", full_text=""),
        ],
        best_language=lambda: "en",
    )
    monkeypatch.setattr(row_construction, "fetch_qids", lambda *_args, **_kwargs: [summary])
    summaries = {}

    enriched = row_construction.enrich_polygon(
        polygon,
        wikidata_client=InMemoryWikidataClient({}),
        wikipedia_client=InMemoryWikipediaClient({}),
        settings=Settings(languages=("en", "fr")),
        summaries=summaries,
    )

    assert summaries["Q1"] is summary
    assert enriched.has_wikipedia
    assert enriched.wikipedia_languages == '["en","fr"]'
    assert enriched.wikipedia_article_count == 2
    assert enriched.has_english_wikipedia and enriched.has_french_wikipedia
    assert enriched.text_available
