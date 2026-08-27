from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from osm_polygon_wikidata_only.domain.schema import empty_row, polygon_schema
from osm_polygon_wikidata_only.v2 import maps as v2_maps
from osm_polygon_wikidata_only.v2.card import render_v2_card
from osm_polygon_wikidata_only.v2.maps import (
    generate_v2_added_wikipedia_tag_map,
    generate_v2_map_assets,
)
from osm_polygon_wikidata_only.v2.schema import wikipedia_document_v2_schema
from osm_polygon_wikidata_only.v2.storage import write_v2_region


def _polygon() -> dict[str, object]:
    row = empty_row(tuple(field.name for field in polygon_schema()))
    row.update(
        {
            "polygon_id": "region-latest:way:1",
            "source_pbf": "region-latest.osm.pbf",
            "region": "region",
            "osm_type": "way",
            "osm_id": 1,
            "lon": 2.0,
            "lat": 48.0,
            "wikidata": None,
            "has_wikidata": False,
            "tags": '{"wikipedia":"en:Example"}',
        }
    )
    return row


def test_v2_card_is_viewer_ready_and_documents_v1_comparison(tmp_path: Path) -> None:
    write_v2_region(tmp_path, "region-latest", polygons=[_polygon()], documents=[], links=[])

    card = render_v2_card(tmp_path)

    assert "configs:" in card
    assert "path: polygons/*.parquet" in card
    assert "path: wikipedia/documents/*.parquet" in card
    assert "path: polygon_document_links/*.parquet" in card
    assert "https://github.com/NoeFlandre/osm-polygon-wikidata-only" in card
    assert "## V2 compared with V1" in card
    assert "assets/coverage_map.png" in card
    assert "assets/geographic_text_presence.png" in card
    assert "assets/geographic_text_density.png" in card
    hero = (
        "![NoeFlandre/osm-polygon-wikidata-and-wikipedia dataset overview](assets/dataset_hero.png)"
    )
    assert hero in card
    assert card.index(hero) < card.index("# OSM Polygon Wikidata + Wikipedia, V2")
    assert card.index(hero) < card.index("V2 builds on")
    assert "**Wikipedia-tag-only polygons:** 1" in card
    assert "V2 builds on the [V1 Wikidata-only dataset]" in card
    assert "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only" in card
    assert "not entered manually" not in card
    assert "`wikidata` means the polygon came from an OSM `wikidata=*` tag" in card
    assert "`wikidata_sitelink` means the relationship came from a Wikidata sitelink" in card


def test_v2_card_explains_regional_duplicate_policy(tmp_path: Path) -> None:
    write_v2_region(tmp_path, "region-latest", polygons=[_polygon()], documents=[], links=[])

    card = render_v2_card(tmp_path)

    assert "Regional extracts can overlap" in card
    assert "We keep those copies to preserve regional membership and provenance" in card


def test_v2_card_front_matter_has_viewer_configs_for_every_table(tmp_path: Path) -> None:
    write_v2_region(tmp_path, "region-latest", polygons=[_polygon()], documents=[], links=[])

    front_matter = yaml.safe_load(render_v2_card(tmp_path).split("---", 2)[1])

    assert front_matter["dataset_info"]["version"] == "2.0.0"
    assert front_matter["dataset_contract"] == "wikipedia-tags-v2"
    assert [config["config_name"] for config in front_matter["configs"]] == [
        "polygons",
        "polygon_document_links",
        "wikipedia_documents",
        "wikipedia_sections",
        "wikivoyage_documents",
        "wikivoyage_sections",
        "wikidata_facts",
    ]
    assert all(config["data_files"][0]["split"] for config in front_matter["configs"])


def test_v2_map_assets_render_all_three_views_without_network(tmp_path: Path) -> None:
    write_v2_region(tmp_path, "region-latest", polygons=[_polygon()], documents=[], links=[])
    assets = generate_v2_map_assets(
        tmp_path,
        tmp_path / "assets",
        v1_processed=tmp_path / "v1",
        land_geojson_path=None,
    )

    assert tuple(path.name for path in assets) == (
        "coverage_map.png",
        "geographic_text_presence.png",
        "geographic_text_density.png",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in assets)
    comparison_map = tmp_path / "assets" / "v2_added_wikipedia_tag_documents.png"
    assert comparison_map.is_file() and comparison_map.stat().st_size > 0


def test_v2_map_assets_discovers_sibling_land_cache(tmp_path: Path, monkeypatch) -> None:
    processed_v2 = tmp_path / "processed_v2"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "ne_110m_land.geojson").write_text("{}", encoding="utf-8")
    write_v2_region(processed_v2, "region-latest", polygons=[_polygon()], documents=[], links=[])

    seen: dict[str, Path | None] = {}

    def fake_coverage(_lons, _lats, output_path, *, land_geojson_path, **_kwargs):
        seen["coverage"] = land_geojson_path
        output_path.touch()

    def fake_presence(_processed_root, output_path, *, land_geojson_path, **_kwargs):
        seen["presence"] = land_geojson_path
        output_path.touch()

    def fake_density(_processed_root, output_path, *, land_cache_dir, **_kwargs):
        seen["density"] = land_cache_dir
        output_path.touch()

    monkeypatch.setattr(v2_maps, "generate_coverage_map", fake_coverage)
    monkeypatch.setattr(v2_maps, "generate_geographic_text_presence", fake_presence)
    monkeypatch.setattr(v2_maps, "generate_geographic_text_density", fake_density)

    generate_v2_map_assets(processed_v2, processed_v2 / "assets")

    assert seen == {
        "coverage": cache_dir / "ne_110m_land.geojson",
        "presence": cache_dir / "ne_110m_land.geojson",
        "density": cache_dir,
    }


def test_v2_comparison_map_contains_only_qualifying_v2_polygons(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    v1_polygon_path = v1 / "polygons" / "region-latest.parquet"
    v1_polygon_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([], schema=polygon_schema()), v1_polygon_path)

    polygon = _polygon()
    polygon.update(
        {
            "polygon_id": "region-latest:way:2",
            "discovery_sources": '["wikipedia_tag"]',
        }
    )
    document = {field.name: None for field in wikipedia_document_v2_schema()}
    document["document_id"] = "new-doc"
    write_v2_region(
        v2,
        "region-latest",
        polygons=[polygon],
        documents=[document],
        links=[
            {
                "polygon_id": "region-latest:way:2",
                "document_id": "new-doc",
                "project": "wikipedia",
                "link_sources": '["osm_wikipedia_tag"]',
            }
        ],
    )

    output = generate_v2_added_wikipedia_tag_map(
        v2,
        v1,
        v2 / "assets" / "v2_added_wikipedia_tag_documents.png",
    )

    assert output.is_file()
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
