from __future__ import annotations

from pathlib import Path

import yaml

from osm_polygon_wikidata_only.domain.schema import empty_row, polygon_schema
from osm_polygon_wikidata_only.v2.card import render_v2_card
from osm_polygon_wikidata_only.v2.maps import generate_v2_map_assets
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
    assert "assets/dataset_hero.png" not in card
    assert "**Wikipedia-tag-only polygons:** 1" in card
    assert "V2 builds on the [V1 Wikidata-only dataset]" in card
    assert "https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-only" in card
    assert "not entered manually" not in card
    assert "`wikidata` means the polygon came from an OSM `wikidata=*` tag" in card
    assert "`wikidata_sitelink` means the relationship came from a Wikidata sitelink" in card


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
    assets = generate_v2_map_assets(tmp_path, tmp_path / "assets", land_geojson_path=None)

    assert tuple(path.name for path in assets) == (
        "coverage_map.png",
        "geographic_text_presence.png",
        "geographic_text_density.png",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in assets)
