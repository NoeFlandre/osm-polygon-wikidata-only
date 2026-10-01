from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pytest

from osm_polygon_wikidata_only.pipeline._link_migration.artifacts import new_processed_entry
from osm_polygon_wikidata_only.pipeline._link_migration.models import StemApplyInputs


def test_reconstruct_missing_manifest_requires_one_polygon_region(tmp_path: Path) -> None:
    inputs = cast(
        StemApplyInputs,
        SimpleNamespace(
            polygons_table=pa.Table.from_pylist([{"region": "north"}, {"region": "south"}]),
            stem_plan=SimpleNamespace(stem="region-latest"),
        ),
    )

    with pytest.raises(ValueError, match="polygon region is not unique"):
        new_processed_entry(inputs, "region-latest.osm.pbf")


@pytest.mark.parametrize(
    "polygon_row",
    [
        {"unused": "region column absent"},
        {"region": None},
        {"region": ""},
    ],
)
def test_reconstruct_missing_manifest_rejects_an_empty_or_absent_region(
    polygon_row: dict[str, str | None],
) -> None:
    inputs = cast(
        StemApplyInputs,
        SimpleNamespace(
            polygons_table=pa.Table.from_pylist([polygon_row]),
            stem_plan=SimpleNamespace(stem="region-latest"),
        ),
    )

    with pytest.raises(ValueError, match="polygon region is not unique"):
        new_processed_entry(inputs, "region-latest.osm.pbf")


def test_reconstruct_missing_manifest_preserves_the_processed_manifest_contract() -> None:
    inputs = cast(
        StemApplyInputs,
        SimpleNamespace(
            polygons_table=pa.Table.from_pylist([{"region": "north"}, {"region": "north"}]),
            stem_plan=SimpleNamespace(stem="region-latest"),
        ),
    )

    entry = new_processed_entry(inputs, "north-latest.osm.pbf")

    assert set(entry) == {
        "source_pbf",
        "region",
        "polygons_path",
        "wikipedia_documents_path",
        "polygon_articles_path",
        "extraction_version",
        "processed_at",
    }
    assert {key: value for key, value in entry.items() if key != "processed_at"} == {
        "source_pbf": "north-latest.osm.pbf",
        "region": "north",
        "polygons_path": "polygons/region-latest.parquet",
        "wikipedia_documents_path": "wikipedia/documents/region-latest.parquet",
        "polygon_articles_path": "polygon_articles/region-latest.parquet",
        "extraction_version": "link-migration",
    }
    assert datetime.fromisoformat(entry["processed_at"]).tzinfo is not None
