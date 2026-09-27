from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pytest

from osm_polygon_wikidata_only.pipeline import link_migration
from osm_polygon_wikidata_only.pipeline.link_migration import _StemApplyInputs


def test_reconstruct_missing_manifest_requires_one_polygon_region(tmp_path: Path) -> None:
    inputs = cast(
        _StemApplyInputs,
        SimpleNamespace(
            polygons_table=pa.Table.from_pylist([{"region": "north"}, {"region": "south"}]),
            stem_plan=SimpleNamespace(stem="region-latest"),
        ),
    )

    with pytest.raises(ValueError, match="polygon region is not unique"):
        link_migration._new_processed_entry(inputs, "region-latest.osm.pbf")
