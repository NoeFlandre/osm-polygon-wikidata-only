from pathlib import Path

import pytest

from osm_polygon_wikidata_only.v2.publication import region_publication_ops


def test_region_publication_requires_wikipedia_sections(tmp_path: Path) -> None:
    stem = "region-latest"
    for relative in (
        f"polygons/{stem}.parquet",
        f"wikipedia/documents/{stem}.parquet",
        f"polygon_document_links/{stem}.parquet",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    with pytest.raises(FileNotFoundError, match="wikipedia/sections"):
        region_publication_ops(tmp_path, stem)
