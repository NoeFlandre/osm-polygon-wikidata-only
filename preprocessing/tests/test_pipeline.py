from shutil import copyfile

import pytest
from osm_polygon_wikidata_only_preprocessing.deduplication.pipeline import (
    deduplicate_region,
)
from test_article_deduplication import _write_source_files


def test_deduplicate_region_uses_layered_data_paths(tmp_path):
    source_articles, source_links = _write_source_files(tmp_path)
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    (raw_root / "articles").mkdir(parents=True)
    (raw_root / "polygon_articles").mkdir(parents=True)
    copyfile(source_articles, raw_root / "articles/example-latest.parquet")
    copyfile(source_links, raw_root / "polygon_articles/example-latest.parquet")

    stats = deduplicate_region(raw_root, processed_root, "example-latest")

    assert stats.output_articles == 2
    assert (processed_root / "articles/example-latest.parquet").is_file()
    assert (processed_root / "article_entities/example-latest.parquet").is_file()
    assert (processed_root / "polygon_articles/example-latest.parquet").is_file()


def test_deduplicate_region_rejects_path_components(tmp_path):
    with pytest.raises(ValueError, match="region_stem must be a filename stem"):
        deduplicate_region(tmp_path / "raw", tmp_path / "processed", "../escape")
