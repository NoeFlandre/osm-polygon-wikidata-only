from shutil import copyfile

from osm_polygon_wikidata_only_preprocessing.cli import main
from osm_polygon_wikidata_only_preprocessing.deduplication.articles import (
    DeduplicationError,
)
from test_article_deduplication import _write_source_files


def test_main_without_arguments_prints_help(capsys):
    exit_code = main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "usage:" in output
    assert "osm-polygon-wikidata-only-preprocessing" in output


def test_deduplicate_articles_command_runs_pipeline(tmp_path, capsys):
    source_articles, source_links = _write_source_files(tmp_path)
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    (raw_root / "articles").mkdir(parents=True)
    (raw_root / "polygon_articles").mkdir(parents=True)
    copyfile(source_articles, raw_root / "articles/example-latest.parquet")
    copyfile(source_links, raw_root / "polygon_articles/example-latest.parquet")

    exit_code = main(
        [
            "deduplicate-articles",
            "--raw-root",
            str(raw_root),
            "--output-root",
            str(output_root),
            "--region-stem",
            "example-latest",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Articles: 3 -> 2 (1 duplicate removed)" in output
    assert "Entity mappings: 3" in output
    assert "Polygon links: 3 -> 3" in output


def test_deduplicate_articles_command_reports_validation_errors(monkeypatch, capsys):
    def reject_deduplication(*args):
        raise DeduplicationError("invalid article data")

    monkeypatch.setattr(
        "osm_polygon_wikidata_only_preprocessing.cli.deduplicate_region",
        reject_deduplication,
    )

    exit_code = main(
        [
            "deduplicate-articles",
            "--raw-root",
            "raw",
            "--output-root",
            "processed",
            "--region-stem",
            "example-latest",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "error: invalid article data\n"
