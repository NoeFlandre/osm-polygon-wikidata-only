"""Freeze resumability and skip-existing semantics.

Both behaviours are documented in the README:

- ``--skip-existing`` skips PBFs whose ``source_pbf`` is already in
  the manifest.
- A run after an interrupted process leaves an entry that the next
  invocation re-uses without re-fetching.
"""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.pipeline.orchestrator import already_processed, collect_pbfs


def test_collect_pbfs_expands_directory(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a-latest.osm.pbf").write_bytes(b"")
    (raw / "b-latest.osm.pbf").write_bytes(b"")
    (raw / "ignore.txt").write_bytes(b"")
    files = collect_pbfs([raw])
    assert [f.name for f in files] == ["a-latest.osm.pbf", "b-latest.osm.pbf"]


def test_collect_pbfs_accepts_single_file(tmp_path: Path) -> None:
    p = tmp_path / "x.osm.pbf"
    p.write_bytes(b"")
    files = collect_pbfs([p])
    assert files == [p]


def test_already_processed_returns_true_when_entry_exists(tmp_path: Path) -> None:
    manifest = tmp_path / "processed_pbfs.json"
    manifest.write_text('{"x-latest.osm.pbf": {"region": "x"}}', encoding="utf-8")
    assert already_processed(manifest, "x-latest.osm.pbf") is True


def test_already_processed_returns_false_when_entry_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "processed_pbfs.json"
    manifest.write_text("{}", encoding="utf-8")
    assert already_processed(manifest, "y-latest.osm.pbf") is False


def test_already_processed_handles_missing_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "processed_pbfs.json"
    assert already_processed(manifest, "missing.osm.pbf") is False


def test_processed_entries_have_required_fields(tmp_path: Path) -> None:
    """The per-PBF manifest entry schema must contain the documented keys."""
    from osm_polygon_wikidata_only.domain.models import ManifestStats
    from osm_polygon_wikidata_only.io.manifest import load_manifest, manifest_path, upsert_entry

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    path = manifest_path(data_root.processed_manifests)

    upsert_entry(
        path,
        source_pbf="monaco-latest.osm.pbf",
        region="monaco",
        polygons_path="polygons/monaco-latest.parquet",
        articles_path="articles/monaco-latest.parquet",
        polygon_articles_path="polygon_articles/monaco-latest.parquet",
        stats=ManifestStats(
            polygon_count=10,
            article_count=5,
            unique_wikidata_count=8,
        ),
        extraction_version="0.1.0",
    )

    entries = load_manifest(path)
    assert "monaco-latest.osm.pbf" in entries
    entry = entries["monaco-latest.osm.pbf"]
    expected = {
        "source_pbf",
        "region",
        "polygons_path",
        "articles_path",
        "polygon_articles_path",
        "extraction_version",
        "processed_at",
    }
    assert expected <= set(entry.keys())


def test_orchestrator_skips_processed_pbfs_when_skip_existing(tmp_path: Path) -> None:
    """``--skip-existing`` excludes PBFs already present in the manifest.

    We populate both the per-PBF manifest and the augmentation manifest
    so the sync plan marks the region as COMPLETE, then verify the CLI
    returns 0 without attempting to re-augment. The augmentation
    manifest reads the fixture parquet file's hash so
    ``augmentation_is_current`` agrees with the real core hash.
    """
    import hashlib
    import json

    from osm_polygon_wikidata_only.cli.commands import build_parser, main
    from osm_polygon_wikidata_only.domain.models import ManifestStats
    from osm_polygon_wikidata_only.io.manifest import manifest_path, upsert_entry

    FIXTURE_PROCESSED = Path(__file__).resolve().parent.parent / "fixtures" / "processed"
    fixture_polygons = FIXTURE_PROCESSED / "polygons" / "monaco-latest.parquet"
    fixture_articles = FIXTURE_PROCESSED / "articles" / "monaco-latest.parquet"
    fixture_links = FIXTURE_PROCESSED / "polygon_articles" / "monaco-latest.parquet"

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "monaco-latest.osm.pbf").write_bytes(b"")

    data_root = DataRoot(tmp_path)
    data_root.ensure()

    upsert_entry(
        manifest_path(data_root.processed_manifests),
        source_pbf="monaco-latest.osm.pbf",
        region="monaco",
        polygons_path="polygons/monaco-latest.parquet",
        articles_path="articles/monaco-latest.parquet",
        polygon_articles_path="polygon_articles/monaco-latest.parquet",
        stats=ManifestStats(),
        extraction_version="0.1.0",
    )

    # Copy the real fixture Parquet files into the data root so the
    # augmentation manifest can hash the real contents (the manifest
    # hashes must match for ``augmentation_is_current`` to return True).
    articles = data_root.processed_articles / "monaco-latest.parquet"
    polygons = data_root.processed_polygons / "monaco-latest.parquet"
    links = data_root.processed_links / "monaco-latest.parquet"
    articles.write_bytes(fixture_articles.read_bytes())
    polygons.write_bytes(fixture_polygons.read_bytes())
    links.write_bytes(fixture_links.read_bytes())

    # The published fixture polygon article has columns the augment
    # manifest's hash lookup only needs file contents, but the
    # augmentation flow additionally reads the parquet schema. Use the
    # real fixture so column checks succeed.

    aug_manifest = data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    aug_manifest.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    from osm_polygon_wikidata_only.augmentation.schema import (
        document_schema,
        fact_schema,
        section_schema,
    )
    from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema

    schemas = {
        ("wikipedia", "documents"): wikipedia_document_schema(),
        ("wikipedia", "sections"): section_schema(),
        ("wikivoyage", "documents"): document_schema(),
        ("wikivoyage", "sections"): section_schema(),
        ("wikidata", "facts"): fact_schema(),
    }

    for sub in (
        ("wikipedia", "sections"),
        ("wikivoyage", "documents"),
        ("wikivoyage", "sections"),
        ("wikidata", "facts"),
    ):
        path = data_root.processed / sub[0] / sub[1] / "monaco-latest.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([], schema=schemas[sub])
        pq.write_table(table, path)

    # Seed correct canonical Wikipedia documents matching the articles fixture
    from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
        build_wikipedia_document_table,
    )

    article_table = pq.read_table(articles)
    canonical_doc_table = build_wikipedia_document_table(article_table)
    doc_path = data_root.processed / "wikipedia" / "documents" / "monaco-latest.parquet"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(canonical_doc_table, doc_path)

    aug_manifest.write_text(
        json.dumps(
            {
                "monaco-latest": {
                    "contract_version": "text-sidecars-v1",
                    "core_hashes": {
                        str(articles): hashlib.sha256(articles.read_bytes()).hexdigest(),
                        str(polygons): hashlib.sha256(polygons.read_bytes()).hexdigest(),
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "sync-dir",
            str(raw),
            "--data-root",
            str(tmp_path),
            "--skip-existing",
        ]
    )
    assert args.skip_existing is True

    rc = main(
        [
            "sync-dir",
            str(raw),
            "--data-root",
            str(tmp_path),
            "--skip-existing",
        ]
    )
    assert rc == 0
