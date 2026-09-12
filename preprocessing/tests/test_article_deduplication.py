from pathlib import Path

import duckdb
import pytest
from osm_polygon_wikidata_only_preprocessing.deduplication import (
    articles as articles_module,
)
from osm_polygon_wikidata_only_preprocessing.deduplication.articles import (
    DeduplicationError,
    deduplicate_article_files,
)


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _write_source_files(tmp_path: Path) -> tuple[Path, Path]:
    articles_path = tmp_path / "articles.parquet"
    links_path = tmp_path / "links.parquet"
    connection = duckdb.connect()
    connection.execute(
        """
        CREATE TABLE articles AS
        SELECT * FROM (VALUES
            ('Q1:en:10:100', 'Q1', 'enwiki', 10, 100, 'en', 'Shared page',
             'Entity one', 'first entity', '[]', '{}', 'same-hash'),
            ('Q2:en:10:100', 'Q2', 'enwiki', 10, 100, 'en', 'Shared page',
             'Entity two', 'second entity', '[]', '{}', 'same-hash'),
            ('Q3:fr:20:200', 'Q3', 'frwiki', 20, 200, 'fr', 'Unique page',
             'Entity three', 'third entity', '[]', '{}', 'unique-hash')
        ) AS source(
            article_id, wikidata, site, page_id, revision_id, language, title,
            wikidata_label, wikidata_description, wikidata_aliases,
            wikidata_sitelinks, content_hash
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE links AS
        SELECT * FROM (VALUES
            ('polygon-1', 'Q1:en:10:100', 'Q1'),
            ('polygon-2', 'Q2:en:10:100', 'Q2'),
            ('polygon-3', 'Q3:fr:20:200', 'Q3')
        ) AS source(polygon_id, article_id, wikidata)
        """
    )
    connection.execute(f"COPY articles TO '{_sql_path(articles_path)}' (FORMAT PARQUET)")
    connection.execute(f"COPY links TO '{_sql_path(links_path)}' (FORMAT PARQUET)")
    connection.close()
    return articles_path, links_path


def test_deduplicates_articles_and_preserves_entity_and_polygon_links(tmp_path):
    articles_path, links_path = _write_source_files(tmp_path)
    output_articles = tmp_path / "processed-articles.parquet"
    output_entities = tmp_path / "article-entities.parquet"
    output_links = tmp_path / "processed-links.parquet"

    stats = deduplicate_article_files(
        articles_path,
        links_path,
        output_articles,
        output_entities,
        output_links,
    )

    connection = duckdb.connect()
    articles = connection.execute(
        "SELECT article_id, site, page_id, revision_id FROM read_parquet(?) ORDER BY article_id",
        [str(output_articles)],
    ).fetchall()
    entities = connection.execute(
        "SELECT article_id, original_article_id, wikidata FROM read_parquet(?) "
        "ORDER BY original_article_id",
        [str(output_entities)],
    ).fetchall()
    links = connection.execute(
        "SELECT polygon_id, article_id, wikidata FROM read_parquet(?) ORDER BY polygon_id",
        [str(output_links)],
    ).fetchall()
    article_schema = [
        row[0]
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(output_articles)]
        ).fetchall()
    ]
    entity_schema = [
        row[0]
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(output_entities)]
        ).fetchall()
    ]
    link_schema = [
        row[0]
        for row in connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(output_links)]
        ).fetchall()
    ]
    connection.close()

    assert articles == [
        ("enwiki:10:100", "enwiki", 10, 100),
        ("frwiki:20:200", "frwiki", 20, 200),
    ]
    assert entities == [
        ("enwiki:10:100", "Q1:en:10:100", "Q1"),
        ("enwiki:10:100", "Q2:en:10:100", "Q2"),
        ("frwiki:20:200", "Q3:fr:20:200", "Q3"),
    ]
    assert links == [
        ("polygon-1", "enwiki:10:100", "Q1"),
        ("polygon-2", "enwiki:10:100", "Q2"),
        ("polygon-3", "frwiki:20:200", "Q3"),
    ]
    assert article_schema == [
        "article_id",
        "site",
        "page_id",
        "revision_id",
        "language",
        "title",
        "content_hash",
    ]
    assert entity_schema == [
        "article_id",
        "original_article_id",
        "wikidata",
        "wikidata_label",
        "wikidata_description",
        "wikidata_aliases",
        "wikidata_sitelinks",
    ]
    assert link_schema == ["polygon_id", "article_id", "wikidata"]
    assert stats.input_articles == 3
    assert stats.output_articles == 2
    assert stats.duplicate_articles_removed == 1
    assert stats.entity_mappings == 3
    assert stats.input_links == 3
    assert stats.output_links == 3


def test_rejects_duplicate_revision_with_conflicting_content(tmp_path):
    articles_path, links_path = _write_source_files(tmp_path)
    conflicting_articles = tmp_path / "conflicting-articles.parquet"
    connection = duckdb.connect()
    connection.execute(
        "COPY (SELECT * REPLACE ("
        "CASE WHEN wikidata = 'Q2' THEN 'different-hash' ELSE content_hash END "
        f"AS content_hash) FROM read_parquet('{_sql_path(articles_path)}')) "
        f"TO '{_sql_path(conflicting_articles)}' (FORMAT PARQUET)"
    )
    connection.close()

    with pytest.raises(
        DeduplicationError,
        match="Duplicate article revisions have conflicting content: enwiki:10:100",
    ):
        deduplicate_article_files(
            conflicting_articles,
            links_path,
            tmp_path / "processed-articles.parquet",
            tmp_path / "article-entities.parquet",
            tmp_path / "processed-links.parquet",
        )


def test_rejects_orphan_polygon_article_links(tmp_path):
    articles_path, links_path = _write_source_files(tmp_path)
    orphaned_links = tmp_path / "orphaned-links.parquet"
    connection = duckdb.connect()
    connection.execute(
        "COPY (SELECT * FROM read_parquet("
        f"'{_sql_path(links_path)}') UNION ALL "
        "SELECT 'polygon-4', 'missing-article', 'Q4') "
        f"TO '{_sql_path(orphaned_links)}' (FORMAT PARQUET)"
    )
    connection.close()

    with pytest.raises(
        DeduplicationError,
        match="Input polygon links contain 1 orphan article reference",
    ):
        deduplicate_article_files(
            articles_path,
            orphaned_links,
            tmp_path / "processed-articles.parquet",
            tmp_path / "article-entities.parquet",
            tmp_path / "processed-links.parquet",
        )


def test_refuses_to_overwrite_existing_outputs(tmp_path):
    articles_path, links_path = _write_source_files(tmp_path)
    output_articles = tmp_path / "processed-articles.parquet"
    output_articles.write_bytes(b"user-owned-output")

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing outputs"):
        deduplicate_article_files(
            articles_path,
            links_path,
            output_articles,
            tmp_path / "article-entities.parquet",
            tmp_path / "processed-links.parquet",
        )

    assert output_articles.read_bytes() == b"user-owned-output"


def test_rejects_duplicate_source_article_ids(tmp_path):
    articles_path, links_path = _write_source_files(tmp_path)
    duplicated_articles = tmp_path / "duplicated-articles.parquet"
    connection = duckdb.connect()
    connection.execute(
        "COPY (SELECT * FROM read_parquet("
        f"'{_sql_path(articles_path)}') UNION ALL "
        f"SELECT * FROM read_parquet('{_sql_path(articles_path)}') "
        "WHERE article_id = 'Q1:en:10:100') "
        f"TO '{_sql_path(duplicated_articles)}' (FORMAT PARQUET)"
    )
    connection.close()

    with pytest.raises(DeduplicationError, match="Input article_id values must be unique"):
        deduplicate_article_files(
            duplicated_articles,
            links_path,
            tmp_path / "processed-articles.parquet",
            tmp_path / "article-entities.parquet",
            tmp_path / "processed-links.parquet",
        )


def test_rejects_duplicate_source_polygon_links(tmp_path):
    articles_path, links_path = _write_source_files(tmp_path)
    duplicated_links = tmp_path / "duplicated-links.parquet"
    connection = duckdb.connect()
    connection.execute(
        "COPY (SELECT * FROM read_parquet("
        f"'{_sql_path(links_path)}') UNION ALL "
        f"SELECT * FROM read_parquet('{_sql_path(links_path)}') "
        "WHERE polygon_id = 'polygon-1') "
        f"TO '{_sql_path(duplicated_links)}' (FORMAT PARQUET)"
    )
    connection.close()

    with pytest.raises(
        DeduplicationError,
        match="Input polygon links must be unique by polygon_id and article_id",
    ):
        deduplicate_article_files(
            articles_path,
            duplicated_links,
            tmp_path / "processed-articles.parquet",
            tmp_path / "article-entities.parquet",
            tmp_path / "processed-links.parquet",
        )


def test_rejects_null_canonical_identity_components(tmp_path):
    articles_path, links_path = _write_source_files(tmp_path)
    null_identity_articles = tmp_path / "null-identity-articles.parquet"
    connection = duckdb.connect()
    connection.execute(
        "COPY (SELECT * REPLACE ("
        "CASE WHEN wikidata = 'Q2' THEN NULL ELSE page_id END AS page_id) "
        f"FROM read_parquet('{_sql_path(articles_path)}')) "
        f"TO '{_sql_path(null_identity_articles)}' (FORMAT PARQUET)"
    )
    connection.close()

    with pytest.raises(
        DeduplicationError,
        match="Article identity columns must not be null: site, page_id, revision_id",
    ):
        deduplicate_article_files(
            null_identity_articles,
            links_path,
            tmp_path / "processed-articles.parquet",
            tmp_path / "article-entities.parquet",
            tmp_path / "processed-links.parquet",
        )


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_removes_partial_outputs_when_publishing_fails(tmp_path, monkeypatch, failure_index):
    articles_path, links_path = _write_source_files(tmp_path)
    outputs = (
        tmp_path / "processed-articles.parquet",
        tmp_path / "article-entities.parquet",
        tmp_path / "processed-links.parquet",
    )
    stale_temporary = tmp_path / ".processed-articles.parquet.tmp"
    stale_temporary.write_bytes(b"stale-output")
    original_hardlink_to = Path.hardlink_to
    publications = 0
    staged_paths = []

    def fail_publication(output, temporary):
        nonlocal publications
        publications += 1
        if publications - 1 == failure_index:
            staged_paths.extend(
                path
                for output_path in outputs
                for path in tmp_path.glob(f".{output_path.name}.*.tmp")
            )
            raise OSError(f"simulated publish failure {failure_index}")
        return original_hardlink_to(output, temporary)

    monkeypatch.setattr(Path, "hardlink_to", fail_publication)

    with pytest.raises(OSError, match=f"simulated publish failure {failure_index}"):
        deduplicate_article_files(
            articles_path,
            links_path,
            *outputs,
        )

    assert all(not path.exists() for path in outputs)
    assert staged_paths
    assert all(not path.exists() for path in staged_paths)
    assert stale_temporary.read_bytes() == b"stale-output"


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("article_id", "NULL", "article_id must not be null"),
        ("content_hash", "NULL", "content_hash must not be null"),
        ("site", "'bad:wiki'", "site must not contain ':'"),
    ],
)
def test_rejects_unsafe_article_identity_values(tmp_path, column, replacement, message):
    articles_path, links_path = _write_source_files(tmp_path)
    invalid_articles = tmp_path / f"invalid-{column}.parquet"
    connection = duckdb.connect()
    connection.execute(
        "COPY (SELECT * REPLACE ("
        f"CASE WHEN wikidata = 'Q2' THEN {replacement} ELSE {column} END "
        f"AS {column}) FROM read_parquet('{_sql_path(articles_path)}')) "
        f"TO '{_sql_path(invalid_articles)}' (FORMAT PARQUET)"
    )
    connection.close()

    with pytest.raises(DeduplicationError, match=message):
        deduplicate_article_files(
            invalid_articles,
            links_path,
            tmp_path / "processed-articles.parquet",
            tmp_path / "article-entities.parquet",
            tmp_path / "processed-links.parquet",
        )


def test_ignores_unrelated_stale_temporary_files(tmp_path):
    articles_path, links_path = _write_source_files(tmp_path)
    stale_temporary = tmp_path / ".processed-articles.parquet.tmp"
    stale_temporary.write_bytes(b"stale-output")

    stats = deduplicate_article_files(
        articles_path,
        links_path,
        tmp_path / "processed-articles.parquet",
        tmp_path / "article-entities.parquet",
        tmp_path / "processed-links.parquet",
    )

    assert stats.output_articles == 2
    assert stale_temporary.read_bytes() == b"stale-output"


def test_does_not_overwrite_output_created_after_preflight(tmp_path, monkeypatch):
    articles_path, links_path = _write_source_files(tmp_path)
    outputs = (
        tmp_path / "processed-articles.parquet",
        tmp_path / "article-entities.parquet",
        tmp_path / "processed-links.parquet",
    )
    original_check = articles_module._require_new_outputs

    def create_competing_output(paths):
        original_check(paths)
        outputs[0].write_bytes(b"concurrent-output")

    monkeypatch.setattr(
        articles_module,
        "_require_new_outputs",
        create_competing_output,
    )

    with pytest.raises(FileExistsError):
        deduplicate_article_files(
            articles_path,
            links_path,
            *outputs,
        )

    assert outputs[0].read_bytes() == b"concurrent-output"
    assert all(not path.exists() for path in outputs[1:])


def test_deduplication_accepts_paths_with_apostrophes(tmp_path):
    source_root = tmp_path / "operator's data"
    source_root.mkdir()
    articles_path, links_path = _write_source_files(source_root)

    stats = deduplicate_article_files(
        articles_path,
        links_path,
        source_root / "processed-articles.parquet",
        source_root / "article-entities.parquet",
        source_root / "processed-links.parquet",
    )

    assert stats.output_articles == 2
