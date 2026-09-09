from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import duckdb

from .identity import CANONICAL_ARTICLE_ID_SQL


class DeduplicationError(ValueError):
    pass


@dataclass(frozen=True)
class DeduplicationStats:
    input_articles: int
    output_articles: int
    duplicate_articles_removed: int
    entity_mappings: int
    input_links: int
    output_links: int


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _sql_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _require_new_outputs(paths: tuple[Path, ...]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {formatted}")


def _cleanup_outputs(paths: tuple[Path, ...]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def deduplicate_article_files(
    articles_path: Path,
    links_path: Path,
    output_articles_path: Path,
    output_entities_path: Path,
    output_links_path: Path,
) -> DeduplicationStats:
    input_paths = (articles_path, links_path)
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_paths = (
        output_articles_path,
        output_entities_path,
        output_links_path,
    )
    staging_id = uuid4().hex
    temporary_paths = tuple(
        path.with_name(f".{path.name}.{staging_id}.tmp") for path in output_paths
    )
    _require_new_outputs(output_paths + temporary_paths)

    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            CREATE TEMP TABLE source_articles AS
            SELECT *, {CANONICAL_ARTICLE_ID_SQL} AS canonical_id
            FROM read_parquet('{_sql_path(articles_path)}')
            """
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE source_links AS
            SELECT * FROM read_parquet('{_sql_path(links_path)}')
            """
        )
        source_columns = [
            row[0] for row in connection.execute("DESCRIBE source_articles").fetchall()
        ]
        wikidata_columns = [
            column
            for column in source_columns
            if column == "wikidata" or column.startswith("wikidata_")
        ]
        canonical_exclusions = ", ".join(
            _sql_identifier(column)
            for column in ("article_id", *wikidata_columns, "canonical_id")
        )
        entity_projection = ", ".join(
            _sql_identifier(column) for column in wikidata_columns
        )

        (
            null_article_ids,
            null_identity_components,
            null_content_hashes,
            invalid_sites,
        ) = connection.execute(
            """
            SELECT
                count(*) FILTER (WHERE article_id IS NULL),
                count(*) FILTER (
                    WHERE site IS NULL OR page_id IS NULL OR revision_id IS NULL
                ),
                count(*) FILTER (WHERE content_hash IS NULL),
                count(*) FILTER (WHERE contains(site, ':'))
            FROM source_articles
            """
        ).fetchone()
        if null_article_ids:
            raise DeduplicationError("article_id must not be null")
        if null_identity_components:
            raise DeduplicationError(
                "Article identity columns must not be null: site, page_id, revision_id"
            )
        if null_content_hashes:
            raise DeduplicationError("content_hash must not be null")
        if invalid_sites:
            raise DeduplicationError("site must not contain ':'")

        duplicate_ids = connection.execute(
            """
            SELECT count(*) - count(DISTINCT article_id)
            FROM source_articles
            """
        ).fetchone()[0]
        if duplicate_ids:
            raise DeduplicationError("Input article_id values must be unique")

        duplicate_links = connection.execute(
            """
            SELECT count(*) - count(DISTINCT (polygon_id, article_id))
            FROM source_links
            """
        ).fetchone()[0]
        if duplicate_links:
            raise DeduplicationError(
                "Input polygon links must be unique by polygon_id and article_id"
            )

        conflicts = connection.execute(
            """
            SELECT canonical_id
            FROM source_articles
            GROUP BY canonical_id
            HAVING count(DISTINCT content_hash) > 1
            ORDER BY canonical_id
            """
        ).fetchall()
        if conflicts:
            identifiers = ", ".join(row[0] for row in conflicts)
            raise DeduplicationError(
                f"Duplicate article revisions have conflicting content: {identifiers}"
            )

        orphan_links = connection.execute(
            """
            SELECT count(*)
            FROM source_links AS links
            LEFT JOIN source_articles AS articles USING (article_id)
            WHERE articles.article_id IS NULL
            """
        ).fetchone()[0]
        if orphan_links:
            noun = "reference" if orphan_links == 1 else "references"
            raise DeduplicationError(
                f"Input polygon links contain {orphan_links} orphan article {noun}"
            )

        connection.execute(
            f"""
            CREATE TEMP VIEW canonical_articles AS
            SELECT
                canonical_id AS article_id,
                * EXCLUDE ({canonical_exclusions})
            FROM source_articles
            QUALIFY row_number() OVER (
                PARTITION BY canonical_id ORDER BY article_id
            ) = 1
            """
        )
        connection.execute(
            f"""
            CREATE TEMP VIEW article_entities AS
            SELECT
                canonical_id AS article_id,
                article_id AS original_article_id,
                {entity_projection}
            FROM source_articles
            """
        )
        connection.execute(
            """
            CREATE TEMP VIEW remapped_links AS
            SELECT
                links.polygon_id,
                articles.canonical_id AS article_id,
                links.* EXCLUDE (polygon_id, article_id)
            FROM source_links AS links
            JOIN source_articles AS articles USING (article_id)
            """
        )

        (
            input_articles,
            output_articles,
            entity_mappings,
            input_links,
            output_links,
        ) = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM source_articles),
                (SELECT count(*) FROM canonical_articles),
                (SELECT count(*) FROM article_entities),
                (SELECT count(*) FROM source_links),
                (SELECT count(*) FROM remapped_links)
            """
        ).fetchone()

        for path in output_paths:
            path.parent.mkdir(parents=True, exist_ok=True)

        connection.execute(
            f"""
            COPY (SELECT * FROM canonical_articles ORDER BY article_id)
            TO '{_sql_path(temporary_paths[0])}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        connection.execute(
            f"""
            COPY (
                SELECT * FROM article_entities
                ORDER BY article_id, original_article_id
            )
            TO '{_sql_path(temporary_paths[1])}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        connection.execute(
            f"""
            COPY (
                SELECT * FROM remapped_links
                ORDER BY polygon_id, article_id, wikidata
            )
            TO '{_sql_path(temporary_paths[2])}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    except Exception:
        _cleanup_outputs(temporary_paths)
        raise
    finally:
        connection.close()

    published_paths: list[Path] = []
    try:
        for temporary_path, output_path in zip(
            temporary_paths, output_paths, strict=True
        ):
            output_path.hardlink_to(temporary_path)
            published_paths.append(output_path)
        _cleanup_outputs(temporary_paths)
    except Exception:
        _cleanup_outputs(temporary_paths + tuple(published_paths))
        raise

    return DeduplicationStats(
        input_articles=input_articles,
        output_articles=output_articles,
        duplicate_articles_removed=input_articles - output_articles,
        entity_mappings=entity_mappings,
        input_links=input_links,
        output_links=output_links,
    )
