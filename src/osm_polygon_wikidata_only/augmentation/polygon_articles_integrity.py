"""Validate polygon-article links against canonical polygon identities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.augmentation.integrity_io import (
    read_polygon_wikidata_map,
    read_table_required,
)
from osm_polygon_wikidata_only.augmentation.integrity_models import (
    REASON_POLYGON_ARTICLES_MISMATCH,
    PolygonArticlesIntegrityResult,
    RejectionRecord,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import POLYGON_ARTICLE_COLUMNS
from osm_polygon_wikidata_only.io.parquet import write_polygon_articles


def _partition_polygon_article_rows(
    stem: str,
    rows: list[dict[str, Any]],
    polygon_wikidata: dict[str, str],
) -> tuple[list[dict[str, Any]], list[RejectionRecord], set[str]]:
    """Split polygon-article rows by canonical Wikidata agreement."""
    retained_rows: list[dict[str, Any]] = []
    rejections: list[RejectionRecord] = []
    seen_polygon_ids: set[str] = set()
    for row in rows:
        polygon_id = str(row.get("polygon_id", ""))
        link_wikidata = str(row.get("wikidata", ""))
        seen_polygon_ids.add(polygon_id)
        expected = polygon_wikidata.get(polygon_id)
        if expected is None or expected != link_wikidata:
            rejections.append(
                RejectionRecord(
                    shard=stem,
                    source_table="polygon_articles",
                    identifier=polygon_id,
                    wikidata=link_wikidata,
                    expected=expected,
                    reason=REASON_POLYGON_ARTICLES_MISMATCH,
                )
            )
            continue
        retained_rows.append({column: row.get(column) for column in POLYGON_ARTICLE_COLUMNS})
    return retained_rows, rejections, seen_polygon_ids


def _missing_polygon_ids(seen_polygon_ids: set[str], polygon_wikidata: dict[str, str]) -> list[str]:
    """Return polygon IDs referenced by links but absent from the core table."""
    return sorted(seen_polygon_ids - polygon_wikidata.keys())


def _ensure_polygon_ids_exist(
    stem: str,
    missing_polygon_ids: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Raise when polygon-article rows reference absent core polygons."""
    missing_with_links = sorted(
        polygon_id
        for polygon_id in missing_polygon_ids
        if any(str(row.get("polygon_id", "")) == polygon_id for row in rows)
    )
    if not missing_with_links:
        return
    sample = ", ".join(missing_with_links[:5])
    raise ValueError(
        f"polygon_articles rows reference polygon_id(s) absent from polygons parquet "
        f"for shard {stem!r}: {sample}"
    )


def _write_polygon_articles_if_needed(
    links_path: Path,
    retained_rows: list[dict[str, Any]],
    *,
    rewritten: bool,
) -> None:
    """Rewrite polygon-article rows only when an integrity defect was found."""
    if rewritten:
        write_polygon_articles(links_path, retained_rows)


def enforce_polygon_articles_integrity(
    data_root: DataRoot, stem: str, *, dry_run: bool = False
) -> PolygonArticlesIntegrityResult:
    """Reject polygon-article links whose QID differs from the polygon master.

    Rows are retained only when the polygon exists and its canonical Wikidata
    value matches. A missing polygon reference raises instead of being dropped.
    Successful writes use the canonical polygon-article schema and do not
    rewrite a shard that has no rejected rows.
    """
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    links_path = data_root.processed_links / f"{stem}.parquet"

    polygon_wikidata = read_polygon_wikidata_map(polygons_path)
    table = read_table_required(
        links_path,
        label="polygon_articles",
        columns=POLYGON_ARTICLE_COLUMNS,
    )
    rows = table.to_pylist()
    retained_rows, rejections, seen_polygon_ids = _partition_polygon_article_rows(
        stem, rows, polygon_wikidata
    )
    _ensure_polygon_ids_exist(
        stem,
        _missing_polygon_ids(seen_polygon_ids, polygon_wikidata),
        rows,
    )

    original_count = len(rows)
    retained_count = len(retained_rows)
    rejected_count = len(rejections)
    rewritten = rejected_count > 0
    _write_polygon_articles_if_needed(
        links_path,
        retained_rows,
        rewritten=rewritten and not dry_run,
    )

    rejections_tuple = tuple(
        sorted(rejections, key=lambda record: (record.identifier, record.wikidata))
    )
    return PolygonArticlesIntegrityResult(
        shard=stem,
        original_row_count=original_count,
        retained_row_count=retained_count,
        rejected_row_count=rejected_count,
        rewritten=rewritten,
        rejections=rejections_tuple,
    )
