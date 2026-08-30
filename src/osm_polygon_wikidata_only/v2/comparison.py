"""V2-only selections used by the public V1 comparison artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, MutableMapping
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.storage import load_v2_manifest

_BATCH_SIZE = 65_536


def select_v2_added_wikipedia_tag_document_polygon_ids(
    processed_v2: Path,
    v1_processed: Path,
) -> set[str]:
    """Return polygon IDs added by V2's direct Wikipedia-tag route.

    A selected polygon must be absent from V1, have exactly the
    ``wikipedia_tag`` discovery source, and link to a V2 Wikipedia document
    identity that is absent from V1 through ``osm_wikipedia_tag``.
    """
    v2_polygon_files = _manifest_files(processed_v2, "polygons")
    v2_document_files = _manifest_files(processed_v2, "wikipedia/documents")
    v2_link_files = _manifest_files(processed_v2, "polygon_document_links")
    v1_polygon_files = tuple(sorted((v1_processed / "polygons").glob("*.parquet")))
    v1_document_files = _v1_document_files(v1_processed)

    v2_polygon_ids = _unique_values(v2_polygon_files, "polygon_id")
    new_polygon_ids = v2_polygon_ids - _unique_values(v1_polygon_files, "polygon_id")
    v2_document_ids = _unique_values(v2_document_files, "document_id")
    v1_document_ids = _unique_values(v1_document_files, "document_id")
    if not v1_document_ids:
        v1_document_ids = _unique_values(v1_document_files, "article_id")
    new_document_ids = v2_document_ids - v1_document_ids

    return select_v2_added_wikipedia_tag_document_polygon_ids_from_files(
        v2_polygon_files,
        v2_link_files,
        new_polygon_ids=new_polygon_ids,
        new_document_ids=new_document_ids,
    )


def select_v2_added_wikipedia_tag_document_polygon_ids_from_files(
    v2_polygon_files: Iterable[Path],
    v2_link_files: Iterable[Path],
    *,
    new_polygon_ids: set[str],
    new_document_ids: set[str],
) -> set[str]:
    """Select qualifying IDs from already-discovered comparison file sets."""
    sources = _polygon_sources(v2_polygon_files, new_polygon_ids)
    direct_documents = _direct_documents_by_polygon(v2_link_files)
    return {
        polygon_id
        for polygon_id, discovery_sources in sources.items()
        if discovery_sources == {"wikipedia_tag"}
        and direct_documents.get(polygon_id, set()) & new_document_ids
    }


def _manifest_files(processed_v2: Path, subdir: str) -> tuple[Path, ...]:
    stems = sorted(load_v2_manifest(processed_v2))
    return tuple(
        path for stem in stems if (path := processed_v2 / subdir / f"{stem}.parquet").is_file()
    )


def _v1_document_files(processed: Path) -> tuple[Path, ...]:
    wikipedia = tuple(sorted((processed / "wikipedia/documents").glob("*.parquet")))
    if wikipedia:
        return wikipedia
    return tuple(sorted((processed / "articles").glob("*.parquet")))


def _unique_values(paths: Iterable[Path], column: str) -> set[str]:
    values: set[str] = set()
    for path in paths:
        with pq.ParquetFile(path) as parquet_file:
            if column not in parquet_file.schema_arrow.names:
                continue
            for batch in parquet_file.iter_batches(columns=[column], batch_size=_BATCH_SIZE):
                values.update(_values_from_batch(batch))
    return values


def _values_from_batch(batch: Any) -> set[str]:
    return {str(value) for value in batch.column(0).to_pylist() if value not in (None, "")}


def _polygon_sources(paths: Iterable[Path], identities: set[str]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    if not identities:
        return values
    for path in paths:
        for batch in _iter_polygon_source_batches(path):
            for identity, raw_sources in zip(
                batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
            ):
                _record_polygon_source(values, identity, raw_sources, identities, path)
    return values


def _iter_polygon_source_batches(path: Path) -> Iterator[Any]:
    with pq.ParquetFile(path) as parquet_file:
        if not {"polygon_id", "discovery_sources"}.issubset(parquet_file.schema_arrow.names):
            return
        yield from parquet_file.iter_batches(
            columns=["polygon_id", "discovery_sources"], batch_size=_BATCH_SIZE
        )


def _record_polygon_source(
    values: MutableMapping[str, set[str]],
    identity: Any,
    raw_sources: Any,
    identities: set[str],
    path: Path,
) -> None:
    if identity not in identities:
        return
    values.setdefault(str(identity), set()).update(
        _source_list(raw_sources, identity, path, "discovery_sources")
    )


def _direct_documents_by_polygon(paths: Iterable[Path]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        for batch in _iter_direct_document_batches(path):
            for polygon_id, document_id, project, raw_sources in zip(
                *(batch.column(index).to_pylist() for index in range(4)), strict=True
            ):
                _record_direct_document(
                    values,
                    polygon_id,
                    document_id,
                    project,
                    raw_sources,
                    path,
                )
    return dict(values)


def _iter_direct_document_batches(path: Path) -> Iterator[Any]:
    required = {"polygon_id", "document_id", "project", "link_sources"}
    with pq.ParquetFile(path) as parquet_file:
        if not required.issubset(parquet_file.schema_arrow.names):
            return
        yield from parquet_file.iter_batches(
            columns=["polygon_id", "document_id", "project", "link_sources"],
            batch_size=_BATCH_SIZE,
        )


def _record_direct_document(
    values: MutableMapping[str, set[str]],
    polygon_id: Any,
    document_id: Any,
    project: Any,
    raw_sources: Any,
    path: Path,
) -> None:
    if not polygon_id or not document_id or project != "wikipedia":
        return
    sources = _source_list(raw_sources, polygon_id, path, "link_sources")
    if "osm_wikipedia_tag" in sources:
        values.setdefault(str(polygon_id), set()).add(str(document_id))


def _source_list(raw_sources: Any, identity: Any, path: Path, field: str) -> list[str]:
    try:
        parsed = json_loads(str(raw_sources or "[]"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} for identity {identity!r} in {path}") from error
    return _validated_source_list(parsed, identity, path, field)


def _validated_source_list(
    parsed: object,
    identity: Any,
    path: Path,
    field: str,
) -> list[str]:
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return cast(list[str], parsed)
    raise ValueError(f"Invalid {field} for identity {identity!r} in {path}")


__all__ = [
    "select_v2_added_wikipedia_tag_document_polygon_ids",
    "select_v2_added_wikipedia_tag_document_polygon_ids_from_files",
]
