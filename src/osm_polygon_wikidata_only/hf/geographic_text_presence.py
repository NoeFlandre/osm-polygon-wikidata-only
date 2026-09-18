"""Factual polygon coverage by non-empty Wikipedia or Wikivoyage text."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet

from ._geographic.models import CoverageMapError, RenderResult
from ._geographic.parquet_inputs import require_directory, sorted_parquets
from ._geographic.polygon_identities import (
    PolygonIdentity,
    PolygonIndex,
    PolygonRecord,
    load_unique_polygon_records,
)
from ._links.reader import DocumentLink, is_canonical_link_schema, read_document_links
from .coverage_map import generate_coverage_map


@dataclass(frozen=True, slots=True)
class CoveredPoint:
    polygon_id: str
    wikidata: str
    lon: float
    lat: float
    identity: PolygonIdentity = ("legacy", "")


@dataclass(frozen=True, slots=True)
class TextPresenceSnapshot:
    polygon_count: int
    wikipedia_covered_polygon_ids: frozenset[str]
    combined_covered_polygon_ids: frozenset[str]
    wikipedia_document_ids: frozenset[str]
    wikivoyage_document_ids: frozenset[str]
    covered_points: tuple[CoveredPoint, ...]
    wikipedia_polygon_identities: frozenset[PolygonIdentity] = frozenset()
    combined_polygon_identities: frozenset[PolygonIdentity] = frozenset()


_PRESENCE_CACHE: dict[tuple[Path, Path | None], tuple[tuple[Any, ...], TextPresenceSnapshot]] = {}
# The sync runner drives publication from worker threads, so the cache is
# guarded. The lock covers only the dictionary access and never the scan, so
# callers for different roots are not serialized behind one another.
_PRESENCE_CACHE_LOCK = threading.Lock()


def _document_identity_column(names: set[str] | list[str] | tuple[str, ...]) -> str:
    return "document_id" if "document_id" in names else "article_id"


def load_text_presence(
    processed_root: Path,
    *,
    links_dir: Path | None = None,
) -> TextPresenceSnapshot:
    """Return the text-presence snapshot, reusing an identical recent scan.

    One publication asks for this snapshot several times -- the card
    headline, the continent table, and each coverage map all need it --
    and every call would otherwise re-read the whole document corpus.
    The result is memoised against a fingerprint of the input
    directories, so repeated calls within a run are free while any
    change on disk produces a fresh scan. Sharing one snapshot also
    guarantees the figures those callers publish cannot disagree.
    """
    key = (
        processed_root.resolve(),
        links_dir.resolve() if links_dir is not None else None,
    )
    fingerprint = _presence_fingerprint(key[0], key[1])
    with _PRESENCE_CACHE_LOCK:
        cached = _PRESENCE_CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    snapshot = _load_text_presence_uncached(processed_root, links_dir=links_dir)
    with _PRESENCE_CACHE_LOCK:
        _PRESENCE_CACHE[key] = (fingerprint, snapshot)
    return snapshot


def _presence_fingerprint(processed_root: Path, links_dir: Path | None) -> tuple[Any, ...]:
    """Summarise the inputs cheaply enough to validate a cached scan."""
    directories = [
        processed_root / "polygons",
        _wikipedia_documents_directory(processed_root),
        links_dir or processed_root / "polygon_articles",
        processed_root / "wikivoyage" / "documents",
    ]
    return tuple(_directory_fingerprint(directory) for directory in directories)


def _directory_fingerprint(directory: Path) -> tuple[int, int, int]:
    if not directory.is_dir():
        return (0, 0, 0)
    count = 0
    total = 0
    newest = 0
    for path in directory.glob("*.parquet"):
        stat = path.stat()
        count += 1
        total += stat.st_size
        newest = max(newest, stat.st_mtime_ns)
    return (count, total, newest)


def _load_text_presence_uncached(
    processed_root: Path,
    *,
    links_dir: Path | None = None,
) -> TextPresenceSnapshot:
    """Load exact Wikipedia and combined text coverage from canonical tables.

    V1 callers use the default ``polygon_articles`` directory.  Isolated
    contracts may provide their own unified link directory.
    """
    polygons_dir, wikipedia_dir, source_links_dir, wikivoyage_dir = _presence_directories(
        processed_root, links_dir
    )

    polygon_index = load_unique_polygon_records(sorted_parquets(polygons_dir))
    document_ids, legacy_wikivoyage_qids = _text_document_ids(wikipedia_dir, wikivoyage_dir)
    links = read_document_links(processed_root, links_dir=source_links_dir)
    wikipedia_polygons, combined_ids, unresolved = _linked_text_polygon_ids(
        links,
        document_ids,
        polygon_index,
    )
    _merge_legacy_wikivoyage_ids(
        combined_ids,
        polygon_index,
        legacy_wikivoyage_qids,
        source_links_dir,
    )
    _validate_link_polygon_ids(unresolved, polygon_index.by_polygon_id)
    return _presence_snapshot(
        polygon_index,
        wikipedia_polygons,
        combined_ids,
        document_ids,
    )


def _presence_directories(
    processed_root: Path,
    links_dir: Path | None,
) -> tuple[Path, Path, Path, Path]:
    polygons_dir = require_directory(processed_root / "polygons", label="polygons")
    wikipedia_dir = require_directory(
        _wikipedia_documents_directory(processed_root),
        label="wikipedia/documents",
    )
    source_links_dir = links_dir or processed_root / "polygon_articles"
    require_directory(source_links_dir, label="polygon links")
    return (
        polygons_dir,
        wikipedia_dir,
        source_links_dir,
        processed_root / "wikivoyage" / "documents",
    )


def _wikipedia_documents_directory(processed_root: Path) -> Path:
    canonical = processed_root / "wikipedia" / "documents"
    return canonical if canonical.exists() else processed_root / "articles"


def _text_document_ids(
    wikipedia_dir: Path,
    wikivoyage_dir: Path,
) -> tuple[dict[str, set[str]], set[str]]:
    wikipedia_ids = _wikipedia_text_ids(wikipedia_dir)
    wikivoyage_ids, legacy_wikivoyage_qids = _wikivoyage_text_ids(wikivoyage_dir)
    return {
        "wikipedia": wikipedia_ids,
        "wikivoyage": wikivoyage_ids,
    }, legacy_wikivoyage_qids


def _merge_legacy_wikivoyage_ids(
    combined_ids: set[PolygonIdentity],
    polygon_index: PolygonIndex,
    legacy_wikivoyage_qids: set[str],
    source_links_dir: Path,
) -> None:
    if _has_canonical_links(source_links_dir):
        return
    combined_ids.update(
        identity
        for identity, record in polygon_index.records.items()
        if record.wikidata in legacy_wikivoyage_qids
    )


def _presence_snapshot(
    polygon_index: PolygonIndex,
    wikipedia_polygons: set[PolygonIdentity],
    combined_ids: set[PolygonIdentity],
    document_ids: dict[str, set[str]],
) -> TextPresenceSnapshot:
    points = tuple(
        _covered_point_from_record(polygon_index.records[identity])
        for identity in sorted(combined_ids, key=_identity_sort_key)
    )
    return TextPresenceSnapshot(
        polygon_count=len(polygon_index.records),
        wikipedia_covered_polygon_ids=frozenset(
            polygon_index.records[identity].polygon_id for identity in wikipedia_polygons
        ),
        combined_covered_polygon_ids=frozenset(
            polygon_index.records[identity].polygon_id for identity in combined_ids
        ),
        wikipedia_document_ids=frozenset(document_ids["wikipedia"]),
        wikivoyage_document_ids=frozenset(document_ids["wikivoyage"]),
        covered_points=points,
        wikipedia_polygon_identities=frozenset(wikipedia_polygons),
        combined_polygon_identities=frozenset(combined_ids),
    )


def _identity_sort_key(identity: PolygonIdentity) -> tuple[str, str]:
    return identity[0], str(identity[1])


def _wikipedia_text_ids(wikipedia_dir: Path) -> set[str]:
    values: set[str] = set()
    for path in sorted_parquets(wikipedia_dir):
        values.update(_wikipedia_file_text_ids(path))
    return values


def _wikipedia_file_text_ids(path: Path) -> set[str]:
    return _successful_text_ids(path, label="wikipedia")[0]


def _successful_text_ids(
    path: Path,
    *,
    label: str,
    with_wikidata: bool = False,
) -> tuple[set[str], set[str]]:
    """Return identifiers (and optionally QIDs) of successful non-empty rows.

    The filter runs as Arrow compute kernels over each record batch rather
    than as a Python loop over materialized rows. ``full_text`` is by far
    the largest column in these tables and is needed only to test whether
    the trimmed value is non-empty, so it is never decoded into Python
    objects -- only the surviving identifiers are.
    """
    names = set(pq.read_schema(path).names)
    identifier_column = _document_identity_column(names)
    _require_text_columns(path, label, identifier_column, names)
    columns = _scan_columns(identifier_column, names, with_wikidata=with_wikidata)
    identifiers: set[str] = set()
    qids: set[str] = set()
    _scan_text_batches(
        path,
        label,
        columns,
        lambda batch: _collect_successful_batch(batch, identifier_column, identifiers, qids),
    )
    return identifiers, qids


def _require_text_columns(
    path: Path,
    label: str,
    identifier_column: str,
    names: set[str],
) -> None:
    missing = sorted({identifier_column, "full_text"} - names)
    if missing:
        raise CoverageMapError(f"{label} parquet {path} is missing required columns: {missing}")


def _scan_columns(
    identifier_column: str,
    names: set[str],
    *,
    with_wikidata: bool,
) -> list[str]:
    optional = {"fetch_status"} | ({"wikidata"} if with_wikidata else set())
    return [identifier_column, "full_text", *sorted(optional & names)]


def _scan_text_batches(
    path: Path,
    label: str,
    columns: list[str],
    consume: Callable[[pa.RecordBatch], None],
) -> None:
    """Feed every record batch of ``path`` to ``consume``.

    The reader is driven inside one frame rather than exposed as a
    generator. A generator that holds ``ParquetFile`` open across a yield
    leaves Arrow's prefetch threads queued against a reader that may be
    closed early if the caller stops consuming, which can deadlock.
    """
    try:
        with open_parquet(path) as parquet_file:
            for batch in iter_record_batches(parquet_file, batch_size=65_536, columns=columns):
                consume(batch)
    except OSError as error:
        raise CoverageMapError(f"Could not read {label} parquet {path}: {error}") from error
    except pa.ArrowInvalid as error:
        raise CoverageMapError(
            f"{label} parquet {path} could not be read as columns {columns}"
        ) from error


def _collect_successful_batch(
    batch: pa.RecordBatch,
    identifier_column: str,
    identifiers: set[str],
    qids: set[str],
) -> None:
    selected = batch.filter(_successful_row_mask(batch))
    if selected.num_rows == 0:
        return
    identifiers.update(_column_values(selected, identifier_column))
    qids.update(_column_values(selected, "wikidata"))


def _successful_row_mask(batch: pa.RecordBatch) -> Any:
    """Build the non-empty-text (and fetch_status=ok) mask for one batch.

    The emptiness test trims ``full_text`` itself rather than reading the
    cheaper ``article_length_chars`` column. That column records
    ``len(text)`` on the untrimmed string, so a whitespace-only document
    would report a positive length; the two agree on every row published
    today only because no such row exists, which is an accident of the
    current corpus and not a contract worth depending on.

    Generated PyArrow kernels are reached through ``call_function`` so the
    module stays statically checkable, matching the existing convention.
    """
    trimmed = pc.call_function("utf8_trim_whitespace", [batch.column("full_text")])
    lengths = pc.call_function("utf8_length", [trimmed])
    non_empty = pc.call_function("greater", [lengths, pa.scalar(0, type=pa.int32())])
    mask = pc.fill_null(non_empty, False)
    if "fetch_status" in batch.schema.names:
        is_ok = pc.call_function("equal", [batch.column("fetch_status"), pa.scalar("ok")])
        mask = pc.call_function("and_kleene", [mask, pc.fill_null(is_ok, False)])
    return mask


def _column_values(batch: pa.RecordBatch, column: str) -> set[str]:
    """Return the non-empty string values of ``column`` when it is present."""
    if column not in batch.schema.names:
        return set()
    return {str(value) for value in batch.column(column).to_pylist() if value}


def _wikivoyage_text_ids(directory: Path) -> tuple[set[str], set[str]]:
    document_ids: set[str] = set()
    qids: set[str] = set()
    for path in sorted_parquets(directory):
        file_ids, file_qids = _wikivoyage_file_text_ids(path)
        document_ids.update(file_ids)
        qids.update(file_qids)
    return document_ids, qids


def _wikivoyage_file_text_ids(path: Path) -> tuple[set[str], set[str]]:
    return _successful_text_ids(path, label="wikivoyage", with_wikidata=True)


def _has_canonical_links(source_links_dir: Path) -> bool:
    return any(
        is_canonical_link_schema(pq.read_schema(path)) for path in sorted_parquets(source_links_dir)
    )


def _linked_text_polygon_ids(
    links: tuple[DocumentLink, ...],
    document_ids: dict[str, set[str]],
    polygon_index: PolygonIndex,
) -> tuple[set[PolygonIdentity], set[PolygonIdentity], set[str]]:
    wikipedia_ids: set[PolygonIdentity] = set()
    combined_ids: set[PolygonIdentity] = set()
    unresolved: set[str] = set()
    for link in links:
        if link.document_id not in document_ids.get(link.project, set()):
            continue
        identity = polygon_index.by_polygon_id.get(link.polygon_id)
        if identity is None:
            unresolved.add(link.polygon_id)
            continue
        combined_ids.add(identity)
        if link.project == "wikipedia":
            wikipedia_ids.add(identity)
    return wikipedia_ids, combined_ids, unresolved


def _covered_point_from_record(record: PolygonRecord) -> CoveredPoint:
    if record.lon is None or record.lat is None:
        raise CoverageMapError(
            f"polygons parquet {record.source_path} has invalid lat/lon coordinates "
            f"for {record.polygon_id}"
        )
    return CoveredPoint(
        record.polygon_id,
        record.wikidata,
        record.lon,
        record.lat,
        record.identity,
    )


def _validate_link_polygon_ids(
    unresolved: set[str],
    polygon_ids: dict[str, PolygonIdentity],
) -> None:
    missing = unresolved - polygon_ids.keys()
    if missing:
        raise CoverageMapError(
            f"polygon_articles contains {len(missing)} polygon id(s) absent from polygons"
        )


def generate_geographic_text_presence(
    processed_root: Path,
    output_path: Path,
    *,
    land_geojson_path: Path | None = None,
    snapshot: TextPresenceSnapshot | None = None,
) -> RenderResult:
    """Render one point for every polygon with Wikipedia or Wikivoyage text."""
    snapshot = snapshot or load_text_presence(processed_root)
    points = snapshot.covered_points
    generate_coverage_map(
        [point.lon for point in points],
        [point.lat for point in points],
        output_path,
        land_geojson_path=land_geojson_path,
        title="Polygons with Wikipedia or Wikivoyage text",
        point_color="#2563EB",
        point_edge="#1E40AF",
    )
    rate = len(points) / snapshot.polygon_count if snapshot.polygon_count else 0.0
    caption = (
        f"{len(points):,} of {snapshot.polygon_count:,} unique (osm_type, osm_id) "
        f"polygon identities ({rate:.1%}) have successfully extracted "
        "(`fetch_status=ok`) non-empty Wikipedia or Wikivoyage text."
    )
    return RenderResult(output_path=output_path, caption=caption)


__all__ = [
    "CoveredPoint",
    "TextPresenceSnapshot",
    "generate_geographic_text_presence",
    "load_text_presence",
]
