"""Load finalized V1 rows and sidecars for the V2 merge."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import CANONICAL_COLUMNS
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.v2.checkpoints import (
    RegionFetchCheckpoint,
    region_input_fingerprint,
)
from osm_polygon_wikidata_only.v2.extractor import V2ExtractedPbf
from osm_polygon_wikidata_only.v2.wikipedia_tags import WikipediaTagRef, parse_wikipedia_tags

LOGGER = logging.getLogger(__name__)
PARQUET_BATCH_SIZE = 65_536
_V1_LINK_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*CANONICAL_COLUMNS, "document_id", "article_id", "project", "wikidata"))
)

SIDECAR_SUBDIRS: tuple[str, ...] = (
    "wikipedia/sections",
    "wikivoyage/documents",
    "wikivoyage/sections",
    "wikidata/facts",
)


@dataclass(frozen=True, slots=True)
class V1RegionData:
    """Rows and sidecar source paths loaded from one finalized V1 shard."""

    polygons: tuple[dict[str, Any], ...]
    documents: tuple[dict[str, Any], ...]
    links: tuple[dict[str, Any], ...]
    sidecars: tuple[Path, ...]


DirectInput = tuple[str, dict[str, Any], tuple[WikipediaTagRef, ...]]


@dataclass(frozen=True, slots=True)
class MergeInputs:
    """Stable V1/V2 rows used by the merge orchestration."""

    stem: str
    polygons: dict[str, dict[str, Any]]
    base_documents: dict[str, dict[str, Any]]
    base_links: dict[tuple[str, str, str], dict[str, Any]]
    direct_inputs: tuple[DirectInput, ...]
    all_refs: tuple[WikipediaTagRef, ...]


def iter_parquet_rows(
    path: Path, *, columns: Iterable[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Yield rows batch by batch, optionally projecting to ``columns``.

    Projected columns absent from the file are skipped; callers read rows with
    ``dict.get`` so a missing column and an unread column behave the same.
    """
    if not path.is_file():
        return
    with open_parquet(path) as parquet_file:
        projection = _present_columns(parquet_file, columns)
        for batch in iter_record_batches(
            parquet_file, batch_size=PARQUET_BATCH_SIZE, columns=projection
        ):
            yield from batch.to_pylist()


def _present_columns(parquet_file: Any, columns: Iterable[str] | None) -> list[str] | None:
    if columns is None:
        return None
    names = set(parquet_file.schema_arrow.names)
    return [column for column in columns if column in names]


def load_v1_region(data_root: DataRoot, stem: str) -> V1RegionData:
    """Load V1 rows while accepting both pre- and post-migration links."""
    polygons = iter_parquet_rows(data_root.processed_polygons / f"{stem}.parquet")
    documents = _load_v1_documents(data_root, stem)
    links = iter_parquet_rows(
        data_root.processed_links / f"{stem}.parquet", columns=_V1_LINK_COLUMNS
    )
    by_article = {str(row.get("article_id")): row for row in documents}
    normalized_links = _normalize_links(links, by_article)
    sidecars = _v1_sidecar_paths(data_root, stem)
    return V1RegionData(
        polygons=tuple(_v2_polygon_row(row) for row in polygons),
        documents=tuple(documents),
        links=tuple(normalized_links),
        sidecars=sidecars,
    )


def _load_v1_documents(data_root: DataRoot, stem: str) -> list[dict[str, Any]]:
    documents_path = data_root.processed / "wikipedia/documents" / f"{stem}.parquet"
    if documents_path.is_file():
        return list(iter_parquet_rows(documents_path))
    articles_path = data_root.processed_articles / f"{stem}.parquet"
    return [
        wikipedia_document_from_article_row(row).to_dict()
        for row in iter_parquet_rows(articles_path)
    ]


def _normalize_links(
    links: Iterable[dict[str, Any]],
    by_article: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [_normalize_link(row, by_article) for row in links]


def _v1_sidecar_paths(data_root: DataRoot, stem: str) -> tuple[Path, ...]:
    paths: list[Path] = []
    for subdir in SIDECAR_SUBDIRS:
        source = data_root.processed / subdir / f"{stem}.parquet"
        if source.is_file():
            paths.append(source)
    return tuple(paths)


def _v2_polygon_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    tags = _polygon_tags(row)
    refs, rejections = parse_wikipedia_tags(tags)
    result.update(
        {
            "wikipedia_tag_refs": _serialize_tag_refs(refs),
            "wikipedia_tag_rejections": _serialize_tag_rejections(rejections),
            "discovery_sources": _discovery_sources(row, refs),
        }
    )
    return result


def _polygon_tags(row: dict[str, Any]) -> dict[str, str]:
    try:
        parsed = json.loads(str(row.get("tags", "{}")))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _serialize_tag_refs(refs: tuple[WikipediaTagRef, ...]) -> str:
    return json.dumps(
        [
            {
                "language": ref.language,
                "title": ref.title,
                "raw_key": ref.raw_key,
                "raw_value": ref.raw_value,
            }
            for ref in refs
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _serialize_tag_rejections(rejections: Any) -> str:
    return json.dumps(
        [
            {"raw_key": item.raw_key, "raw_value": item.raw_value, "reason": item.reason}
            for item in rejections
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _discovery_sources(row: dict[str, Any], refs: tuple[WikipediaTagRef, ...]) -> str:
    sources = ["wikidata"] if row.get("wikidata") else []
    if refs:
        sources.append("wikipedia_tag")
    return json.dumps(sources, separators=(",", ":"))


def _normalize_link(row: dict[str, Any], by_article: dict[str, dict[str, Any]]) -> dict[str, Any]:
    document_id = row.get("document_id")
    if not document_id:
        document = by_article.get(str(row.get("article_id", "")))
        document_id = document.get("document_id") if document else None
    if not document_id:
        raise ValueError(f"V1 link has no resolvable document identity: {row}")
    return {
        **{column: row.get(column) for column in CANONICAL_COLUMNS},
        "document_id": str(document_id),
        "project": row.get("project", "wikipedia"),
        "wikidata": row.get("wikidata"),
        "link_sources": json.dumps(["wikidata_sitelink"], separators=(",", ":")),
    }


def copy_v1_sidecars(data_root: DataRoot, stem: str, destination: Path) -> tuple[Path, ...]:
    """Copy finalized V1 sidecars into isolated V2 storage idempotently."""
    copied: list[Path] = []
    for subdir in SIDECAR_SUBDIRS:
        target = _copy_v1_sidecar(data_root, stem, destination, subdir)
        if target is not None:
            copied.append(target)
    return tuple(copied)


def _copy_v1_sidecar(
    data_root: DataRoot,
    stem: str,
    destination: Path,
    subdir: str,
) -> Path | None:
    source = data_root.processed / subdir / f"{stem}.parquet"
    if not source.is_file():
        return None
    target = destination / subdir / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256_file(source) == sha256_file(target):
        return target
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    return target


def polygon_refs(polygon: dict[str, Any]) -> tuple[WikipediaTagRef, ...]:
    try:
        raw_refs = json.loads(str(polygon.get("wikipedia_tag_refs", "[]")))
    except json.JSONDecodeError:
        return ()
    if not isinstance(raw_refs, list):
        return ()
    refs: list[WikipediaTagRef] = []
    for item in raw_refs:
        if not isinstance(item, dict):
            continue
        parsed, _ = parse_wikipedia_tags(
            {str(item.get("raw_key", "")): str(item.get("raw_value", ""))}
        )
        refs.extend(parsed)
    return tuple(refs)


def parse_link_sources(row: dict[str, Any]) -> set[str]:
    try:
        raw = json.loads(str(row.get("link_sources", "[]")))
    except json.JSONDecodeError:
        return set()
    return {str(source) for source in raw} if isinstance(raw, list) else set()


def _wikidata_qids(value: Any) -> frozenset[str]:
    """Normalize one polygon Wikidata tag for set-based merge comparison."""
    return frozenset(qids_from_osm_tag(str(value or "")))


def _drop_stale_v1_links(
    links: dict[tuple[str, str, str], dict[str, Any]],
    current_wikidata: dict[str, frozenset[str]],
) -> int:
    """Drop V1 links whose QID is absent from the current PBF tag.

    V1 links are historical snapshots. A changed OSM Wikidata tag makes a
    QID link stale, while direct Wikipedia-tag links (which have no QID) stay
    eligible for the V2 enrichment pass.
    """
    stale_keys = [key for key, row in links.items() if _is_stale_link(row, current_wikidata)]
    for key in stale_keys:
        del links[key]
    return len(stale_keys)


def _is_stale_link(
    row: dict[str, Any],
    current_wikidata: dict[str, frozenset[str]],
) -> bool:
    polygon_id = str(row.get("polygon_id", ""))
    qid = str(row.get("wikidata") or "")
    return bool(polygon_id in current_wikidata and qid and qid not in current_wikidata[polygon_id])


def _direct_inputs(polygons: dict[str, dict[str, Any]]) -> tuple[DirectInput, ...]:
    """Return direct Wikipedia-tag work in deterministic polygon order."""
    return tuple(
        (polygon_id, polygon, refs)
        for polygon_id, polygon in sorted(polygons.items())
        if (refs := polygon_refs(polygon))
    )


def _merge_extracted_polygons(
    v1_polygons: tuple[dict[str, Any], ...],
    extracted: V2ExtractedPbf,
    stem: str,
) -> dict[str, dict[str, Any]]:
    polygons = {str(row["polygon_id"]): dict(row) for row in v1_polygons}
    for discovered in extracted.polygons:
        _merge_discovered_polygon(polygons, discovered, stem)
    return polygons


def _merge_discovered_polygon(
    polygons: dict[str, dict[str, Any]],
    discovered: dict[str, Any],
    stem: str,
) -> None:
    key = str(discovered["polygon_id"])
    existing = polygons.get(key)
    if existing is None:
        polygons[key] = dict(discovered)
        return
    previous_qids = _wikidata_qids(existing.get("wikidata"))
    discovered_qids = _wikidata_qids(discovered.get("wikidata"))
    if previous_qids != discovered_qids:
        LOGGER.warning(
            "V2 %s: current PBF Wikidata tag changed for %s (%s -> %s); "
            "using the current value and dropping stale V1 links",
            stem,
            key,
            ";".join(sorted(previous_qids)) or "none",
            ";".join(sorted(discovered_qids)) or "none",
        )
    existing.update(discovered)


def load_merge_inputs(data_root: DataRoot, extracted: V2ExtractedPbf) -> MergeInputs:
    stem = extracted.stem.stem
    v1 = load_v1_region(data_root, stem)
    polygons = _merge_extracted_polygons(v1.polygons, extracted, stem)
    base_documents = {str(row["document_id"]): dict(row) for row in v1.documents}
    base_links = {(link_key(row)): dict(row) for row in v1.links}
    current_wikidata = _current_wikidata(extracted)
    _drop_stale_links_with_logging(base_links, current_wikidata, stem)
    direct_inputs = _direct_inputs(polygons)
    return MergeInputs(
        stem=stem,
        polygons=polygons,
        base_documents=base_documents,
        base_links=base_links,
        direct_inputs=direct_inputs,
        all_refs=_all_refs(direct_inputs),
    )


def _current_wikidata(extracted: V2ExtractedPbf) -> dict[str, frozenset[str]]:
    return {
        str(row["polygon_id"]): _wikidata_qids(row.get("wikidata")) for row in extracted.polygons
    }


def _all_refs(direct_inputs: tuple[DirectInput, ...]) -> tuple[WikipediaTagRef, ...]:
    return tuple(ref for _polygon_id, _polygon, refs in direct_inputs for ref in refs)


def _drop_stale_links_with_logging(
    links: dict[tuple[str, str, str], dict[str, Any]],
    current_wikidata: dict[str, frozenset[str]],
    stem: str,
) -> None:
    stale_link_count = _drop_stale_v1_links(links, current_wikidata)
    if stale_link_count:
        LOGGER.info("V2 %s: dropped %d stale V1 polygon-document link(s)", stem, stale_link_count)


def open_fetch_checkpoint(
    checkpoint_dir: Path | None,
    stem: str,
    polygons: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    fetch_full_text: bool,
) -> RegionFetchCheckpoint | None:
    if checkpoint_dir is None:
        return None
    return RegionFetchCheckpoint(
        checkpoint_dir,
        stem,
        input_fingerprint=region_input_fingerprint(polygons),
        fetch_full_text=fetch_full_text,
    )


def link_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("polygon_id", "")),
        str(row.get("project", "")),
        str(row.get("document_id", "")),
    )
