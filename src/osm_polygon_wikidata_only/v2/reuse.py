"""Lossless V1 reuse and V2 relationship assembly.

The V2 build starts from finalized V1 shards.  Existing Wikipedia,
Wikivoyage, Wikidata, and polygon-link rows are copied unchanged where
possible.  Only direct Wikipedia-tag relationships not already represented
by a V1 document are fetched.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.polygon_document_links import CANONICAL_COLUMNS
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.v2.checkpoints import (
    RegionFetchCheckpoint,
    region_input_fingerprint,
)
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    DirectEnrichmentResult,
    _cached_client,
    _link_row,
    _lookup_titles,
    _title_key,
    enrich_wikipedia_refs,
    reconcile_wikipedia_refs,
)
from osm_polygon_wikidata_only.v2.extractor import V2ExtractedPbf
from osm_polygon_wikidata_only.v2.sections import (
    SectionClient,
)
from osm_polygon_wikidata_only.v2.sections import (
    build_missing_sections as _build_missing_sections,
)
from osm_polygon_wikidata_only.v2.storage import write_v2_region
from osm_polygon_wikidata_only.v2.wikipedia_tags import WikipediaTagRef, parse_wikipedia_tags

LOGGER = logging.getLogger(__name__)
_RECONCILIATION_LOOKUP_BATCH_SIZE = 256

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


_DirectInput = tuple[str, dict[str, Any], tuple[WikipediaTagRef, ...]]


@dataclass(frozen=True, slots=True)
class _MergeInputs:
    """Stable V1/V2 rows used by the merge orchestration."""

    stem: str
    polygons: dict[str, dict[str, Any]]
    base_documents: dict[str, dict[str, Any]]
    base_links: dict[tuple[str, str, str], dict[str, Any]]
    direct_inputs: tuple[_DirectInput, ...]
    all_refs: tuple[WikipediaTagRef, ...]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with pq.ParquetFile(path) as parquet_file:
        return parquet_file.read().to_pylist()


def load_v1_region(data_root: DataRoot, stem: str) -> V1RegionData:
    """Load V1 rows while accepting both pre- and post-migration links."""
    polygons = _rows(data_root.processed_polygons / f"{stem}.parquet")
    documents = _load_v1_documents(data_root, stem)
    links = _rows(data_root.processed_links / f"{stem}.parquet")
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
        return _rows(documents_path)
    articles_path = data_root.processed_articles / f"{stem}.parquet"
    return [wikipedia_document_from_article_row(row).to_dict() for row in _rows(articles_path)]


def _normalize_links(
    links: list[dict[str, Any]],
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
    if target.is_file() and _sha256(source) == _sha256(target):
        return target
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    return target


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _polygon_refs(polygon: dict[str, Any]) -> tuple[WikipediaTagRef, ...]:
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


def _link_sources(row: dict[str, Any]) -> set[str]:
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


def _direct_inputs(polygons: dict[str, dict[str, Any]]) -> tuple[_DirectInput, ...]:
    """Return direct Wikipedia-tag work in deterministic polygon order."""
    return tuple(
        (polygon_id, polygon, refs)
        for polygon_id, polygon in sorted(polygons.items())
        if (refs := _polygon_refs(polygon))
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


def _load_merge_inputs(data_root: DataRoot, extracted: V2ExtractedPbf) -> _MergeInputs:
    stem = extracted.stem.stem
    v1 = load_v1_region(data_root, stem)
    polygons = _merge_extracted_polygons(v1.polygons, extracted, stem)
    base_documents = {str(row["document_id"]): dict(row) for row in v1.documents}
    base_links = {(_link_key(row)): dict(row) for row in v1.links}
    current_wikidata = _current_wikidata(extracted)
    _drop_stale_links_with_logging(base_links, current_wikidata, stem)
    direct_inputs = _direct_inputs(polygons)
    return _MergeInputs(
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


def _all_refs(direct_inputs: tuple[_DirectInput, ...]) -> tuple[WikipediaTagRef, ...]:
    return tuple(ref for _polygon_id, _polygon, refs in direct_inputs for ref in refs)


def _drop_stale_links_with_logging(
    links: dict[tuple[str, str, str], dict[str, Any]],
    current_wikidata: dict[str, frozenset[str]],
    stem: str,
) -> None:
    stale_link_count = _drop_stale_v1_links(links, current_wikidata)
    if stale_link_count:
        LOGGER.info("V2 %s: dropped %d stale V1 polygon-document link(s)", stem, stale_link_count)


def _fetch_checkpoint(
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


def _update_polygon_text_fields(
    polygons: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> None:
    links_by_polygon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in links.values():
        links_by_polygon[str(row["polygon_id"])].append(row)
    for polygon_id, polygon in polygons.items():
        _update_polygon_text_fields_for_row(
            polygon,
            links_by_polygon.get(polygon_id, []),
            documents,
        )


def _update_polygon_text_fields_for_row(
    polygon: dict[str, Any],
    rows: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> None:
    wikipedia_rows = [row for row in rows if row.get("project") == "wikipedia"]
    languages = _wikipedia_languages(wikipedia_rows)
    polygon.update(
        {
            "has_wikipedia": bool(wikipedia_rows),
            "wikipedia_language_count": len(languages),
            "wikipedia_languages": json.dumps(languages, separators=(",", ":")),
            "wikipedia_article_count": len(wikipedia_rows),
            "has_english_wikipedia": "en" in languages,
            "has_french_wikipedia": "fr" in languages,
            "text_available": _has_wikipedia_text(wikipedia_rows, documents),
        }
    )
    if languages and not polygon.get("best_language"):
        polygon["best_language"] = languages[0]


def _wikipedia_languages(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("language", "")) for row in rows if row.get("language")})


def _has_wikipedia_text(
    rows: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> bool:
    return any(bool(documents.get(str(row["document_id"]), {}).get("full_text")) for row in rows)


def _enrich_direct_item(
    item: _DirectInput,
    *,
    index: Any,
    wikipedia_client: Any,
    fetch_full_text: bool,
    initial_matches: dict[tuple[str, str], Any],
) -> DirectEnrichmentResult:
    polygon_id, polygon, refs = item
    return enrich_wikipedia_refs(
        polygon_id,
        refs,
        index=index,
        wikipedia_client=wikipedia_client,
        polygon_context=polygon,
        cache=None,
        fetch_full_text=fetch_full_text,
        wait_for_index=False,
        initial_matches=initial_matches,
        defer_final_lookup=True,
    )


def _record_direct_result(
    item: _DirectInput,
    result: DirectEnrichmentResult,
    *,
    results_by_polygon: dict[str, DirectEnrichmentResult],
    fetch_checkpoint: RegionFetchCheckpoint | None,
    stem: str,
    total: int,
    saved: int,
) -> int:
    polygon_id, _polygon, refs = item
    results_by_polygon[polygon_id] = result
    if fetch_checkpoint is None:
        return saved
    fetch_checkpoint.save_direct(polygon_id, refs, result)
    if result.deferred_errors:
        return saved
    next_saved = saved + 1
    if next_saved == 1 or next_saved % 100 == 0:
        LOGGER.info(
            "V2 %s: direct Wikipedia checkpoints saved %d/%d polygons",
            stem,
            next_saved,
            total,
        )
    return next_saved


def _run_direct_workers(
    pending_inputs: list[_DirectInput],
    *,
    workers: int,
    enrich_one: Any,
    record_result: Any,
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Future[DirectEnrichmentResult], tuple[int, _DirectInput]] = {}
        completed: dict[int, tuple[_DirectInput, Future[DirectEnrichmentResult]]] = {}
        inputs = iter(enumerate(pending_inputs))
        next_to_record = 0

        def submit_next() -> None:
            try:
                position, item = next(inputs)
            except StopIteration:
                return
            pending[executor.submit(enrich_one, item)] = (position, item)

        for _ in range(workers):
            submit_next()
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                position, item = pending.pop(future)
                completed[position] = (item, future)
                submit_next()
            while next_to_record in completed:
                item, future = completed.pop(next_to_record)
                record_result(item, future.result())
                next_to_record += 1


def _enrich_direct_inputs(
    stem: str,
    direct_inputs: tuple[_DirectInput, ...],
    *,
    index: Any,
    wikipedia_client: Any,
    cache: Any,
    fetch_full_text: bool,
    direct_workers: int,
    initial_matches: dict[tuple[str, str], Any],
    fetch_checkpoint: RegionFetchCheckpoint | None,
) -> tuple[DirectEnrichmentResult, ...]:
    direct_client = _cached_client(wikipedia_client, cache)
    results_by_polygon, pending_inputs = _load_checkpointed_direct_results(
        stem,
        direct_inputs,
        fetch_checkpoint,
    )

    def enrich_one(item: _DirectInput) -> DirectEnrichmentResult:
        return _enrich_direct_item(
            item,
            index=index,
            wikipedia_client=direct_client,
            fetch_full_text=fetch_full_text,
            initial_matches=initial_matches,
        )

    saved = 0

    def record_result(item: _DirectInput, result: DirectEnrichmentResult) -> None:
        nonlocal saved
        saved = _record_direct_result(
            item,
            result,
            results_by_polygon=results_by_polygon,
            fetch_checkpoint=fetch_checkpoint,
            stem=stem,
            total=len(direct_inputs),
            saved=saved,
        )

    _enrich_pending_direct_inputs(
        pending_inputs,
        direct_workers=direct_workers,
        enrich_one=enrich_one,
        record_result=record_result,
    )
    return tuple(results_by_polygon[polygon_id] for polygon_id, _polygon, _refs in direct_inputs)


def _load_checkpointed_direct_results(
    stem: str,
    direct_inputs: tuple[_DirectInput, ...],
    fetch_checkpoint: RegionFetchCheckpoint | None,
) -> tuple[dict[str, DirectEnrichmentResult], list[_DirectInput]]:
    results: dict[str, DirectEnrichmentResult] = {}
    pending: list[_DirectInput] = []
    for item in direct_inputs:
        polygon_id, _polygon, refs = item
        cached_result = (
            fetch_checkpoint.load_direct(polygon_id, refs) if fetch_checkpoint is not None else None
        )
        if cached_result is None:
            pending.append(item)
            continue
        results[polygon_id] = cached_result
        LOGGER.info("V2 %s: reused checkpointed Wikipedia fetch for %s", stem, polygon_id)
    return results, pending


def _enrich_pending_direct_inputs(
    pending_inputs: list[_DirectInput],
    *,
    direct_workers: int,
    enrich_one: Any,
    record_result: Any,
) -> None:
    if not pending_inputs:
        return
    if direct_workers > 1:
        LOGGER.info(
            "V2 direct Wikipedia enrichment: %d polygon(s) with up to %d workers",
            len(pending_inputs),
            direct_workers,
        )
        _run_direct_workers(
            pending_inputs,
            workers=min(direct_workers, len(pending_inputs)),
            enrich_one=enrich_one,
            record_result=record_result,
        )
        return
    for item in pending_inputs:
        record_result(item, enrich_one(item))


def _speculative_results(
    direct_inputs: tuple[_DirectInput, ...],
    results: tuple[DirectEnrichmentResult, ...],
    *,
    index: Any,
    initial_matches: dict[tuple[str, str], Any],
) -> tuple[DirectEnrichmentResult, ...]:
    provisional_matches = dict(initial_matches)
    unresolved_refs = _unresolved_refs(direct_inputs, initial_matches)
    if unresolved_refs:
        provisional_matches.update(_lookup_titles(index, unresolved_refs))
    return tuple(
        reconcile_wikipedia_refs(
            polygon_id,
            refs,
            result,
            index=index,
            polygon_context=polygon,
            title_matches=provisional_matches,
        )
        for (polygon_id, polygon, refs), result in zip(direct_inputs, results, strict=True)
    )


def _unresolved_refs(
    direct_inputs: tuple[_DirectInput, ...],
    initial_matches: dict[tuple[str, str], Any],
) -> tuple[WikipediaTagRef, ...]:
    refs = (ref for _id, _polygon, refs in direct_inputs for ref in refs)
    return tuple(
        ref for ref in refs if not initial_matches.get(_title_key(ref.language, ref.title))
    )


def _add_direct_result(
    documents: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    direct: DirectEnrichmentResult,
) -> None:
    for document in direct.documents:
        documents[str(document["document_id"])] = dict(document)
    for link in direct.links:
        key = _link_key(link)
        if key not in links:
            links[key] = dict(link)
            continue
        sources = set(json.loads(links[key].get("link_sources", "[]")))
        sources.update(json.loads(link.get("link_sources", "[]")))
        links[key]["link_sources"] = json.dumps(sorted(sources), separators=(",", ":"))


def _build_region_sections(
    stem: str,
    data_root: DataRoot,
    documents: dict[str, dict[str, Any]],
    *,
    section_client: SectionClient | None,
    section_workers: int,
    fetch_checkpoint: RegionFetchCheckpoint | None,
    filter_document_ids: bool,
) -> list[dict[str, Any]]:
    sections, completed_section_ids = _load_section_rows(
        data_root,
        stem,
        documents=documents,
        fetch_checkpoint=fetch_checkpoint,
        filter_document_ids=filter_document_ids,
    )
    return _build_missing_sections(
        list(documents.values()),
        sections,
        section_client=section_client,
        section_workers=section_workers,
        on_document=_section_checkpoint_callback(stem, fetch_checkpoint),
        completed_document_ids=completed_section_ids,
    )


def _load_section_rows(
    data_root: DataRoot,
    stem: str,
    *,
    documents: dict[str, dict[str, Any]],
    fetch_checkpoint: RegionFetchCheckpoint | None,
    filter_document_ids: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    sections_path = data_root.processed_v2 / "wikipedia" / "sections" / f"{stem}.parquet"
    sections = _rows(sections_path)
    sections, completed_section_ids = _merge_checkpoint_sections(sections, fetch_checkpoint)
    return _filter_section_rows(sections, completed_section_ids, documents, filter_document_ids)


def _merge_checkpoint_sections(
    sections: list[dict[str, Any]],
    fetch_checkpoint: RegionFetchCheckpoint | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    if fetch_checkpoint is None:
        return sections, set()
    checkpoint_sections, completed_section_ids = fetch_checkpoint.load_section_state()
    sections.extend(checkpoint_sections)
    return list(
        {str(row.get("section_id", "")): row for row in sections}.values()
    ), completed_section_ids


def _filter_section_rows(
    sections: list[dict[str, Any]],
    completed_section_ids: set[str],
    documents: dict[str, dict[str, Any]],
    filter_document_ids: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    if not filter_document_ids:
        return sections, completed_section_ids
    filtered = [row for row in sections if str(row.get("document_id", "")) in documents]
    return filtered, completed_section_ids


def _section_checkpoint_callback(
    stem: str,
    fetch_checkpoint: RegionFetchCheckpoint | None,
) -> Any:
    if fetch_checkpoint is None:
        return None
    saved = 0

    def save_section_checkpoint(document_id: str, rows: list[dict[str, Any]]) -> None:
        nonlocal saved
        fetch_checkpoint.save_sections(document_id, rows)
        saved += 1
        if saved == 1 or saved % 100 == 0:
            LOGGER.info(
                "V2 %s: Wikipedia section checkpoints saved %d documents",
                stem,
                saved,
            )

    return save_section_checkpoint


def _reconcile_merge_results(
    direct_inputs: tuple[_DirectInput, ...],
    speculative_results: tuple[DirectEnrichmentResult, ...],
    *,
    index: Any,
    documents: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    base_documents: dict[str, dict[str, Any]],
    base_links: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    documents = dict(base_documents)
    links = dict(base_links)
    for (polygon_id, polygon, refs), speculative in zip(
        direct_inputs, speculative_results, strict=True
    ):
        _add_direct_result(
            documents,
            links,
            reconcile_wikipedia_refs(
                polygon_id,
                refs,
                speculative,
                index=index,
                polygon_context=polygon,
            ),
        )
    return documents, links


def _collect_speculative_direct_results(
    inputs: _MergeInputs,
    *,
    index: Any,
    wikipedia_client: Any,
    cache: Any,
    fetch_full_text: bool,
    direct_workers: int,
    fetch_checkpoint: RegionFetchCheckpoint | None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    tuple[DirectEnrichmentResult, ...],
]:
    documents = dict(inputs.base_documents)
    links = dict(inputs.base_links)
    initial_matches = _lookup_titles(index, inputs.all_refs)
    fetched = _enrich_direct_inputs(
        inputs.stem,
        inputs.direct_inputs,
        index=index,
        wikipedia_client=wikipedia_client,
        cache=cache,
        fetch_full_text=fetch_full_text,
        direct_workers=direct_workers,
        initial_matches=initial_matches,
        fetch_checkpoint=fetch_checkpoint,
    )
    speculative = _speculative_results(
        inputs.direct_inputs,
        fetched,
        index=index,
        initial_matches=initial_matches,
    )
    for direct in speculative:
        _add_direct_result(documents, links, direct)
    return documents, links, speculative


def _prepare_merge_sections(
    inputs: _MergeInputs,
    data_root: DataRoot,
    documents: dict[str, dict[str, Any]],
    *,
    section_client: SectionClient | None,
    section_workers: int,
    fetch_checkpoint: RegionFetchCheckpoint | None,
) -> list[dict[str, Any]]:
    if inputs.direct_inputs:
        LOGGER.info(
            "V2 %s: building sections for speculative pages before V1 index completion",
            inputs.stem,
        )
    copy_v1_sidecars(data_root, inputs.stem, data_root.processed_v2)
    return _build_region_sections(
        inputs.stem,
        data_root,
        documents,
        section_client=section_client,
        section_workers=section_workers,
        fetch_checkpoint=fetch_checkpoint,
        filter_document_ids=False,
    )


def _reconcile_merge_if_ready(
    inputs: _MergeInputs,
    speculative_results: tuple[DirectEnrichmentResult, ...],
    *,
    index: Any,
    documents: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    wait_for_index: bool,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    if not inputs.direct_inputs or not wait_for_index:
        return documents, links
    LOGGER.info("V2 %s: waiting once for final V1 reuse-index reconciliation", inputs.stem)
    index.wait_until_ready()
    LOGGER.info("V2 %s: final V1 reuse-index reconciliation started", inputs.stem)
    return _reconcile_merge_results(
        inputs.direct_inputs,
        speculative_results,
        index=index,
        documents=documents,
        links=links,
        base_documents=inputs.base_documents,
        base_links=inputs.base_links,
    )


def _write_merged_region(
    data_root: DataRoot,
    inputs: _MergeInputs,
    documents: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    sections: list[dict[str, Any]],
    *,
    wait_for_index: bool,
) -> None:
    final_document_ids = set(documents)
    sections = [row for row in sections if str(row.get("document_id", "")) in final_document_ids]
    write_v2_region(
        data_root.processed_v2,
        inputs.stem,
        polygons=sorted(inputs.polygons.values(), key=lambda row: str(row["polygon_id"])),
        documents=sorted(documents.values(), key=lambda row: str(row["document_id"])),
        links=sorted(links.values(), key=lambda row: _link_key(row)),
        sections=sections,
        v1_index_reconciled=wait_for_index or not inputs.direct_inputs,
    )


def _load_reconciliation_rows(
    data_root: DataRoot,
    stem: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str], list[dict[str, Any]]],
]:
    polygons_rows = _rows(data_root.processed_v2 / "polygons" / f"{stem}.parquet")
    documents = {
        str(row["document_id"]): dict(row)
        for row in _rows(data_root.processed_v2 / "wikipedia/documents" / f"{stem}.parquet")
    }
    links = {
        _link_key(row): dict(row)
        for row in _rows(data_root.processed_v2 / "polygon_document_links" / f"{stem}.parquet")
    }
    polygons = {str(row["polygon_id"]): dict(row) for row in polygons_rows}
    current_by_title = _documents_by_title(documents)
    return polygons_rows, polygons, documents, links, current_by_title


def _documents_by_title(
    documents: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    current_by_title: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in documents.values():
        key = (
            str(row.get("language", "")).casefold(),
            str(row.get("title", "")).replace("_", " ").casefold(),
        )
        current_by_title.setdefault(key, []).append(row)
    for candidates in current_by_title.values():
        candidates.sort(key=lambda row: str(row.get("document_id", "")))
    return current_by_title


def _remove_speculative_links(
    links: dict[tuple[str, str, str], dict[str, Any]],
) -> set[str]:
    direct_document_ids: set[str] = set()
    for key, row in list(links.items()):
        _remove_speculative_link(key, row, links, direct_document_ids)
    return direct_document_ids


def _remove_speculative_link(
    key: tuple[str, str, str],
    row: dict[str, Any],
    links: dict[tuple[str, str, str], dict[str, Any]],
    direct_document_ids: set[str],
) -> None:
    sources = _link_sources(row)
    if "osm_wikipedia_tag" not in sources:
        return
    document_id = str(row.get("document_id", ""))
    if row.get("project") == "wikipedia" and not row.get("wikidata"):
        direct_document_ids.add(document_id)
    sources.discard("osm_wikipedia_tag")
    if sources:
        row["link_sources"] = json.dumps(sorted(sources), separators=(",", ":"))
        return
    del links[key]


def _reconciliation_ref_items(
    polygons: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any], WikipediaTagRef]]:
    return [
        (polygon_id, polygon, ref)
        for polygon_id, polygon in sorted(polygons.items())
        for ref in _polygon_refs(polygon)
    ]


def _reconcile_ref_items(
    ref_items: list[tuple[str, dict[str, Any], WikipediaTagRef]],
    *,
    index: Any,
    documents: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    current_by_title: dict[tuple[str, str], list[dict[str, Any]]],
    wikipedia_client: Any | None,
    cache: Any,
    fetch_full_text: bool,
) -> None:
    for offset in range(0, len(ref_items), _RECONCILIATION_LOOKUP_BATCH_SIZE):
        _reconcile_ref_chunk(
            ref_items[offset : offset + _RECONCILIATION_LOOKUP_BATCH_SIZE],
            index=index,
            documents=documents,
            links=links,
            current_by_title=current_by_title,
            wikipedia_client=wikipedia_client,
            cache=cache,
            fetch_full_text=fetch_full_text,
        )


def _reconcile_ref_chunk(
    chunk: list[tuple[str, dict[str, Any], WikipediaTagRef]],
    *,
    index: Any,
    documents: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    current_by_title: dict[tuple[str, str], list[dict[str, Any]]],
    wikipedia_client: Any | None,
    cache: Any,
    fetch_full_text: bool,
) -> None:
    matches = _lookup_titles(index, [ref for _id, _polygon, ref in chunk])
    for polygon_id, polygon, ref in chunk:
        candidates = _find_reconciliation_candidates(
            polygon_id,
            polygon,
            ref,
            matches=matches,
            current_by_title=current_by_title,
            index=index,
            wikipedia_client=wikipedia_client,
            cache=cache,
            fetch_full_text=fetch_full_text,
        )
        if candidates:
            _apply_reconciliation_candidate(
                polygon_id,
                polygon,
                candidates[0],
                documents=documents,
                links=links,
            )


def _find_reconciliation_candidates(
    polygon_id: str,
    polygon: dict[str, Any],
    ref: WikipediaTagRef,
    *,
    matches: dict[tuple[str, str], Any],
    current_by_title: dict[tuple[str, str], list[dict[str, Any]]],
    index: Any,
    wikipedia_client: Any | None,
    cache: Any,
    fetch_full_text: bool,
) -> Any:
    candidates = matches.get(_title_key(ref.language, ref.title), ()) or current_by_title.get(
        (ref.language.casefold(), ref.title.replace("_", " ").casefold()),
        (),
    )
    if candidates or wikipedia_client is None:
        return candidates
    recovered = enrich_wikipedia_refs(
        polygon_id,
        (ref,),
        index=index,
        wikipedia_client=wikipedia_client,
        polygon_context=polygon,
        cache=cache,
        fetch_full_text=fetch_full_text,
        wait_for_index=True,
    )
    return recovered.documents


def _apply_reconciliation_candidate(
    polygon_id: str,
    polygon: dict[str, Any],
    candidate: dict[str, Any],
    *,
    documents: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    document = dict(candidate)
    document_id = str(document["document_id"])
    documents[document_id] = document
    link = _link_row(
        polygon_id,
        document,
        polygon_context=polygon,
        sources=("osm_wikipedia_tag",),
    )
    _merge_reconciliation_link(links, link)


def _merge_reconciliation_link(
    links: dict[tuple[str, str, str], dict[str, Any]],
    link: dict[str, Any],
) -> None:
    key = _link_key(link)
    if key not in links:
        links[key] = link
        return
    sources = _link_sources(links[key])
    sources.update(_link_sources(link))
    links[key]["link_sources"] = json.dumps(sorted(sources), separators=(",", ":"))


def _drop_unreferenced_direct_documents(
    documents: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    direct_document_ids: set[str],
) -> None:
    referenced = {str(row.get("document_id", "")) for row in links.values()}
    for document_id in direct_document_ids - referenced:
        documents.pop(document_id, None)


def _write_reconciled_region(
    data_root: DataRoot,
    stem: str,
    polygons: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    sections: list[dict[str, Any]],
) -> None:
    write_v2_region(
        data_root.processed_v2,
        stem,
        polygons=sorted(polygons.values(), key=lambda row: str(row["polygon_id"])),
        documents=sorted(documents.values(), key=lambda row: str(row["document_id"])),
        links=sorted(links.values(), key=_link_key),
        sections=sections,
        v1_index_reconciled=True,
    )


def merge_v2_region(
    data_root: DataRoot,
    extracted: V2ExtractedPbf,
    *,
    index: Any,
    wikipedia_client: Any,
    cache: Any = None,
    fetch_full_text: bool = True,
    section_client: SectionClient | None = None,
    section_workers: int = 8,
    direct_workers: int = 1,
    wait_for_index: bool = True,
    checkpoint_dir: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Merge V1 rows with V2 discoveries and persist one canonical region.

    With ``wait_for_index=False``, direct pages and their sections are fetched
    speculatively and the region is written with a pending reconciliation
    marker.  The runner reconciles those regions after the shared V1 index is
    complete, so no provisional artifact is published as final data.
    """
    inputs = _load_merge_inputs(data_root, extracted)
    fetch_checkpoint = _fetch_checkpoint(
        checkpoint_dir,
        inputs.stem,
        extracted.polygons,
        fetch_full_text,
    )
    documents, links, speculative_results = _collect_speculative_direct_results(
        inputs,
        index=index,
        wikipedia_client=wikipedia_client,
        cache=cache,
        fetch_full_text=fetch_full_text,
        direct_workers=direct_workers,
        fetch_checkpoint=fetch_checkpoint,
    )
    sections = _prepare_merge_sections(
        inputs,
        data_root,
        documents,
        section_client=section_client,
        section_workers=section_workers,
        fetch_checkpoint=fetch_checkpoint,
    )
    documents, links = _reconcile_merge_if_ready(
        inputs,
        speculative_results,
        index=index,
        documents=documents,
        links=links,
        wait_for_index=wait_for_index,
    )
    _update_polygon_text_fields(inputs.polygons, links, documents)
    _write_merged_region(
        data_root,
        inputs,
        documents,
        links,
        sections,
        wait_for_index=wait_for_index,
    )
    return tuple(inputs.polygons.values())


def reconcile_v2_region(
    data_root: DataRoot,
    stem: str,
    *,
    index: Any,
    wikipedia_client: Any | None = None,
    cache: Any = None,
    fetch_full_text: bool = True,
    section_client: SectionClient | None = None,
    section_workers: int = 8,
    checkpoint_dir: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Finalize one provisionally written region after V1 indexing.

    Direct Wikipedia-tag pages are retained unless the completed V1 index
    proves that the exact title already has a canonical V1 document.  In that
    case the direct document and its sections are discarded and the V1 row is
    linked instead.  The final region is written atomically with its
    reconciliation marker set.
    """
    polygons_rows, polygons, documents, links, current_by_title = _load_reconciliation_rows(
        data_root,
        stem,
    )
    fetch_checkpoint = _fetch_checkpoint(
        checkpoint_dir,
        stem,
        polygons_rows,
        fetch_full_text,
    )
    direct_document_ids = _remove_speculative_links(links)
    _reconcile_ref_items(
        _reconciliation_ref_items(polygons),
        index=index,
        documents=documents,
        links=links,
        current_by_title=current_by_title,
        wikipedia_client=wikipedia_client,
        cache=cache,
        fetch_full_text=fetch_full_text,
    )
    _drop_unreferenced_direct_documents(documents, links, direct_document_ids)
    sections = _build_region_sections(
        stem,
        data_root,
        documents,
        section_client=section_client,
        section_workers=section_workers,
        fetch_checkpoint=fetch_checkpoint,
        filter_document_ids=True,
    )
    _update_polygon_text_fields(polygons, links, documents)
    _write_reconciled_region(
        data_root,
        stem,
        polygons,
        documents,
        links,
        sections,
    )
    LOGGER.info("V2 %s: provisional direct pages reconciled against the completed V1 index", stem)
    return tuple(polygons.values())


def _link_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("polygon_id", "")),
        str(row.get("project", "")),
        str(row.get("document_id", "")),
    )


__all__ = [
    "SIDECAR_SUBDIRS",
    "SectionClient",
    "V1RegionData",
    "copy_v1_sidecars",
    "load_v1_region",
    "merge_v2_region",
    "reconcile_v2_region",
]
