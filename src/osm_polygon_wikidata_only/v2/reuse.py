"""Lossless V1 reuse and V2 relationship assembly.

The V2 build starts from finalized V1 shards.  Existing Wikipedia,
Wikivoyage, Wikidata, and polygon-link rows are copied unchanged where
possible.  Only direct Wikipedia-tag relationships not already represented
by a V1 document are fetched.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
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


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with pq.ParquetFile(path) as parquet_file:
        return parquet_file.read().to_pylist()


def load_v1_region(data_root: DataRoot, stem: str) -> V1RegionData:
    """Load V1 rows while accepting both pre- and post-migration links."""
    polygons = _rows(data_root.processed_polygons / f"{stem}.parquet")
    documents_path = data_root.processed / "wikipedia/documents" / f"{stem}.parquet"
    if documents_path.is_file():
        documents = _rows(documents_path)
    else:
        articles_path = data_root.processed_articles / f"{stem}.parquet"
        documents = [
            wikipedia_document_from_article_row(row).to_dict() for row in _rows(articles_path)
        ]
    links = _rows(data_root.processed_links / f"{stem}.parquet")
    by_article = {str(row.get("article_id")): row for row in documents}
    normalized_links = [_normalize_link(row, by_article) for row in links]
    sidecars = tuple(
        source / f"{stem}.parquet"
        for source in (data_root.processed / subdir for subdir in SIDECAR_SUBDIRS)
        if (source / f"{stem}.parquet").is_file()
    )
    return V1RegionData(
        polygons=tuple(_v2_polygon_row(row) for row in polygons),
        documents=tuple(documents),
        links=tuple(normalized_links),
        sidecars=sidecars,
    )


def _v2_polygon_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    tags: dict[str, str]
    try:
        parsed = json.loads(str(row.get("tags", "{}")))
        tags = parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        tags = {}
    refs, rejections = parse_wikipedia_tags(tags)
    result["wikipedia_tag_refs"] = json.dumps(
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
    result["wikipedia_tag_rejections"] = json.dumps(
        [
            {"raw_key": item.raw_key, "raw_value": item.raw_value, "reason": item.reason}
            for item in rejections
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    sources = ["wikidata"] if row.get("wikidata") else []
    if refs:
        sources.append("wikipedia_tag")
    result["discovery_sources"] = json.dumps(sources, separators=(",", ":"))
    return result


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
    for source_root in (data_root.processed / subdir for subdir in SIDECAR_SUBDIRS):
        source = source_root / f"{stem}.parquet"
        if not source.is_file():
            continue
        target = destination / source_root.relative_to(data_root.processed) / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or _sha256(source) != _sha256(target):
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(target)
        copied.append(target)
    return tuple(copied)


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
    stale_keys = [
        key
        for key, row in links.items()
        if str(row.get("polygon_id", "")) in current_wikidata
        and (qid := str(row.get("wikidata") or ""))
        and qid not in current_wikidata[str(row["polygon_id"])]
    ]
    for key in stale_keys:
        del links[key]
    return len(stale_keys)


def _update_polygon_text_fields(
    polygons: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> None:
    links_by_polygon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in links.values():
        links_by_polygon[str(row["polygon_id"])].append(row)
    for polygon_id, polygon in polygons.items():
        rows = links_by_polygon.get(polygon_id, [])
        wikipedia_rows = [row for row in rows if row.get("project") == "wikipedia"]
        languages = sorted(
            {str(row.get("language", "")) for row in wikipedia_rows if row.get("language")}
        )
        # Keep the V1 field semantics: Wikivoyage relationships are part of
        # the unified link table, but do not make a polygon look like it has
        # a Wikipedia document.
        polygon["has_wikipedia"] = bool(wikipedia_rows)
        polygon["wikipedia_language_count"] = len(languages)
        polygon["wikipedia_languages"] = json.dumps(languages, separators=(",", ":"))
        polygon["wikipedia_article_count"] = len(wikipedia_rows)
        polygon["has_english_wikipedia"] = "en" in languages
        polygon["has_french_wikipedia"] = "fr" in languages
        polygon["text_available"] = any(
            bool(documents.get(str(row["document_id"]), {}).get("full_text"))
            for row in wikipedia_rows
        )
        if languages and not polygon.get("best_language"):
            polygon["best_language"] = languages[0]


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
    stem = extracted.stem.stem
    fetch_checkpoint = (
        RegionFetchCheckpoint(
            checkpoint_dir,
            stem,
            input_fingerprint=region_input_fingerprint(extracted.polygons),
            fetch_full_text=fetch_full_text,
        )
        if checkpoint_dir is not None
        else None
    )
    v1 = load_v1_region(data_root, stem)
    polygons = {str(row["polygon_id"]): dict(row) for row in v1.polygons}
    current_wikidata: dict[str, frozenset[str]] = {}
    for discovered in extracted.polygons:
        key = str(discovered["polygon_id"])
        discovered_qids = _wikidata_qids(discovered.get("wikidata"))
        current_wikidata[key] = discovered_qids
        existing = polygons.get(key)
        if existing is not None:
            previous_qids = _wikidata_qids(existing.get("wikidata"))
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
        else:
            polygons[key] = dict(discovered)
    base_documents = {str(row["document_id"]): dict(row) for row in v1.documents}
    base_links = {(_link_key(row)): dict(row) for row in v1.links}
    documents = dict(base_documents)
    links = dict(base_links)
    stale_link_count = _drop_stale_v1_links(links, current_wikidata)
    if stale_link_count:
        LOGGER.info("V2 %s: dropped %d stale V1 polygon-document link(s)", stem, stale_link_count)
    direct_client = _cached_client(wikipedia_client, cache)
    direct_inputs: list[tuple[str, dict[str, Any], tuple[WikipediaTagRef, ...]]] = []
    for polygon_id, polygon in sorted(polygons.items()):
        refs = _polygon_refs(polygon)
        if not refs:
            continue
        direct_inputs.append((polygon_id, polygon, refs))

    all_refs = tuple(ref for _polygon_id, _polygon, refs in direct_inputs for ref in refs)
    initial_matches = _lookup_titles(index, all_refs)

    def enrich_one(
        item: tuple[str, dict[str, Any], tuple[WikipediaTagRef, ...]],
    ) -> DirectEnrichmentResult:
        polygon_id, polygon, refs = item
        return enrich_wikipedia_refs(
            polygon_id,
            refs,
            index=index,
            wikipedia_client=direct_client,
            polygon_context=polygon,
            cache=None,
            fetch_full_text=fetch_full_text,
            wait_for_index=False,
            initial_matches=initial_matches,
            defer_final_lookup=True,
        )

    results_by_polygon: dict[str, DirectEnrichmentResult] = {}
    pending_inputs: list[tuple[str, dict[str, Any], tuple[WikipediaTagRef, ...]]] = []
    direct_checkpoints_saved = 0
    for item in direct_inputs:
        polygon_id, _polygon, refs = item
        cached_result = (
            fetch_checkpoint.load_direct(polygon_id, refs) if fetch_checkpoint is not None else None
        )
        if cached_result is None:
            pending_inputs.append(item)
        else:
            results_by_polygon[polygon_id] = cached_result
            LOGGER.info("V2 %s: reused checkpointed Wikipedia fetch for %s", stem, polygon_id)

    def record_result(
        item: tuple[str, dict[str, Any], tuple[WikipediaTagRef, ...]],
        result: DirectEnrichmentResult,
    ) -> None:
        nonlocal direct_checkpoints_saved
        polygon_id, _polygon, refs = item
        results_by_polygon[polygon_id] = result
        if fetch_checkpoint is not None:
            fetch_checkpoint.save_direct(polygon_id, refs, result)
            if not result.deferred_errors:
                direct_checkpoints_saved += 1
                if direct_checkpoints_saved == 1 or direct_checkpoints_saved % 100 == 0:
                    LOGGER.info(
                        "V2 %s: direct Wikipedia checkpoints saved %d/%d polygons",
                        stem,
                        direct_checkpoints_saved,
                        len(direct_inputs),
                    )

    if pending_inputs and direct_workers > 1:
        LOGGER.info(
            "V2 direct Wikipedia enrichment: %d polygon(s) with up to %d workers",
            len(pending_inputs),
            direct_workers,
        )
        workers = min(direct_workers, len(pending_inputs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: deque[
                tuple[
                    tuple[str, dict[str, Any], tuple[WikipediaTagRef, ...]],
                    Future[DirectEnrichmentResult],
                ]
            ] = deque()
            inputs = iter(pending_inputs)
            for _ in range(workers):
                try:
                    item = next(inputs)
                    pending.append((item, executor.submit(enrich_one, item)))
                except StopIteration:
                    break
            while pending:
                item, future = pending.popleft()
                record_result(item, future.result())
                with contextlib.suppress(StopIteration):
                    next_item = next(inputs)
                    pending.append((next_item, executor.submit(enrich_one, next_item)))
    else:
        for item in pending_inputs:
            record_result(item, enrich_one(item))

    speculative_results = [
        results_by_polygon[polygon_id] for polygon_id, _polygon, _refs in direct_inputs
    ]

    provisional_matches = dict(initial_matches)
    unresolved_refs = tuple(
        ref for ref in all_refs if not initial_matches.get(_title_key(ref.language, ref.title))
    )
    if unresolved_refs:
        provisional_matches.update(_lookup_titles(index, unresolved_refs))
    speculative_results = [
        reconcile_wikipedia_refs(
            polygon_id,
            refs,
            result,
            index=index,
            polygon_context=polygon,
            title_matches=provisional_matches,
        )
        for (polygon_id, polygon, refs), result in zip(
            direct_inputs, speculative_results, strict=True
        )
    ]

    def add_direct_result(direct: DirectEnrichmentResult) -> None:
        for document in direct.documents:
            documents[str(document["document_id"])] = dict(document)
        for link in direct.links:
            key = _link_key(link)
            if key in links:
                sources = set(json.loads(links[key].get("link_sources", "[]")))
                sources.update(json.loads(link.get("link_sources", "[]")))
                links[key]["link_sources"] = json.dumps(sorted(sources), separators=(",", ":"))
            else:
                links[key] = dict(link)

    for direct in speculative_results:
        add_direct_result(direct)

    # Build direct pages' sections while the V1 index is still scanning.  They
    # may be discarded below if the completed index proves that a page already
    # existed in V1, but the network and parsing work is then already finished.
    if direct_inputs:
        LOGGER.info(
            "V2 %s: building sections for speculative pages before V1 index completion",
            stem,
        )
    copy_v1_sidecars(data_root, stem, data_root.processed_v2)
    sections_path = data_root.processed_v2 / "wikipedia" / "sections" / f"{stem}.parquet"
    sections = _rows(sections_path)
    completed_section_ids: set[str] = set()
    if fetch_checkpoint is not None:
        checkpoint_sections, completed_section_ids = fetch_checkpoint.load_section_state()
        sections.extend(checkpoint_sections)
        sections = list({str(row.get("section_id", "")): row for row in sections}.values())
    section_checkpoints_saved = 0

    def save_section_checkpoint(document_id: str, rows: list[dict[str, Any]]) -> None:
        nonlocal section_checkpoints_saved
        if fetch_checkpoint is None:
            return
        fetch_checkpoint.save_sections(document_id, rows)
        section_checkpoints_saved += 1
        if section_checkpoints_saved == 1 or section_checkpoints_saved % 100 == 0:
            LOGGER.info(
                "V2 %s: Wikipedia section checkpoints saved %d documents",
                stem,
                section_checkpoints_saved,
            )

    sections = _build_missing_sections(
        list(documents.values()),
        sections,
        section_client=section_client,
        section_workers=section_workers,
        on_document=(save_section_checkpoint if fetch_checkpoint is not None else None),
        completed_document_ids=completed_section_ids,
    )

    # A miss is authoritative only after every V1 shard has been checked.  We
    # wait once, then rebuild the in-memory relationship set from the stable
    # V1 base and reconciled direct results before the atomic write.  Regions
    # with no direct references need no reconciliation barrier.
    if direct_inputs and wait_for_index:
        LOGGER.info("V2 %s: waiting once for final V1 reuse-index reconciliation", stem)
        index.wait_until_ready()
        LOGGER.info("V2 %s: final V1 reuse-index reconciliation started", stem)
        documents = dict(base_documents)
        links = dict(base_links)
        for item, speculative in zip(direct_inputs, speculative_results, strict=True):
            polygon_id, polygon, refs = item
            add_direct_result(
                reconcile_wikipedia_refs(
                    polygon_id,
                    refs,
                    speculative,
                    index=index,
                    polygon_context=polygon,
                )
            )

    _update_polygon_text_fields(polygons, links, documents)

    final_document_ids = set(documents)
    sections = [row for row in sections if str(row.get("document_id", "")) in final_document_ids]
    write_v2_region(
        data_root.processed_v2,
        stem,
        polygons=sorted(polygons.values(), key=lambda row: str(row["polygon_id"])),
        documents=sorted(documents.values(), key=lambda row: str(row["document_id"])),
        links=sorted(links.values(), key=lambda row: _link_key(row)),
        sections=sections,
        v1_index_reconciled=wait_for_index or not direct_inputs,
    )
    return tuple(polygons.values())


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
    polygons_rows = _rows(data_root.processed_v2 / "polygons" / f"{stem}.parquet")
    fetch_checkpoint = (
        RegionFetchCheckpoint(
            checkpoint_dir,
            stem,
            input_fingerprint=region_input_fingerprint(polygons_rows),
            fetch_full_text=fetch_full_text,
        )
        if checkpoint_dir is not None
        else None
    )
    documents = {
        str(row["document_id"]): dict(row)
        for row in _rows(data_root.processed_v2 / "wikipedia/documents" / f"{stem}.parquet")
    }
    links = {
        _link_key(row): dict(row)
        for row in _rows(data_root.processed_v2 / "polygon_document_links" / f"{stem}.parquet")
    }
    polygons = {str(row["polygon_id"]): dict(row) for row in polygons_rows}
    direct_document_ids: set[str] = set()
    current_by_title: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in documents.values():
        current_by_title.setdefault(
            (
                str(row.get("language", "")).casefold(),
                str(row.get("title", "")).replace("_", " ").casefold(),
            ),
            [],
        ).append(row)
    for candidates in current_by_title.values():
        candidates.sort(key=lambda row: str(row.get("document_id", "")))

    # Remove the speculative source from existing links.  A link that also
    # came from the V1 sitelink remains in the table and can receive the
    # source again below if the final lookup selects the same document.
    for key, row in list(links.items()):
        sources = _link_sources(row)
        if "osm_wikipedia_tag" not in sources:
            continue
        document_id = str(row.get("document_id", ""))
        if row.get("project") == "wikipedia" and not row.get("wikidata"):
            direct_document_ids.add(document_id)
        sources.discard("osm_wikipedia_tag")
        if sources:
            row["link_sources"] = json.dumps(sorted(sources), separators=(",", ":"))
        else:
            del links[key]

    ref_items = [
        (polygon_id, polygon, ref)
        for polygon_id, polygon in sorted(polygons.items())
        for ref in _polygon_refs(polygon)
    ]
    for offset in range(0, len(ref_items), _RECONCILIATION_LOOKUP_BATCH_SIZE):
        chunk = ref_items[offset : offset + _RECONCILIATION_LOOKUP_BATCH_SIZE]
        matches = _lookup_titles(index, [ref for _polygon_id, _polygon, ref in chunk])
        for polygon_id, polygon, ref in chunk:
            existing = matches.get(_title_key(ref.language, ref.title), ())
            candidates = existing or current_by_title.get(
                (
                    ref.language.casefold(),
                    ref.title.replace("_", " ").casefold(),
                ),
                (),
            )
            if not candidates and wikipedia_client is not None:
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
                candidates = recovered.documents
            if not candidates:
                continue
            document = dict(candidates[0])
            document_id = str(document["document_id"])
            documents[document_id] = document
            link = _link_row(
                polygon_id,
                document,
                polygon_context=polygon,
                sources=("osm_wikipedia_tag",),
            )
            key = _link_key(link)
            if key in links:
                sources = _link_sources(links[key])
                sources.update(_link_sources(link))
                links[key]["link_sources"] = json.dumps(sorted(sources), separators=(",", ":"))
            else:
                links[key] = link

    referenced = {str(row.get("document_id", "")) for row in links.values()}
    for document_id in direct_document_ids - referenced:
        documents.pop(document_id, None)
    sections_path = data_root.processed_v2 / "wikipedia" / "sections" / f"{stem}.parquet"
    sections = [row for row in _rows(sections_path) if str(row.get("document_id", "")) in documents]
    completed_section_ids: set[str] = set()
    if fetch_checkpoint is not None:
        checkpoint_sections, completed_section_ids = fetch_checkpoint.load_section_state()
        sections.extend(
            row for row in checkpoint_sections if str(row.get("document_id", "")) in documents
        )
        sections = list({str(row.get("section_id", "")): row for row in sections}.values())
    section_checkpoints_saved = 0

    def save_section_checkpoint(document_id: str, rows: list[dict[str, Any]]) -> None:
        nonlocal section_checkpoints_saved
        if fetch_checkpoint is None:
            return
        fetch_checkpoint.save_sections(document_id, rows)
        section_checkpoints_saved += 1
        if section_checkpoints_saved == 1 or section_checkpoints_saved % 100 == 0:
            LOGGER.info(
                "V2 %s: Wikipedia section checkpoints saved %d documents",
                stem,
                section_checkpoints_saved,
            )

    sections = _build_missing_sections(
        list(documents.values()),
        sections,
        section_client=section_client,
        section_workers=section_workers,
        on_document=(save_section_checkpoint if fetch_checkpoint is not None else None),
        completed_document_ids=completed_section_ids,
    )
    _update_polygon_text_fields(polygons, links, documents)
    write_v2_region(
        data_root.processed_v2,
        stem,
        polygons=sorted(polygons.values(), key=lambda row: str(row["polygon_id"])),
        documents=sorted(documents.values(), key=lambda row: str(row["document_id"])),
        links=sorted(links.values(), key=_link_key),
        sections=sections,
        v1_index_reconciled=True,
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
