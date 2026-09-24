"""Direct Wikipedia-tag enrichment, checkpointed workers, and section assembly for V2 merges."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.v2.checkpoints import (
    RegionFetchCheckpoint,
)
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    DirectEnrichmentResult,
    enrich_wikipedia_refs,
    reconcile_wikipedia_refs,
)
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    cached_client as _cached_client,
)
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    lookup_titles as _lookup_titles,
)
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    title_key as _title_key,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    PARQUET_BATCH_SIZE as _PARQUET_BATCH_SIZE,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    DirectInput as _DirectInput,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    iter_parquet_rows as _rows,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    link_key as _link_key,
)
from osm_polygon_wikidata_only.v2.sections import (
    SectionClient,
)
from osm_polygon_wikidata_only.v2.sections import (
    build_missing_sections as _build_missing_sections,
)
from osm_polygon_wikidata_only.v2.wikipedia_tags import WikipediaTagRef

LOGGER = logging.getLogger(__name__)
_SECTION_KEY_COLUMNS: tuple[str, ...] = ("section_id", "document_id")


def update_polygon_text_fields(
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


def enrich_direct_inputs(
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


def speculative_direct_results(
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


def add_direct_result(
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


def build_region_sections(
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
    checkpoint_state = (
        fetch_checkpoint.load_section_state() if fetch_checkpoint is not None else None
    )
    if filter_document_ids:
        checkpoint_rows = checkpoint_state[0] if checkpoint_state is not None else []
        sections = _document_section_rows(sections_path, documents, checkpoint_rows)
    else:
        sections = list(_rows(sections_path))
    sections, completed_section_ids = _merge_checkpoint_sections(sections, checkpoint_state)
    return _filter_section_rows(sections, completed_section_ids, documents, filter_document_ids)


def _document_section_rows(
    path: Path,
    documents: dict[str, dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read only the stored sections that can survive the document filter.

    A stored row is kept when any row sharing its ``section_id`` (stored or
    checkpointed) belongs to a kept document.  Every other row would be
    dropped by :func:`_filter_section_rows` without affecting the position
    or value of a surviving section, so the merged output is unchanged.
    The filter runs in Arrow, and only kept rows become Python dicts.
    """
    if not path.is_file():
        return []
    with open_parquet(path) as parquet_file:
        if not _has_string_section_keys(parquet_file.schema_arrow):
            return list(_rows(path))
        kept_section_ids = _kept_section_ids(parquet_file, documents, checkpoint_rows)
        return _rows_with_section_ids(parquet_file, kept_section_ids)


def _kept_section_ids(
    parquet_file: Any,
    documents: dict[str, dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
) -> set[str]:
    """Collect section ids owned by a kept document, reading only key columns."""
    document_ids = pa.array(list(documents), type=pa.string())
    kept_section_ids = {
        str(row.get("section_id", ""))
        for row in checkpoint_rows
        if str(row.get("document_id", "")) in documents
    }
    for batch in iter_record_batches(
        parquet_file, batch_size=_PARQUET_BATCH_SIZE, columns=_SECTION_KEY_COLUMNS
    ):
        section_ids = _section_key(batch.column(0))
        kept = _compute("filter", section_ids, _is_in(_section_key(batch.column(1)), document_ids))
        kept_section_ids.update(kept.to_pylist())
    return kept_section_ids


def _rows_with_section_ids(parquet_file: Any, section_ids: set[str]) -> list[dict[str, Any]]:
    """Convert to Python only the stored rows whose section id is in ``section_ids``."""
    section_id_set = pa.array(sorted(section_ids), type=pa.string())
    rows: list[dict[str, Any]] = []
    for batch in iter_record_batches(parquet_file, batch_size=_PARQUET_BATCH_SIZE):
        keys = _section_key(batch.column(batch.schema.get_field_index("section_id")))
        rows.extend(batch.filter(_is_in(keys, section_id_set)).to_pylist())
    return rows


def _has_string_section_keys(schema: pa.Schema) -> bool:
    names = set(schema.names)
    return all(
        column in names and pa.types.is_string(schema.field(column).type)
        for column in _SECTION_KEY_COLUMNS
    )


def _section_key(values: Any) -> Any:
    """Mirror ``str(row.get(column, ""))`` for a nullable string column."""
    return pc.fill_null(values, "None")


def _is_in(values: Any, value_set: Any) -> Any:
    return _compute("is_in", values, options=pc.SetLookupOptions(value_set))


def _compute(function: str, *arguments: Any, options: Any = None) -> Any:
    return pc.call_function(function, list(arguments), options=options)


def _merge_checkpoint_sections(
    sections: list[dict[str, Any]],
    checkpoint_state: tuple[list[dict[str, Any]], set[str]] | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    if checkpoint_state is None:
        return sections, set()
    checkpoint_sections, completed_section_ids = checkpoint_state
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
