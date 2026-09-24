"""Speculative direct-link reconciliation for V2 merges."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.v2.checkpoints import (
    RegionFetchCheckpoint,
)
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    DirectEnrichmentResult,
    enrich_wikipedia_refs,
    reconcile_wikipedia_refs,
)
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    link_row as _link_row,
)
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    lookup_titles as _lookup_titles,
)
from osm_polygon_wikidata_only.v2.direct_enrichment import (
    title_key as _title_key,
)
from osm_polygon_wikidata_only.v2.reuse_direct import (
    add_direct_result as _add_direct_result,
)
from osm_polygon_wikidata_only.v2.reuse_direct import (
    build_region_sections as _build_region_sections,
)
from osm_polygon_wikidata_only.v2.reuse_direct import (
    enrich_direct_inputs as _enrich_direct_inputs,
)
from osm_polygon_wikidata_only.v2.reuse_direct import (
    speculative_direct_results as _speculative_results,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    DirectInput as _DirectInput,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    MergeInputs as _MergeInputs,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    copy_v1_sidecars,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    iter_parquet_rows as _rows,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    link_key as _link_key,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    parse_link_sources as _link_sources,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    polygon_refs as _polygon_refs,
)
from osm_polygon_wikidata_only.v2.sections import (
    SectionClient,
)
from osm_polygon_wikidata_only.v2.storage import write_v2_region
from osm_polygon_wikidata_only.v2.wikipedia_tags import WikipediaTagRef

LOGGER = logging.getLogger(__name__)

_RECONCILIATION_LOOKUP_BATCH_SIZE = 256


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


def collect_speculative_direct_results(
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


def prepare_merge_sections(
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


def reconcile_merge_if_ready(
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


def write_merged_region(
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


def load_reconciliation_rows(
    data_root: DataRoot,
    stem: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str], list[dict[str, Any]]],
]:
    polygons_rows = list(_rows(data_root.processed_v2 / "polygons" / f"{stem}.parquet"))
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


def remove_speculative_links(
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


def reconciliation_ref_items(
    polygons: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any], WikipediaTagRef]]:
    return [
        (polygon_id, polygon, ref)
        for polygon_id, polygon in sorted(polygons.items())
        for ref in _polygon_refs(polygon)
    ]


def reconcile_ref_items(
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


def drop_unreferenced_direct_documents(
    documents: dict[str, dict[str, Any]],
    links: dict[tuple[str, str, str], dict[str, Any]],
    direct_document_ids: set[str],
) -> None:
    referenced = {str(row.get("document_id", "")) for row in links.values()}
    for document_id in direct_document_ids - referenced:
        documents.pop(document_id, None)


def write_reconciled_region(
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
