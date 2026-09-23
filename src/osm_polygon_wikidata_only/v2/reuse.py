"""Lossless V1 reuse and V2 relationship assembly.

The V2 build starts from finalized V1 shards.  Existing Wikipedia,
Wikivoyage, Wikidata, and polygon-link rows are copied unchanged where
possible.  Only direct Wikipedia-tag relationships not already represented
by a V1 document are fetched.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.v2.extractor import V2ExtractedPbf
from osm_polygon_wikidata_only.v2.reuse_direct import (
    build_region_sections as _build_region_sections,
)
from osm_polygon_wikidata_only.v2.reuse_direct import (
    update_polygon_text_fields as _update_polygon_text_fields,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    SIDECAR_SUBDIRS,
    V1RegionData,
    copy_v1_sidecars,
    load_v1_region,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    load_merge_inputs as _load_merge_inputs,
)
from osm_polygon_wikidata_only.v2.reuse_load import (
    open_fetch_checkpoint as _fetch_checkpoint,
)
from osm_polygon_wikidata_only.v2.reuse_reconcile import (
    collect_speculative_direct_results as _collect_speculative_direct_results,
)
from osm_polygon_wikidata_only.v2.reuse_reconcile import (
    drop_unreferenced_direct_documents as _drop_unreferenced_direct_documents,
)
from osm_polygon_wikidata_only.v2.reuse_reconcile import (
    load_reconciliation_rows as _load_reconciliation_rows,
)
from osm_polygon_wikidata_only.v2.reuse_reconcile import (
    prepare_merge_sections as _prepare_merge_sections,
)
from osm_polygon_wikidata_only.v2.reuse_reconcile import (
    reconcile_merge_if_ready as _reconcile_merge_if_ready,
)
from osm_polygon_wikidata_only.v2.reuse_reconcile import (
    reconcile_ref_items as _reconcile_ref_items,
)
from osm_polygon_wikidata_only.v2.reuse_reconcile import (
    reconciliation_ref_items as _reconciliation_ref_items,
)
from osm_polygon_wikidata_only.v2.reuse_reconcile import (
    remove_speculative_links as _remove_speculative_links,
)
from osm_polygon_wikidata_only.v2.reuse_reconcile import (
    write_merged_region as _write_merged_region,
)
from osm_polygon_wikidata_only.v2.reuse_reconcile import (
    write_reconciled_region as _write_reconciled_region,
)
from osm_polygon_wikidata_only.v2.sections import SectionClient
from osm_polygon_wikidata_only.v2.sections import (  # noqa: F401 - compatibility seam
    build_missing_sections as _build_missing_sections,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "SIDECAR_SUBDIRS",
    "SectionClient",
    "V1RegionData",
    "copy_v1_sidecars",
    "load_v1_region",
    "merge_v2_region",
    "reconcile_v2_region",
]


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
