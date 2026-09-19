"""Resolve upstream entities and classify audited QID relationships."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from osm_polygon_wikidata_only.enrichment.wikidata.models import (
    BatchWikidataClient,
    WikidataClient,
    WikidataEntity,
)
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import language_from_site
from osm_polygon_wikidata_only.utils.retry import (
    _cancel_pending_retries,
    _reset_retry_cancellation,
)

from .audit_types import RegionScan
from .models import QidAuditResult, RecoveryClassification, RegionAuditResult

_UPSTREAM_BATCH_WINDOW = 3


def resolve_entities(
    client: WikidataClient,
    qids: list[str],
    *,
    batch_size: int,
    progress: Callable[[int, int], None] | None = None,
    as_completed_fn: Callable[[object], Any] | None = None,
) -> tuple[dict[str, WikidataEntity | None], int]:
    if isinstance(client, BatchWikidataClient):
        return _resolve_batch_entities(
            client,
            qids,
            batch_size=batch_size,
            progress=progress,
            as_completed_fn=as_completed_fn,
        )
    return _resolve_single_entities(client, qids, progress=progress)


def _resolve_single_entities(
    client: WikidataClient,
    qids: list[str],
    *,
    progress: Callable[[int, int], None] | None,
) -> tuple[dict[str, WikidataEntity | None], int]:
    resolved: dict[str, WikidataEntity | None] = {}
    for index, qid in enumerate(qids, start=1):
        resolved[qid] = client.get_entity(qid)
        if progress is not None:
            progress(index, len(qids))
    return resolved, 0


def _resolve_batch_entities(
    client: BatchWikidataClient,
    qids: list[str],
    *,
    batch_size: int,
    progress: Callable[[int, int], None] | None,
    as_completed_fn: Callable[[object], Any] | None,
) -> tuple[dict[str, WikidataEntity | None], int]:
    chunks = [qids[start : start + batch_size] for start in range(0, len(qids), batch_size)]
    if not chunks:
        return {}, 0
    completed = _run_batch_futures(
        client,
        chunks,
        total=len(qids),
        progress=progress,
        as_completed_fn=as_completed_fn,
    )
    resolved: dict[str, WikidataEntity | None] = {}
    cache_hits = 0
    for index in range(len(chunks)):
        chunk, results, hits = completed[index]
        resolved.update(zip(chunk, results, strict=True))
        cache_hits += hits
    return resolved, cache_hits


def _resolve_batch_chunk(
    client: BatchWikidataClient,
    index: int,
    chunk: list[str],
) -> tuple[int, list[str], list[WikidataEntity | None], int]:
    results = client.get_entities(chunk)
    if len(results) != len(chunk):
        raise RuntimeError("Wikidata batch client returned the wrong result count")
    raw_hits = getattr(client, "last_batch_cache_hits", 0)
    hits = raw_hits if isinstance(raw_hits, int) else 0
    return index, chunk, results, hits


def _run_batch_futures(
    client: BatchWikidataClient,
    chunks: list[list[str]],
    *,
    total: int,
    progress: Callable[[int, int], None] | None,
    as_completed_fn: Callable[[object], Any] | None,
) -> dict[int, tuple[list[str], list[WikidataEntity | None], int]]:
    completed: dict[int, tuple[list[str], list[WikidataEntity | None], int]] = {}
    completed_qids = 0
    _reset_retry_cancellation()
    try:
        with ThreadPoolExecutor(max_workers=min(_UPSTREAM_BATCH_WINDOW, len(chunks))) as executor:
            futures = [
                executor.submit(_resolve_batch_chunk, client, index, chunk)
                for index, chunk in enumerate(chunks)
            ]
            try:
                completion = as_completed if as_completed_fn is None else as_completed_fn
                for future in completion(futures):
                    index, chunk, results, hits = future.result()
                    completed[index] = (chunk, results, hits)
                    completed_qids += len(chunk)
                    report_batch_progress(progress, completed_qids, total)
            except BaseException:
                _cancel_pending_retries()
                for future in futures:
                    future.cancel()
                raise
    finally:
        _reset_retry_cancellation()
    return completed


def report_batch_progress(
    progress: Callable[[int, int], None] | None,
    completed: int,
    total: int,
) -> None:
    if progress is not None:
        progress(completed, total)


def progress_checkpoint(completed: int, total: int, *, every: int) -> bool:
    return total > 0 and (completed == 1 or completed == total or completed % every == 0)


def eligible_sitelinks(
    entity: WikidataEntity | None,
    *,
    languages: tuple[str, ...] | None,
    max_articles_per_qid: int | None,
) -> tuple[tuple[str, str], ...]:
    if entity is None:
        return ()
    return limit_sitelinks(filtered_sitelinks(entity, languages), max_articles_per_qid)


def filtered_sitelinks(
    entity: WikidataEntity,
    languages: tuple[str, ...] | None,
) -> tuple[tuple[str, str], ...]:
    allowed = set(languages) if languages is not None else None
    return tuple(
        (site, title)
        for site, title in sorted(entity.sitelinks.items())
        if allowed is None or language_from_site(site) in allowed
    )


def limit_sitelinks(
    sitelinks: tuple[tuple[str, str], ...],
    max_articles_per_qid: int | None,
) -> tuple[tuple[str, str], ...]:
    if max_articles_per_qid is None:
        return sitelinks
    return sitelinks[: max(0, max_articles_per_qid)]


def classify_region(
    scan: RegionScan,
    entities: dict[str, WikidataEntity | None],
    eligible: dict[str, tuple[tuple[str, str], ...]],
) -> RegionAuditResult:
    if scan.blocked_reason:
        return blocked_region_result(scan)
    missing = dict(scan.missing_polygon_ids_by_qid)
    classifications: list[tuple[str, RecoveryClassification]] = []
    affected: list[tuple[str, tuple[str, ...]]] = []
    for qid, _polygon_ids in scan.polygon_ids_by_qid:
        state, affected_polygon_ids = classify_qid(qid, missing, entities, eligible)
        affected.extend(affected_qid_entry(qid, affected_polygon_ids))
        classifications.append((qid, state))
    return classified_region_result(scan, classifications, affected)


def classified_region_result(
    scan: RegionScan,
    classifications: list[tuple[str, RecoveryClassification]],
    affected: list[tuple[str, tuple[str, ...]]],
) -> RegionAuditResult:
    affected_qids = tuple(qid for qid, _ in affected)
    affected_polygon_ids = {polygon_id for _, polygon_ids in affected for polygon_id in polygon_ids}
    return RegionAuditResult(
        stem=scan.stem,
        fingerprints=scan.fingerprints,
        classifications=tuple(classifications),
        polygon_ids_by_qid=scan.polygon_ids_by_qid,
        affected_polygon_ids_by_qid=tuple(affected),
        affected_qids=affected_qids,
        affected_polygon_count=len(affected_polygon_ids),
        orphan_fact_ids=scan.orphan_fact_ids,
        orphan_document_ids=scan.orphan_document_ids,
    )


def affected_qid_entry(
    qid: str,
    polygon_ids: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not polygon_ids:
        return ()
    return ((qid, polygon_ids),)


def blocked_region_result(scan: RegionScan) -> RegionAuditResult:
    return RegionAuditResult(
        stem=scan.stem,
        fingerprints=scan.fingerprints,
        classifications=(),
        polygon_ids_by_qid=(),
        affected_polygon_ids_by_qid=(),
        affected_qids=(),
        affected_polygon_count=0,
        orphan_fact_ids=scan.orphan_fact_ids,
        orphan_document_ids=scan.orphan_document_ids,
        blocked_reason=scan.blocked_reason,
    )


def classify_qid(
    qid: str,
    missing: Mapping[str, tuple[str, ...]],
    entities: Mapping[str, WikidataEntity | None],
    eligible: Mapping[str, tuple[tuple[str, str], ...]],
) -> tuple[RecoveryClassification, tuple[str, ...]]:
    if qid not in missing:
        return RecoveryClassification.CURRENT, ()
    if entities[qid] is None:
        return RecoveryClassification.AUTHORITATIVE_MISSING, ()
    if not eligible[qid]:
        return RecoveryClassification.AUTHORITATIVE_NO_SITELINK, ()
    return RecoveryClassification.REPAIR_REQUIRED, missing[qid]


def global_qid_results(
    regions: list[RegionAuditResult],
    eligible: dict[str, tuple[tuple[str, str], ...]],
) -> list[QidAuditResult]:
    states, region_names, polygon_ids = collect_qid_data(regions)
    return [
        build_qid_result(qid, states, region_names, polygon_ids, eligible)
        for qid in sorted(states)
    ]


def collect_qid_data(
    regions: list[RegionAuditResult],
) -> tuple[
    dict[str, list[RecoveryClassification]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    states: dict[str, list[RecoveryClassification]] = {}
    region_names: dict[str, set[str]] = {}
    polygon_ids: dict[str, set[str]] = {}
    for region in regions:
        collect_region_qid_data(region, states, region_names, polygon_ids)
    return states, region_names, polygon_ids


def collect_region_qid_data(
    region: RegionAuditResult,
    states: dict[str, list[RecoveryClassification]],
    region_names: dict[str, set[str]],
    polygon_ids: dict[str, set[str]],
) -> None:
    region_polygon_ids = dict(region.polygon_ids_by_qid)
    for qid, state in region.classifications:
        states.setdefault(qid, []).append(state)
        region_names.setdefault(qid, set()).add(region.stem)
        polygon_ids.setdefault(qid, set()).update(region_polygon_ids.get(qid, ()))


def build_qid_result(
    qid: str,
    states: Mapping[str, list[RecoveryClassification]],
    region_names: Mapping[str, set[str]],
    polygon_ids: Mapping[str, set[str]],
    eligible: Mapping[str, tuple[tuple[str, str], ...]],
) -> QidAuditResult:
    priority = (
        RecoveryClassification.BLOCKED,
        RecoveryClassification.REPAIR_REQUIRED,
        RecoveryClassification.CURRENT,
        RecoveryClassification.AUTHORITATIVE_NO_ARTICLE,
        RecoveryClassification.AUTHORITATIVE_NO_SITELINK,
        RecoveryClassification.AUTHORITATIVE_MISSING,
    )
    state = next(candidate for candidate in priority if candidate in states[qid])
    return QidAuditResult(
        qid=qid,
        state=state,
        regions=tuple(sorted(region_names[qid])),
        polygon_ids=tuple(sorted(polygon_ids[qid])),
        sitelinks=eligible.get(qid, ()),
    )


__all__ = [
    "affected_qid_entry",
    "blocked_region_result",
    "build_qid_result",
    "classified_region_result",
    "classify_qid",
    "classify_region",
    "collect_qid_data",
    "collect_region_qid_data",
    "eligible_sitelinks",
    "filtered_sitelinks",
    "global_qid_results",
    "limit_sitelinks",
    "progress_checkpoint",
    "resolve_entities",
]
