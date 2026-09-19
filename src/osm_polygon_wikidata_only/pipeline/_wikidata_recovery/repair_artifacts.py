"""Build recoverable Wikidata artifacts and durable recovery batches."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from osm_polygon_wikidata_only.augmentation.models import document_from_article_row
from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.augmentation.sections import parse_sections
from osm_polygon_wikidata_only.augmentation.steps import (
    AugmentationClient,
    build_wikidata_facts,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
)
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.ids import article_id
from osm_polygon_wikidata_only.domain.schema import ARTICLE_COLUMNS
from osm_polygon_wikidata_only.enrichment.article_linker import PREFERRED_LANGUAGES, LinkSummary
from osm_polygon_wikidata_only.enrichment.wikidata.models import (
    BatchWikidataClient,
    WikidataClient,
    WikidataEntity,
)
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
    language_from_site,
    qids_from_osm_tag,
)
from osm_polygon_wikidata_only.enrichment.wikipedia.models import WikipediaClient
from osm_polygon_wikidata_only.pipeline.completeness import NON_FATAL_FETCH_STATUSES
from osm_polygon_wikidata_only.pipeline.row_construction import article_row
from osm_polygon_wikidata_only.utils.json import dumps
from osm_polygon_wikidata_only.utils.request_scheduler import RequestSchedulerSnapshot
from osm_polygon_wikidata_only.utils.retry import (
    _cancel_pending_retries,
    _reset_retry_cancellation,
)

from .checkpoints import (
    RECOVERY_QID_BATCH_SIZE,
    RecoveryBatchArtifacts,
    RecoveryCheckpointStore,
)
from .models import RecoveryRepairError
from .progress import RecoveryHeartbeat, RecoveryProgress

RECOVERY_NETWORK_WORKERS = 8
RECOVERY_BATCH_WINDOW = 3


def _recovery_qid_batches(affected_qids: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Split affected QIDs into deterministic checkpoint-sized batches."""
    return [
        affected_qids[start : start + RECOVERY_QID_BATCH_SIZE]
        for start in range(0, len(affected_qids), RECOVERY_QID_BATCH_SIZE)
    ]


def _load_recovery_checkpoints(
    stem: str,
    batches: list[tuple[str, ...]],
    checkpoint_store: RecoveryCheckpointStore,
    emit: Callable[[str], None],
) -> tuple[dict[int, RecoveryBatchArtifacts], list[tuple[int, tuple[str, ...]]]]:
    """Load reusable recovery checkpoints and return missing batches."""
    completed: dict[int, RecoveryBatchArtifacts] = {}
    missing: list[tuple[int, tuple[str, ...]]] = []
    for index, batch_qids in enumerate(batches):
        artifacts = checkpoint_store.load(index, batch_qids)
        if artifacts is None:
            missing.append((index, batch_qids))
            continue
        completed[index] = artifacts
        emit(
            f"Wikidata recovery {stem}: batch {index + 1}/{len(batches)} "
            f"reused durable checkpoint ({len(batch_qids)} QIDs)"
        )
    return completed, missing


def _build_and_checkpoint(
    index: int,
    batch_qids: tuple[str, ...],
    *,
    stem: str,
    batch_total: int,
    checkpoint_store: RecoveryCheckpointStore,
    build_batch: Callable[[tuple[str, ...], RecoveryProgress], RecoveryBatchArtifacts],
    emit: Callable[[str], None],
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None,
) -> tuple[int, RecoveryBatchArtifacts]:
    """Build one recovery batch and persist its durable checkpoint."""
    progress = RecoveryProgress(stem, batch_total, scheduler_snapshot=scheduler_snapshot)
    progress.start_batch(index + 1, batch_qids)
    with RecoveryHeartbeat(progress, emit):
        artifacts = build_batch(batch_qids, progress)
    checkpoint_store.save(index, artifacts)
    progress.checkpoint_saved(
        documents=len(artifacts.documents),
        sections=len(artifacts.sections),
        facts=len(artifacts.facts),
    )
    emit(progress.message())
    return index, artifacts


def _collect_recovery_futures(
    futures: list[Future[tuple[int, RecoveryBatchArtifacts]]],
    completed: dict[int, RecoveryBatchArtifacts],
    *,
    as_completed_fn: Callable[[object], Any] | None = None,
) -> None:
    """Collect completed recovery futures and cancel the remainder on failure."""
    completion = as_completed if as_completed_fn is None else as_completed_fn
    try:
        for future in completion(futures):
            index, artifacts = future.result()
            completed[index] = artifacts
    except BaseException:
        _cancel_pending_retries()
        for future in futures:
            future.cancel()
        raise


def _run_missing_recovery_batches(
    missing: list[tuple[int, tuple[str, ...]]],
    completed: dict[int, RecoveryBatchArtifacts],
    *,
    stem: str,
    batch_total: int,
    checkpoint_store: RecoveryCheckpointStore,
    build_batch: Callable[[tuple[str, ...], RecoveryProgress], RecoveryBatchArtifacts],
    emit: Callable[[str], None],
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None,
    batch_window: int,
    as_completed_fn: Callable[[object], Any] | None = None,
) -> None:
    """Build missing batches concurrently while preserving retry cancellation."""
    _reset_retry_cancellation()
    try:
        with ThreadPoolExecutor(max_workers=min(batch_window, len(missing))) as executor:
            futures = [
                executor.submit(
                    _build_and_checkpoint,
                    index,
                    batch_qids,
                    stem=stem,
                    batch_total=batch_total,
                    checkpoint_store=checkpoint_store,
                    build_batch=build_batch,
                    emit=emit,
                    scheduler_snapshot=scheduler_snapshot,
                )
                for index, batch_qids in missing
            ]
            _collect_recovery_futures(
                futures,
                completed,
                as_completed_fn=as_completed_fn,
            )
    finally:
        _reset_retry_cancellation()


def _execute_recovery_batches(
    *,
    stem: str,
    affected_qids: tuple[str, ...],
    checkpoint_store: RecoveryCheckpointStore,
    build_batch: Callable[[tuple[str, ...], RecoveryProgress], RecoveryBatchArtifacts],
    emit: Callable[[str], None],
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None = None,
    batch_window: int = RECOVERY_BATCH_WINDOW,
    as_completed_fn: Callable[[object], Any] | None = None,
) -> list[RecoveryBatchArtifacts]:
    """Build independent recovery batches concurrently and return input order."""
    if batch_window < 1:
        raise ValueError("batch_window must be at least 1")
    batches = _recovery_qid_batches(affected_qids)
    batch_total = len(batches)
    completed, missing = _load_recovery_checkpoints(stem, batches, checkpoint_store, emit)

    if missing:
        _run_missing_recovery_batches(
            missing,
            completed,
            stem=stem,
            batch_total=batch_total,
            checkpoint_store=checkpoint_store,
            build_batch=build_batch,
            emit=emit,
            scheduler_snapshot=scheduler_snapshot,
            batch_window=batch_window,
            as_completed_fn=as_completed_fn,
        )
    return [completed[index] for index in range(batch_total)]


def _build_batch_artifacts(
    qids: tuple[str, ...],
    *,
    existing_documents: list[dict[str, Any]],
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    augmentation_client: AugmentationClient,
    settings: Settings,
    progress: RecoveryProgress,
) -> RecoveryBatchArtifacts:
    progress.set_stage("Wikidata entities", total=len(qids))
    entities = _resolve_entities(wikidata_client, qids)
    progress.advance(len(qids))
    documents = _fetch_missing_documents(
        qids,
        entities=entities,
        existing_documents=existing_documents,
        wikipedia_client=wikipedia_client,
        settings=settings,
        progress=progress,
    )
    document_ids = {str(row["document_id"]) for row in documents}
    sections = _sections_for_new_documents(
        documents,
        document_ids,
        augmentation_client=augmentation_client,
        progress=progress,
    )
    progress.set_stage("Wikidata facts", total=len(qids))
    raw_entities = augmentation_client.entities(list(qids), props="sitelinks|claims")
    missing_raw = sorted(set(qids) - set(raw_entities))
    if missing_raw:
        raise RecoveryRepairError(f"Augmentation Wikidata response omitted QIDs: {missing_raw}")
    facts = [
        fact.to_dict()
        for fact in build_wikidata_facts(
            augmentation_client,
            entities={qid: raw_entities[qid] for qid in qids},
            progress=AugmentationProgress(),
        )
    ]
    progress.advance(len(qids), facts=len(facts))
    return RecoveryBatchArtifacts(
        qids=qids,
        documents=tuple(documents),
        sections=tuple(sections),
        facts=tuple(facts),
    )


def _resolved_entity_map(
    qids: tuple[str, ...],
    values: list[WikidataEntity | None],
) -> dict[str, WikidataEntity]:
    """Validate a Wikidata response and map each requested QID to its entity."""
    if len(values) != len(qids):
        raise RecoveryRepairError("Wikidata client returned the wrong result count")
    resolved: dict[str, WikidataEntity] = {}
    for qid, entity in zip(qids, values, strict=True):
        if entity is None:
            raise RecoveryRepairError(f"Affected QID became authoritatively missing: {qid}")
        resolved[qid] = entity
    return resolved


def _resolve_entities(
    client: WikidataClient,
    qids: tuple[str, ...],
) -> dict[str, WikidataEntity]:
    if isinstance(client, BatchWikidataClient):
        values = client.get_entities(qids)
    else:
        values = [client.get_entity(qid) for qid in qids]
    return _resolved_entity_map(qids, values)


def _eligible_sitelinks(entity: WikidataEntity, settings: Settings) -> list[tuple[str, str]]:
    allowed = set(settings.languages) if settings.languages is not None else None
    links = _language_sitelinks(entity, allowed)
    return _limit_sitelinks(links, settings.max_articles_per_qid)


def _language_sitelinks(
    entity: WikidataEntity,
    allowed: set[str] | None,
) -> list[tuple[str, str]]:
    return [
        (site, title)
        for site, title in sorted(entity.sitelinks.items())
        if allowed is None or language_from_site(site) in allowed
    ]


def _limit_sitelinks(
    links: list[tuple[str, str]],
    max_articles_per_qid: int | None,
) -> list[tuple[str, str]]:
    if max_articles_per_qid is None:
        return links
    return links[: max(0, max_articles_per_qid)]


def _fetch_recovery_document(
    qid: str,
    site: str,
    title: str,
    *,
    entity: WikidataEntity,
    summary: LinkSummary,
    wikipedia_client: WikipediaClient,
    settings: Settings,
    progress: RecoveryProgress | None,
) -> dict[str, Any] | None:
    """Fetch and normalize one missing Wikipedia document."""
    language = language_from_site(site)
    result = _fetch_recovery_article(
        language,
        site,
        title,
        entity=entity,
        wikipedia_client=wikipedia_client,
        settings=settings,
    )
    summary.statuses[site] = result.status
    _validate_recovery_fetch(qid, site, result.status, result.error)
    if _missing_recovery_article(result):
        _advance_missing_document(progress)
        return None
    summary.articles.append(result.article)
    return _document_from_recovery_article(qid, language, result.article, summary, progress)


def _fetch_recovery_article(
    language: str,
    site: str,
    title: str,
    *,
    entity: WikidataEntity,
    wikipedia_client: WikipediaClient,
    settings: Settings,
) -> Any:
    """Call the Wikipedia client with the recovery entity context."""
    return wikipedia_client.fetch_article(
        language,
        site,
        title,
        wikidata_label=entity.labels.get(language) or entity.labels.get("en", ""),
        wikidata_description=entity.descriptions.get(language) or entity.descriptions.get("en", ""),
        wikidata_aliases=entity.aliases.get(language) or entity.aliases.get("en", []),
        fetch_full_text=settings.fetch_full_text,
    )


def _missing_recovery_article(result: Any) -> bool:
    """Return whether a fetch completed without an article payload."""
    return result.article is None or result.status == "article_not_found"


def _validate_recovery_fetch(qid: str, site: str, status: str, error: str) -> None:
    """Reject a terminal Wikipedia fetch status that cannot be preserved."""
    if status not in NON_FATAL_FETCH_STATUSES and status != "ok":
        raise RecoveryRepairError(
            f"Incomplete Wikipedia recovery for {qid}:{site} ({status}): {error}"
        )


def _advance_missing_document(progress: RecoveryProgress | None) -> None:
    """Advance progress for an absent but terminally handled document."""
    if progress is not None:
        progress.advance()


def _document_from_recovery_article(
    qid: str,
    language: str,
    article_result: Any,
    summary: LinkSummary,
    progress: RecoveryProgress | None,
) -> dict[str, Any]:
    """Convert a fetched article into the canonical Wikipedia document row."""
    identifier = article_id(qid, language, article_result.page_id, article_result.revision_id)
    article = article_row(identifier, qid, article_result, summary)
    document = wikipedia_document_from_article_row(article.__dict__)
    if progress is not None:
        progress.advance(documents=1)
    return document.to_dict()


def _fetch_qid_documents(
    qid: str,
    *,
    entity: WikidataEntity,
    existing: set[tuple[str, str, str]],
    wikipedia_client: WikipediaClient,
    settings: Settings,
    progress: RecoveryProgress | None,
) -> list[dict[str, Any]]:
    """Fetch all missing eligible documents for one QID."""
    summary = LinkSummary(qid=qid, entity=entity)
    documents: list[dict[str, Any]] = []
    for site, title in _eligible_sitelinks(entity, settings):
        if (qid, site, title) in existing:
            if progress is not None:
                progress.advance()
            continue
        document = _fetch_recovery_document(
            qid,
            site,
            title,
            entity=entity,
            summary=summary,
            wikipedia_client=wikipedia_client,
            settings=settings,
            progress=progress,
        )
        if document is not None:
            documents.append(document)
    return documents


def _fetch_missing_documents(
    affected_qids: tuple[str, ...],
    *,
    entities: dict[str, WikidataEntity],
    existing_documents: list[dict[str, Any]],
    wikipedia_client: WikipediaClient,
    settings: Settings,
    progress: RecoveryProgress | None = None,
) -> list[dict[str, Any]]:
    existing = {
        (str(row["wikidata"]), str(row["site"]), str(row["title"])) for row in existing_documents
    }
    total_sites = sum(len(_eligible_sitelinks(entities[qid], settings)) for qid in affected_qids)
    if progress is not None:
        progress.set_stage("Wikipedia documents", total=total_sites)
    return _fetch_missing_documents_parallel(
        affected_qids,
        entities=entities,
        existing=existing,
        wikipedia_client=wikipedia_client,
        settings=settings,
        progress=progress,
    )


def _fetch_missing_documents_parallel(
    affected_qids: tuple[str, ...],
    *,
    entities: dict[str, WikidataEntity],
    existing: set[tuple[str, str, str]],
    wikipedia_client: WikipediaClient,
    settings: Settings,
    progress: RecoveryProgress | None,
) -> list[dict[str, Any]]:
    """Fetch affected QIDs concurrently while retaining input order."""
    if not affected_qids:
        return []
    with ThreadPoolExecutor(
        max_workers=min(RECOVERY_NETWORK_WORKERS, len(affected_qids))
    ) as executor:
        per_qid = executor.map(
            lambda qid: _fetch_qid_documents(
                qid,
                entity=entities[qid],
                existing=existing,
                wikipedia_client=wikipedia_client,
                settings=settings,
                progress=progress,
            ),
            affected_qids,
        )
        return [document for documents in per_qid for document in documents]


def _summarize_polygon_links(
    polygon_links: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return unique article IDs and languages in deterministic order."""
    return (
        sorted({str(link["article_id"]) for link in polygon_links}),
        sorted({str(link["language"]) for link in polygon_links}),
    )


def _preferred_language(languages: list[str]) -> str:
    """Choose the configured preferred language, falling back to sorted input."""
    best = next((language for language in PREFERRED_LANGUAGES if language in languages), "")
    if not best and languages:
        return languages[0]
    return best


def _has_article_text(
    article_ids: list[str],
    documents_by_article: dict[str, dict[str, Any]],
) -> bool:
    """Return whether at least one linked document has non-empty full text."""
    return any(
        bool(str(documents_by_article[article]["full_text"]).strip()) for article in article_ids
    )


def _recompute_polygon_row(
    original: dict[str, Any],
    polygon_links: list[dict[str, Any]],
    documents_by_article: dict[str, dict[str, Any]],
    affected_qids: set[str],
) -> tuple[dict[str, Any], str]:
    """Recompute one affected polygon's derived Wikipedia fields."""
    row = dict(original)
    if not set(qids_from_osm_tag(str(row["wikidata"]))) & affected_qids:
        return row, ""
    article_ids, languages = _summarize_polygon_links(polygon_links)
    best = _preferred_language(languages)
    row.update(
        {
            "has_wikipedia": bool(article_ids),
            "wikipedia_language_count": len(languages),
            "wikipedia_languages": dumps(languages),
            "wikipedia_article_count": len(article_ids),
            "has_english_wikipedia": "en" in languages,
            "has_french_wikipedia": "fr" in languages,
            "text_available": _has_article_text(article_ids, documents_by_article),
            "best_language": best,
        }
    )
    return row, best


def _recompute_polygon_rows(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    affected_qids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Recompute derived fields for all affected polygons."""
    documents_by_article = {str(row["article_id"]): row for row in documents}
    links_by_polygon: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        links_by_polygon.setdefault(str(link["polygon_id"]), []).append(link)
    updated: list[dict[str, Any]] = []
    best_by_polygon: dict[str, str] = {}
    for original in polygons:
        polygon_id = str(original["polygon_id"])
        row, best = _recompute_polygon_row(
            original,
            links_by_polygon.get(polygon_id, []),
            documents_by_article,
            affected_qids,
        )
        updated.append(row)
        if best:
            best_by_polygon[polygon_id] = best
    return updated, best_by_polygon


def _apply_best_language_links(
    links: list[dict[str, Any]],
    best_by_polygon: dict[str, str],
) -> list[dict[str, Any]]:
    """Apply the recomputed best-language flag to Wikipedia links."""
    updated: list[dict[str, Any]] = []
    for original in links:
        row = dict(original)
        polygon_id = str(row["polygon_id"])
        if polygon_id in best_by_polygon:
            row["is_best_language"] = str(row["language"]) == best_by_polygon[polygon_id]
        updated.append(row)
    return updated


def _recompute_affected_polygon_fields(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    affected_qids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updated_polygons, best_by_polygon = _recompute_polygon_rows(
        polygons, links, documents, affected_qids
    )
    return updated_polygons, _apply_best_language_links(links, best_by_polygon)


def _sections_for_new_documents(
    documents: list[dict[str, Any]],
    new_document_ids: set[str],
    *,
    augmentation_client: AugmentationClient,
    progress: RecoveryProgress | None = None,
) -> list[dict[str, Any]]:
    selected = _select_new_documents(documents, new_document_ids)
    if progress is not None:
        progress.set_stage("Wikipedia sections", total=len(selected))
    return _parse_recovery_documents(
        selected, augmentation_client=augmentation_client, progress=progress
    )


def _select_new_documents(
    documents: list[dict[str, Any]], new_document_ids: set[str]
) -> list[dict[str, Any]]:
    """Select only documents that need section parsing."""
    return [row for row in documents if str(row["document_id"]) in new_document_ids]


def _parse_recovery_documents(
    selected: list[dict[str, Any]],
    *,
    augmentation_client: AugmentationClient,
    progress: RecoveryProgress | None,
) -> list[dict[str, Any]]:
    """Parse selected recovery documents concurrently in input order."""
    if not selected:
        return []
    with ThreadPoolExecutor(max_workers=min(RECOVERY_NETWORK_WORKERS, len(selected))) as executor:
        per_document = executor.map(
            lambda document_row: _parse_recovery_document(
                document_row,
                augmentation_client=augmentation_client,
                progress=progress,
            ),
            selected,
        )
        return [section for sections in per_document for section in sections]


def _parse_recovery_document(
    document_row: dict[str, Any],
    *,
    augmentation_client: AugmentationClient,
    progress: RecoveryProgress | None,
) -> list[dict[str, Any]]:
    """Fetch and parse sections for one newly recovered document."""
    article = {column: document_row[column] for column in ARTICLE_COLUMNS}
    document = document_from_article_row(article)
    html = augmentation_client.parse_html(document.project, document.language, document.revision_id)
    parsed = [section.to_dict() for section in parse_sections(document, html)]
    if progress is not None:
        progress.advance(sections=len(parsed))
    return parsed
