"""Resolve entities and fetch the documents and sections needed for repair."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from osm_polygon_wikidata_only.augmentation.models import document_from_article_row
from osm_polygon_wikidata_only.augmentation.sections import parse_sections
from osm_polygon_wikidata_only.augmentation.steps import AugmentationClient
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_from_article_row,
)
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.ids import article_id
from osm_polygon_wikidata_only.domain.schema import ARTICLE_COLUMNS
from osm_polygon_wikidata_only.enrichment.article_linker import LinkSummary
from osm_polygon_wikidata_only.enrichment.wikidata.models import (
    BatchWikidataClient,
    WikidataClient,
    WikidataEntity,
)
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import language_from_site
from osm_polygon_wikidata_only.enrichment.wikipedia.models import WikipediaClient
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.models import RecoveryRepairError
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.progress import RecoveryProgress
from osm_polygon_wikidata_only.pipeline.completeness import NON_FATAL_FETCH_STATUSES
from osm_polygon_wikidata_only.pipeline.row_construction import article_row

RECOVERY_NETWORK_WORKERS = 8


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
    with ThreadPoolExecutor(max_workers=min(RECOVERY_NETWORK_WORKERS, len(affected_qids))) as executor:
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


__all__ = [
    "RECOVERY_NETWORK_WORKERS",
    "_advance_missing_document",
    "_document_from_recovery_article",
    "_eligible_sitelinks",
    "_fetch_missing_documents",
    "_fetch_missing_documents_parallel",
    "_fetch_qid_documents",
    "_fetch_recovery_article",
    "_fetch_recovery_document",
    "_language_sitelinks",
    "_limit_sitelinks",
    "_missing_recovery_article",
    "_parse_recovery_document",
    "_parse_recovery_documents",
    "_resolve_entities",
    "_resolved_entity_map",
    "_sections_for_new_documents",
    "_select_new_documents",
    "_validate_recovery_fetch",
]


# Public collaborator spellings used by the recovery facade.
resolved_entity_map = _resolved_entity_map
resolve_entities = _resolve_entities
eligible_sitelinks = _eligible_sitelinks
language_sitelinks = _language_sitelinks
limit_sitelinks = _limit_sitelinks
fetch_recovery_document = _fetch_recovery_document
fetch_recovery_article = _fetch_recovery_article
missing_recovery_article = _missing_recovery_article
validate_recovery_fetch = _validate_recovery_fetch
advance_missing_document = _advance_missing_document
document_from_recovery_article = _document_from_recovery_article
fetch_qid_documents = _fetch_qid_documents
fetch_missing_documents = _fetch_missing_documents
fetch_missing_documents_parallel = _fetch_missing_documents_parallel
sections_for_new_documents = _sections_for_new_documents
select_new_documents = _select_new_documents
parse_recovery_documents = _parse_recovery_documents
parse_recovery_document = _parse_recovery_document
