"""Repair Wikidata enrichment artifacts for a region."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.models import document_from_article_row
from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.augmentation.schema import (
    FACT_COLUMNS,
    SECTION_COLUMNS,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.sections import parse_sections
from osm_polygon_wikidata_only.augmentation.steps import (
    CONTRACT_VERSION,
    AugmentationClient,
    build_wikidata_facts,
    sha256_file,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    WIKIPEDIA_DOCUMENT_COLUMNS,
    wikipedia_document_from_article_row,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.ids import article_id
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    CANONICAL_COLUMNS,
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_COLUMNS,
    polygon_article_schema,
    polygon_schema,
)
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
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.completeness import NON_FATAL_FETCH_STATUSES
from osm_polygon_wikidata_only.pipeline.row_construction import article_row
from osm_polygon_wikidata_only.utils.json import dumps
from osm_polygon_wikidata_only.utils.request_scheduler import RequestSchedulerSnapshot
from osm_polygon_wikidata_only.utils.retry import (
    _cancel_pending_retries,
    _reset_retry_cancellation,
)

from .audit import (
    RECOVERY_CONTRACT_VERSION,
    audit_wikidata_integrity,
    record_region_recovery_receipt,
)
from .checkpoints import (
    RECOVERY_QID_BATCH_SIZE,
    RecoveryBatchArtifacts,
    RecoveryCheckpointStore,
    recovery_plan_key,
)
from .link_rows import (
    canonical_wikipedia_links_to_legacy as _canonical_wikipedia_links_to_legacy,
)
from .link_rows import (
    legacy_wikipedia_links_to_canonical as _legacy_wikipedia_links_to_canonical,
)
from .link_rows import merge_links as _merge_links
from .models import (
    RecoveryClassification,
    RecoveryRepairError,
    RecoveryRepairResult,
    RegionAuditResult,
)
from .progress import RecoveryHeartbeat, RecoveryProgress
from .storage import read_table as _read_table
from .storage import region_paths as _region_paths
from .storage import write_table as _write_table
from .transaction import (
    commit_replacements,
    recover_interrupted_transactions,
    transaction_directory,
)
from .validation import (
    validate_existing_rows as _validate_existing_rows,
)
from .validation import (
    validate_preservation as _validate_preservation,
)

RECOVERY_NETWORK_WORKERS = 8
RECOVERY_BATCH_WINDOW = 3


def repair_wikidata_region(
    data_root: DataRoot,
    region: RegionAuditResult,
    *,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    augmentation_client: AugmentationClient,
    settings: Settings,
    before_commit: Callable[[], None] | None = None,
    log: Callable[[str], None] | None = None,
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None = None,
) -> RecoveryRepairResult:
    """Repair only the affected QID relationships in one finalized shard."""
    if region.blocked_reason:
        raise RecoveryRepairError(region.blocked_reason)
    if not region.requires_repair:
        return RecoveryRepairResult(region.stem, False, (), 0, (), False)

    transaction_root = data_root.cache / "wikidata_recovery" / "transactions"
    recover_interrupted_transactions(transaction_root)
    stem = region.stem
    paths = _region_paths(data_root, stem)
    polygons = _read_table(paths["polygons"], polygon_schema())
    links_schema = pq.read_schema(paths["links"])  # type: ignore[no-untyped-call]
    documents = _read_table(paths["documents"], wikipedia_document_schema())
    canonical_links = links_schema.equals(polygon_document_link_schema(), check_metadata=True)
    affected_qid_set = set(region.affected_qids)
    if canonical_links:
        stored_links = _read_table(paths["links"], polygon_document_link_schema())
        preserved_wikivoyage_links = [
            dict(row) for row in stored_links if row["project"] == "wikivoyage"
        ]
        links = _canonical_wikipedia_links_to_legacy(
            stored_links,
            documents,
            polygons,
            affected_qids=affected_qid_set,
        )
    elif links_schema.equals(polygon_article_schema(), check_metadata=True):
        stored_links = _read_table(paths["links"], polygon_article_schema())
        preserved_wikivoyage_links = []
        links = stored_links
    else:
        raise RecoveryRepairError(f"Recovery input schema mismatch: {paths['links']}")
    sections = _read_table(paths["sections"], section_schema())
    facts = _read_table(paths["facts"], fact_schema())
    orphan_fact_ids = set(region.orphan_fact_ids)
    orphan_document_ids = set(region.orphan_document_ids)
    retained_facts = [row for row in facts if str(row["fact_id"]) not in orphan_fact_ids]
    if len(facts) - len(retained_facts) != len(orphan_fact_ids):
        raise RecoveryRepairError("Recovery plan contains stale or duplicate orphan fact IDs")
    orphan_article_ids = {
        str(row["article_id"])
        for row in documents
        if str(row["document_id"]) in orphan_document_ids
    }
    retained_documents = [
        row for row in documents if str(row["document_id"]) not in orphan_document_ids
    ]
    if len(documents) - len(retained_documents) != len(orphan_document_ids):
        raise RecoveryRepairError("Recovery plan contains stale or duplicate orphan document IDs")
    retained_sections = [
        row for row in sections if str(row["document_id"]) not in orphan_document_ids
    ]
    retained_links = [row for row in links if str(row["article_id"]) not in orphan_article_ids]
    _validate_existing_rows(
        polygons,
        retained_links,
        retained_documents,
        retained_sections,
        retained_facts,
    )

    affected_qids = tuple(sorted(region.affected_qids))
    checkpoint_store = RecoveryCheckpointStore(
        data_root.cache / "wikidata_recovery" / "checkpoints",
        stem,
        recovery_plan_key(
            fingerprints=region.fingerprints,
            affected_qids=affected_qids,
            sections_hash=sha256_file(paths["sections"]),
            settings_identity=(
                tuple(settings.languages) if settings.languages is not None else None,
                settings.max_articles_per_qid,
                settings.fetch_full_text,
            ),
        ),
    )
    emit = log or (lambda _message: None)

    def build_batch(
        batch_qids: tuple[str, ...],
        progress: RecoveryProgress,
    ) -> RecoveryBatchArtifacts:
        return _build_batch_artifacts(
            batch_qids,
            existing_documents=retained_documents,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            augmentation_client=augmentation_client,
            settings=settings,
            progress=progress,
        )

    completed_batches = _execute_recovery_batches(
        stem=stem,
        affected_qids=affected_qids,
        checkpoint_store=checkpoint_store,
        build_batch=build_batch,
        emit=emit,
        scheduler_snapshot=scheduler_snapshot,
    )
    batch_documents = [row for batch in completed_batches for row in batch.documents]
    batch_sections = [row for batch in completed_batches for row in batch.sections]
    batch_facts = [row for batch in completed_batches for row in batch.facts]

    new_documents = batch_documents
    merged_documents, _ = _merge_rows(
        retained_documents,
        new_documents,
        primary_key="document_id",
        label="document_id",
        secondary_key="article_id",
    )
    affected_polygon_ids = {
        polygon_id
        for qid, polygon_ids in region.affected_polygon_ids_by_qid
        if qid in affected_qids
        for polygon_id in polygon_ids
    }
    merged_links = _merge_links(
        polygons,
        retained_links,
        merged_documents,
        affected_qids=set(affected_qids),
    )
    updated_polygons, updated_links = _recompute_affected_polygon_fields(
        polygons,
        merged_links,
        merged_documents,
        affected_qids=set(affected_qids),
    )

    new_sections = batch_sections
    merged_sections, _ = _merge_rows(
        retained_sections,
        new_sections,
        primary_key="section_id",
        label="section_id",
    )
    new_facts = batch_facts
    merged_facts, _ = _merge_rows(
        retained_facts,
        new_facts,
        primary_key="fact_id",
        label="fact_id",
    )

    merged_documents.sort(key=lambda row: str(row["document_id"]))
    merged_sections.sort(key=lambda row: (str(row["document_id"]), int(row["section_index"])))
    merged_facts.sort(key=lambda row: str(row["fact_id"]))
    _validate_existing_rows(
        updated_polygons,
        updated_links,
        merged_documents,
        merged_sections,
        merged_facts,
    )
    _validate_preservation(
        polygons,
        updated_polygons,
        documents,
        merged_documents,
        sections,
        merged_sections,
        retained_facts,
        merged_facts,
        affected_qids=set(affected_qids),
        removed_document_ids=orphan_document_ids,
        removed_section_ids={
            str(row["section_id"])
            for row in sections
            if str(row["document_id"]) in orphan_document_ids
        },
    )

    persisted_links = (
        _legacy_wikipedia_links_to_canonical(
            updated_links,
            merged_documents,
            preserved_wikivoyage_links,
        )
        if canonical_links
        else updated_links
    )
    terminal_classifications = _terminal_classifications(region, persisted_links)
    changed = any(
        before != after
        for before, after in (
            (polygons, updated_polygons),
            (stored_links, persisted_links),
            (documents, merged_documents),
            (sections, merged_sections),
            (facts, merged_facts),
        )
    )
    map_inputs_changed = any(
        before != after
        for before, after in (
            (polygons, updated_polygons),
            (stored_links, persisted_links),
            (documents, merged_documents),
        )
    )
    if not changed:
        record_region_recovery_receipt(data_root, stem, terminal_classifications)
        checkpoint_store.clear()
        return RecoveryRepairResult(
            stem,
            False,
            affected_qids,
            len(affected_polygon_ids),
            (),
            False,
        )

    directory = transaction_directory(transaction_root, stem)
    directory.mkdir(parents=True, exist_ok=False)
    staged = {
        "polygons": directory / "staged-polygons.parquet",
        "links": directory / "staged-polygon-articles.parquet",
        "documents": directory / "staged-wikipedia-documents.parquet",
        "sections": directory / "staged-wikipedia-sections.parquet",
        "facts": directory / "staged-wikidata-facts.parquet",
        "processed_manifest": directory / "staged-processed-manifest.json",
        "augmentation_manifest": directory / "staged-augmentation-manifest.json",
    }
    _write_table(staged["polygons"], updated_polygons, POLYGON_COLUMNS, polygon_schema())
    if canonical_links:
        _write_table(
            staged["links"],
            persisted_links,
            CANONICAL_COLUMNS,
            polygon_document_link_schema(),
        )
    else:
        _write_table(
            staged["links"],
            persisted_links,
            POLYGON_ARTICLE_COLUMNS,
            polygon_article_schema(),
        )
    _write_table(
        staged["documents"],
        merged_documents,
        WIKIPEDIA_DOCUMENT_COLUMNS,
        wikipedia_document_schema(),
    )
    _write_table(staged["sections"], merged_sections, SECTION_COLUMNS, section_schema())
    _write_table(staged["facts"], merged_facts, FACT_COLUMNS, fact_schema())
    _stage_manifests(
        stem,
        paths=paths,
        staged=staged,
        polygons=updated_polygons,
        documents=merged_documents,
        sections=merged_sections,
        facts=merged_facts,
        affected_qids=affected_qids,
        affected_polygon_count=len(affected_polygon_ids),
    )
    replacements = [
        (paths["polygons"], staged["polygons"]),
        (paths["links"], staged["links"]),
        (paths["documents"], staged["documents"]),
        (paths["sections"], staged["sections"]),
        (paths["facts"], staged["facts"]),
        (paths["processed_manifest"], staged["processed_manifest"]),
        (paths["augmentation_manifest"], staged["augmentation_manifest"]),
    ]
    commit_replacements(directory, stem, replacements, before_commit=before_commit)
    record_region_recovery_receipt(data_root, stem, terminal_classifications)
    post_audit = audit_wikidata_integrity(
        data_root,
        [stem],
        wikidata_client,
        batch_size=settings.enrichment_batch_size,
        languages=settings.languages,
        max_articles_per_qid=settings.max_articles_per_qid,
    )
    if post_audit.region(stem).affected_qids:
        raise RecoveryRepairError(f"Recovery did not converge for region {stem!r}")
    checkpoint_store.clear()
    repaired_paths = tuple(target for target, _ in replacements)
    return RecoveryRepairResult(
        stem,
        True,
        affected_qids,
        len(affected_polygon_ids),
        repaired_paths,
        map_inputs_changed,
    )


def _execute_recovery_batches(
    *,
    stem: str,
    affected_qids: tuple[str, ...],
    checkpoint_store: RecoveryCheckpointStore,
    build_batch: Callable[[tuple[str, ...], RecoveryProgress], RecoveryBatchArtifacts],
    emit: Callable[[str], None],
    scheduler_snapshot: Callable[[], RequestSchedulerSnapshot] | None = None,
    batch_window: int = RECOVERY_BATCH_WINDOW,
) -> list[RecoveryBatchArtifacts]:
    """Build independent recovery batches concurrently and return input order."""
    if batch_window < 1:
        raise ValueError("batch_window must be at least 1")
    batches = [
        affected_qids[start : start + RECOVERY_QID_BATCH_SIZE]
        for start in range(0, len(affected_qids), RECOVERY_QID_BATCH_SIZE)
    ]
    batch_total = len(batches)
    completed: dict[int, RecoveryBatchArtifacts] = {}
    missing: list[tuple[int, tuple[str, ...]]] = []
    for index, batch_qids in enumerate(batches):
        artifacts = checkpoint_store.load(index, batch_qids)
        if artifacts is None:
            missing.append((index, batch_qids))
            continue
        completed[index] = artifacts
        emit(
            f"Wikidata recovery {stem}: batch {index + 1}/{batch_total} "
            f"reused durable checkpoint ({len(batch_qids)} QIDs)"
        )

    def build_and_checkpoint(
        index: int,
        batch_qids: tuple[str, ...],
    ) -> tuple[int, RecoveryBatchArtifacts]:
        progress = RecoveryProgress(
            stem,
            batch_total,
            scheduler_snapshot=scheduler_snapshot,
        )
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

    if missing:
        _reset_retry_cancellation()
        try:
            with ThreadPoolExecutor(max_workers=min(batch_window, len(missing))) as executor:
                futures = [
                    executor.submit(build_and_checkpoint, index, batch_qids)
                    for index, batch_qids in missing
                ]
                try:
                    for future in as_completed(futures):
                        index, artifacts = future.result()
                        completed[index] = artifacts
                except BaseException:
                    _cancel_pending_retries()
                    for future in futures:
                        future.cancel()
                    raise
        finally:
            _reset_retry_cancellation()
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


def _resolve_entities(
    client: WikidataClient,
    qids: tuple[str, ...],
) -> dict[str, WikidataEntity]:
    if isinstance(client, BatchWikidataClient):
        values = client.get_entities(qids)
    else:
        values = [client.get_entity(qid) for qid in qids]
    if len(values) != len(qids):
        raise RecoveryRepairError("Wikidata client returned the wrong result count")
    resolved: dict[str, WikidataEntity] = {}
    for qid, entity in zip(qids, values, strict=True):
        if entity is None:
            raise RecoveryRepairError(f"Affected QID became authoritatively missing: {qid}")
        resolved[qid] = entity
    return resolved


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

    def fetch_qid(qid: str) -> list[dict[str, Any]]:
        entity = entities[qid]
        summary = LinkSummary(qid=qid, entity=entity)
        qid_documents: list[dict[str, Any]] = []
        for site, title in _eligible_sitelinks(entity, settings):
            if (qid, site, title) in existing:
                if progress is not None:
                    progress.advance()
                continue
            language = language_from_site(site)
            result = wikipedia_client.fetch_article(
                language,
                site,
                title,
                wikidata_label=entity.labels.get(language) or entity.labels.get("en", ""),
                wikidata_description=entity.descriptions.get(language)
                or entity.descriptions.get("en", ""),
                wikidata_aliases=entity.aliases.get(language) or entity.aliases.get("en", []),
                fetch_full_text=settings.fetch_full_text,
            )
            summary.statuses[site] = result.status
            if result.status not in NON_FATAL_FETCH_STATUSES and result.status != "ok":
                raise RecoveryRepairError(
                    "Incomplete Wikipedia recovery for "
                    f"{qid}:{site} ({result.status}): {result.error}"
                )
            if result.article is None or result.status == "article_not_found":
                if progress is not None:
                    progress.advance()
                continue
            summary.articles.append(result.article)
            identifier = article_id(
                qid, language, result.article.page_id, result.article.revision_id
            )
            article = article_row(identifier, qid, result.article, summary)
            document = wikipedia_document_from_article_row(article.__dict__)
            qid_documents.append(document.to_dict())
            if progress is not None:
                progress.advance(documents=1)
        return qid_documents

    if not affected_qids:
        return []
    with ThreadPoolExecutor(
        max_workers=min(RECOVERY_NETWORK_WORKERS, len(affected_qids))
    ) as executor:
        per_qid = executor.map(fetch_qid, affected_qids)
        return [document for documents in per_qid for document in documents]


def _recompute_affected_polygon_fields(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    affected_qids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents_by_article = {str(row["article_id"]): row for row in documents}
    links_by_polygon: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        links_by_polygon.setdefault(str(link["polygon_id"]), []).append(link)
    updated_polygons: list[dict[str, Any]] = []
    best_by_polygon: dict[str, str] = {}
    for original in polygons:
        row = dict(original)
        if set(qids_from_osm_tag(str(row["wikidata"]))) & affected_qids:
            polygon_links = links_by_polygon.get(str(row["polygon_id"]), [])
            article_ids = sorted({str(link["article_id"]) for link in polygon_links})
            languages = sorted({str(link["language"]) for link in polygon_links})
            best = next((language for language in PREFERRED_LANGUAGES if language in languages), "")
            if not best and languages:
                best = languages[0]
            row.update(
                {
                    "has_wikipedia": bool(article_ids),
                    "wikipedia_language_count": len(languages),
                    "wikipedia_languages": dumps(languages),
                    "wikipedia_article_count": len(article_ids),
                    "has_english_wikipedia": "en" in languages,
                    "has_french_wikipedia": "fr" in languages,
                    "text_available": any(
                        bool(str(documents_by_article[article]["full_text"]).strip())
                        for article in article_ids
                    ),
                    "best_language": best,
                }
            )
            best_by_polygon[str(row["polygon_id"])] = best
        updated_polygons.append(row)
    updated_links: list[dict[str, Any]] = []
    for original in links:
        row = dict(original)
        polygon_id = str(row["polygon_id"])
        if polygon_id in best_by_polygon:
            row["is_best_language"] = str(row["language"]) == best_by_polygon[polygon_id]
        updated_links.append(row)
    return updated_polygons, updated_links


def _sections_for_new_documents(
    documents: list[dict[str, Any]],
    new_document_ids: set[str],
    *,
    augmentation_client: AugmentationClient,
    progress: RecoveryProgress | None = None,
) -> list[dict[str, Any]]:
    selected = [row for row in documents if str(row["document_id"]) in new_document_ids]
    if progress is not None:
        progress.set_stage("Wikipedia sections", total=len(selected))

    def parse_document(document_row: dict[str, Any]) -> list[dict[str, Any]]:
        article = {column: document_row[column] for column in ARTICLE_COLUMNS}
        document = document_from_article_row(article)
        html = augmentation_client.parse_html(
            document.project,
            document.language,
            document.revision_id,
        )
        parsed = [section.to_dict() for section in parse_sections(document, html)]
        if progress is not None:
            progress.advance(sections=len(parsed))
        return parsed

    if not selected:
        return []
    with ThreadPoolExecutor(max_workers=min(RECOVERY_NETWORK_WORKERS, len(selected))) as executor:
        per_document = executor.map(parse_document, selected)
        return [section for sections in per_document for section in sections]


def _merge_rows(
    existing: list[dict[str, Any]],
    additions: Iterable[dict[str, Any]],
    *,
    primary_key: str,
    label: str,
    secondary_key: str | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    merged = [dict(row) for row in existing]
    primary: set[str] = set()
    secondary: set[str] = set()
    for row in existing:
        identifier = str(row[primary_key])
        if identifier in primary:
            raise RecoveryRepairError(f"duplicate {label} {identifier!r}")
        primary.add(identifier)
        if secondary_key is not None:
            secondary_identifier = str(row[secondary_key])
            if secondary_identifier in secondary:
                raise RecoveryRepairError(f"duplicate {secondary_key} {secondary_identifier!r}")
            secondary.add(secondary_identifier)
    added: set[str] = set()
    for row in additions:
        identifier = str(row[primary_key])
        if identifier in primary:
            continue
        if secondary_key is not None:
            secondary_identifier = str(row[secondary_key])
            if secondary_identifier in secondary:
                raise RecoveryRepairError(f"duplicate {secondary_key} {secondary_identifier!r}")
            secondary.add(secondary_identifier)
        primary.add(identifier)
        added.add(identifier)
        merged.append(dict(row))
    return merged, added


def _terminal_classifications(
    region: RegionAuditResult,
    links: list[dict[str, Any]],
) -> dict[str, RecoveryClassification]:
    linked_polygon_qids = {
        (str(link["polygon_id"]), str(link["wikidata"]))
        for link in links
        if link.get("project", "wikipedia") == "wikipedia"
    }
    polygons_by_qid = dict(region.polygon_ids_by_qid)
    terminal: dict[str, RecoveryClassification] = {}
    for qid, state in region.classifications:
        if qid not in region.affected_qids:
            terminal[qid] = state
            continue
        terminal[qid] = (
            RecoveryClassification.CURRENT
            if all((polygon_id, qid) in linked_polygon_qids for polygon_id in polygons_by_qid[qid])
            else RecoveryClassification.AUTHORITATIVE_NO_ARTICLE
        )
    return terminal


def _stage_manifests(
    stem: str,
    *,
    paths: dict[str, Path],
    staged: dict[str, Path],
    polygons: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    affected_qids: tuple[str, ...],
    affected_polygon_count: int,
) -> None:
    """Stage both manifests from the same repaired artifact snapshot."""
    _stage_processed_manifest(
        stem,
        paths=paths,
        staged=staged["processed_manifest"],
        polygons=polygons,
        documents=documents,
        affected_qids=affected_qids,
        affected_polygon_count=affected_polygon_count,
    )
    _stage_augmentation_manifest(
        stem,
        paths=paths,
        staged=staged["augmentation_manifest"],
        staged_polygons=staged["polygons"],
        staged_documents=staged["documents"],
        documents=documents,
        sections=sections,
        facts=facts,
    )


def _stage_processed_manifest(
    stem: str,
    *,
    paths: dict[str, Path],
    staged: Path,
    polygons: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    affected_qids: tuple[str, ...],
    affected_polygon_count: int,
) -> None:
    manifest = load_manifest(paths["processed_manifest"])
    manifest_key = f"{stem}.osm.pbf"
    if manifest_key not in manifest:
        raise RecoveryRepairError(f"Processed manifest is missing {manifest_key!r}")
    entry = dict(manifest[manifest_key])
    entry.update(
        _processed_manifest_statistics(
            polygons=polygons,
            documents=documents,
            affected_qids=affected_qids,
            affected_polygon_count=affected_polygon_count,
        )
    )
    manifest[manifest_key] = entry
    atomic_write_text(staged, dumps(manifest) + "\n")


def _processed_manifest_statistics(
    *,
    polygons: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    affected_qids: tuple[str, ...],
    affected_polygon_count: int,
) -> dict[str, object]:
    languages = sorted({str(row["language"]) for row in documents})
    return {
        "polygon_count": len(polygons),
        "unique_wikidata_count": len(_polygon_qids(polygons)),
        "article_count": len(documents),
        "language_count": len(languages),
        "languages": languages,
        "rows_with_wikipedia": sum(bool(row["has_wikipedia"]) for row in polygons),
        "rows_with_full_text": sum(bool(row["text_available"]) for row in polygons),
        "total_full_text_chars": sum(len(str(row["full_text"])) for row in documents),
        "wikidata_recovery": {
            "contract_version": RECOVERY_CONTRACT_VERSION,
            "affected_qids": list(affected_qids),
            "affected_polygon_count": affected_polygon_count,
        },
    }


def _polygon_qids(polygons: list[dict[str, Any]]) -> set[str]:
    return {qid for row in polygons for qid in qids_from_osm_tag(str(row["wikidata"]))}


def _stage_augmentation_manifest(
    stem: str,
    *,
    paths: dict[str, Path],
    staged: Path,
    staged_polygons: Path,
    staged_documents: Path,
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> None:
    try:
        augmentation: object = json.loads(
            paths["augmentation_manifest"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryRepairError(f"Augmentation manifest is unreadable: {error}") from error
    if not isinstance(augmentation, dict) or not isinstance(augmentation.get(stem), dict):
        raise RecoveryRepairError(f"Augmentation manifest is missing region {stem!r}")
    augmentation_mapping = cast(dict[str, object], augmentation)
    augmentation_entry = dict(cast(dict[str, object], augmentation_mapping[stem]))
    counts = augmentation_entry.get("counts")
    if not isinstance(counts, dict):
        raise RecoveryRepairError(f"Augmentation manifest counts are invalid for {stem!r}")
    updated_counts = dict(cast(dict[str, object], counts))
    updated_counts.update(
        {
            "wikipedia_documents": len(documents),
            "wikipedia_sections": len(sections),
            "wikidata_facts": len(facts),
        }
    )
    augmentation_entry.update(
        {
            "contract_version": CONTRACT_VERSION,
            "core_hashes": {
                str(paths["polygons"]): sha256_file(staged_polygons),
                str(paths["documents"]): sha256_file(staged_documents),
            },
            "counts": updated_counts,
        }
    )
    augmentation_mapping[stem] = augmentation_entry
    atomic_write_text(staged, dumps(augmentation_mapping) + "\n")


__all__ = ["RecoveryRepairError", "RecoveryRepairResult", "repair_wikidata_region"]
