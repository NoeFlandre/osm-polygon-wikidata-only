from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
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

from .audit import (
    RECOVERY_CONTRACT_VERSION,
    audit_wikidata_integrity,
    record_region_recovery_receipt,
)
from .models import RecoveryClassification, RegionAuditResult
from .transaction import (
    commit_replacements,
    recover_interrupted_transactions,
    transaction_directory,
)


class RecoveryRepairError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryRepairResult:
    stem: str
    changed: bool
    affected_qids: tuple[str, ...]
    affected_polygon_count: int
    repaired_paths: tuple[Path, ...]


def repair_wikidata_region(
    data_root: DataRoot,
    region: RegionAuditResult,
    *,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    augmentation_client: AugmentationClient,
    settings: Settings,
    before_commit: Callable[[], None] | None = None,
) -> RecoveryRepairResult:
    """Repair only the affected QID relationships in one finalized shard."""
    if region.blocked_reason:
        raise RecoveryRepairError(region.blocked_reason)
    if not region.requires_repair:
        return RecoveryRepairResult(region.stem, False, (), 0, ())

    transaction_root = data_root.cache / "wikidata_recovery" / "transactions"
    recover_interrupted_transactions(transaction_root)
    stem = region.stem
    paths = _region_paths(data_root, stem)
    polygons = _read_table(paths["polygons"], polygon_schema())
    links = _read_table(paths["links"], polygon_article_schema())
    documents = _read_table(paths["documents"], wikipedia_document_schema())
    sections = _read_table(paths["sections"], section_schema())
    facts = _read_table(paths["facts"], fact_schema())
    orphan_fact_ids = set(region.orphan_fact_ids)
    retained_facts = [row for row in facts if str(row["fact_id"]) not in orphan_fact_ids]
    if len(facts) - len(retained_facts) != len(orphan_fact_ids):
        raise RecoveryRepairError("Recovery plan contains stale or duplicate orphan fact IDs")
    _validate_existing_rows(polygons, links, documents, sections, retained_facts)

    affected_qids = tuple(sorted(region.affected_qids))
    entities = _resolve_entities(wikidata_client, affected_qids)
    new_documents = _fetch_missing_documents(
        affected_qids,
        entities=entities,
        existing_documents=documents,
        wikipedia_client=wikipedia_client,
        settings=settings,
    )
    merged_documents, new_document_ids = _merge_rows(
        documents,
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
        links,
        merged_documents,
        affected_qids=set(affected_qids),
    )
    updated_polygons, updated_links = _recompute_affected_polygon_fields(
        polygons,
        merged_links,
        merged_documents,
        affected_qids=set(affected_qids),
    )

    new_sections = _sections_for_new_documents(
        merged_documents,
        new_document_ids,
        augmentation_client=augmentation_client,
    )
    merged_sections, _ = _merge_rows(
        sections,
        new_sections,
        primary_key="section_id",
        label="section_id",
    )
    new_facts: list[dict[str, Any]] = []
    if affected_qids:
        raw_entities = augmentation_client.entities(list(affected_qids), props="sitelinks|claims")
        missing_raw = sorted(set(affected_qids) - set(raw_entities))
        if missing_raw:
            raise RecoveryRepairError(f"Augmentation Wikidata response omitted QIDs: {missing_raw}")
        new_facts = [
            fact.to_dict()
            for fact in build_wikidata_facts(
                augmentation_client,
                entities={qid: raw_entities[qid] for qid in affected_qids},
                progress=AugmentationProgress(),
            )
        ]
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
    )

    terminal_classifications = _terminal_classifications(region, updated_links)
    changed = any(
        before != after
        for before, after in (
            (polygons, updated_polygons),
            (links, updated_links),
            (documents, merged_documents),
            (sections, merged_sections),
            (facts, merged_facts),
        )
    )
    if not changed:
        record_region_recovery_receipt(data_root, stem, terminal_classifications)
        return RecoveryRepairResult(stem, False, affected_qids, len(affected_polygon_ids), ())

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
    _write_table(staged["links"], updated_links, POLYGON_ARTICLE_COLUMNS, polygon_article_schema())
    _write_table(
        staged["documents"],
        merged_documents,
        WIKIPEDIA_DOCUMENT_COLUMNS,
        wikipedia_document_schema(),
    )
    _write_table(staged["sections"], merged_sections, SECTION_COLUMNS, section_schema())
    _write_table(staged["facts"], merged_facts, FACT_COLUMNS, fact_schema())
    _stage_manifests(
        data_root,
        stem,
        paths=paths,
        staged=staged,
        polygons=updated_polygons,
        links=updated_links,
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
    repaired_paths = tuple(target for target, _ in replacements)
    return RecoveryRepairResult(
        stem,
        True,
        affected_qids,
        len(affected_polygon_ids),
        repaired_paths,
    )


def _region_paths(data_root: DataRoot, stem: str) -> dict[str, Path]:
    return {
        "polygons": data_root.processed_polygons / f"{stem}.parquet",
        "links": data_root.processed_links / f"{stem}.parquet",
        "documents": data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet",
        "sections": data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet",
        "facts": data_root.processed / "wikidata" / "facts" / f"{stem}.parquet",
        "processed_manifest": data_root.processed_manifests / "processed_pbfs.json",
        "augmentation_manifest": data_root.processed
        / "augmentation"
        / "manifests"
        / "augmentation_manifest.json",
    }


def _read_table(path: Path, schema: pa.Schema) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RecoveryRepairError(f"Recovery input is missing: {path}")
    actual: pa.Schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
    if not actual.equals(schema, check_metadata=True):
        raise RecoveryRepairError(f"Recovery input schema mismatch: {path}")
    table: pa.Table = pq.read_table(path)  # type: ignore[no-untyped-call]
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


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
    links = [
        (site, title)
        for site, title in sorted(entity.sitelinks.items())
        if allowed is None or language_from_site(site) in allowed
    ]
    if settings.max_articles_per_qid is not None:
        links = links[: max(0, settings.max_articles_per_qid)]
    return links


def _fetch_missing_documents(
    affected_qids: tuple[str, ...],
    *,
    entities: dict[str, WikidataEntity],
    existing_documents: list[dict[str, Any]],
    wikipedia_client: WikipediaClient,
    settings: Settings,
) -> list[dict[str, Any]]:
    existing = {
        (str(row["wikidata"]), str(row["site"]), str(row["title"])) for row in existing_documents
    }
    new_documents: list[dict[str, Any]] = []
    for qid in affected_qids:
        entity = entities[qid]
        summary = LinkSummary(qid=qid, entity=entity)
        for site, title in _eligible_sitelinks(entity, settings):
            if (qid, site, title) in existing:
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
                continue
            summary.articles.append(result.article)
            identifier = article_id(
                qid, language, result.article.page_id, result.article.revision_id
            )
            article = article_row(identifier, qid, result.article, summary)
            document = wikipedia_document_from_article_row(article.__dict__)
            new_documents.append(document.to_dict())
    return new_documents


def _merge_links(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    affected_qids: set[str],
) -> list[dict[str, Any]]:
    existing_identities: set[tuple[str, str]] = set()
    merged = [dict(row) for row in links]
    for row in links:
        identity = (str(row["polygon_id"]), str(row["article_id"]))
        if identity in existing_identities:
            raise RecoveryRepairError(f"duplicate polygon-article identity {identity!r}")
        existing_identities.add(identity)
    documents_by_qid: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        documents_by_qid.setdefault(str(document["wikidata"]), []).append(document)
    for values in documents_by_qid.values():
        values.sort(key=lambda row: str(row["document_id"]))
    for polygon in polygons:
        for qid in qids_from_osm_tag(str(polygon["wikidata"])):
            if qid not in affected_qids:
                continue
            for document in documents_by_qid.get(qid, []):
                identity = (str(polygon["polygon_id"]), str(document["article_id"]))
                if identity in existing_identities:
                    continue
                merged.append(
                    {
                        "polygon_id": polygon["polygon_id"],
                        "article_id": document["article_id"],
                        "wikidata": qid,
                        "language": document["language"],
                        "source_pbf": polygon["source_pbf"],
                        "region": polygon["region"],
                        "osm_type": polygon["osm_type"],
                        "osm_id": polygon["osm_id"],
                        "page_id": document["page_id"],
                        "revision_id": document["revision_id"],
                        "is_best_language": False,
                    }
                )
                existing_identities.add(identity)
    return merged


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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_row in documents:
        if str(document_row["document_id"]) not in new_document_ids:
            continue
        article = {column: document_row[column] for column in ARTICLE_COLUMNS}
        document = document_from_article_row(article)
        html = augmentation_client.parse_html(
            document.project,
            document.language,
            document.revision_id,
        )
        rows.extend(section.to_dict() for section in parse_sections(document, html))
    return rows


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


def _validate_existing_rows(
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> None:
    polygon_tags = _unique_mapping(polygons, "polygon_id", "wikidata", "polygon_id")
    polygon_qids = {
        polygon_id: qids_from_osm_tag(raw_tag) for polygon_id, raw_tag in polygon_tags.items()
    }
    invalid = next(
        (raw_tag for raw_tag in polygon_tags.values() if not qids_from_osm_tag(raw_tag)), None
    )
    if invalid is not None:
        raise RecoveryRepairError(f"polygon contains invalid Wikidata identifier {invalid!r}")
    documents_by_article = _unique_rows(documents, "article_id", "article_id")
    _unique_rows(documents, "document_id", "document_id")
    document_ids = {str(row["document_id"]) for row in documents}
    _unique_rows(sections, "section_id", "section_id")
    _unique_rows(facts, "fact_id", "fact_id")
    link_ids: set[tuple[str, str]] = set()
    for link in links:
        polygon_id = str(link["polygon_id"])
        article_identifier = str(link["article_id"])
        identity = (polygon_id, article_identifier)
        if identity in link_ids:
            raise RecoveryRepairError(f"duplicate polygon-article identity {identity!r}")
        link_ids.add(identity)
        if polygon_id not in polygon_qids:
            raise RecoveryRepairError(f"link references missing polygon {polygon_id!r}")
        document = documents_by_article.get(article_identifier)
        if document is None:
            raise RecoveryRepairError(f"link references missing document {article_identifier!r}")
        qid = str(link["wikidata"])
        if qid not in polygon_qids[polygon_id] or str(document["wikidata"]) != qid:
            raise RecoveryRepairError(f"link QID mismatch for {identity!r}")
    for section in sections:
        if str(section["document_id"]) not in document_ids:
            raise RecoveryRepairError(
                f"section references missing document {section['document_id']!r}"
            )
    valid_qids = {qid for qids in polygon_qids.values() for qid in qids}
    for fact in facts:
        if str(fact["wikidata"]) not in valid_qids:
            raise RecoveryRepairError(f"fact references absent QID {fact['wikidata']!r}")


def _unique_mapping(
    rows: list[dict[str, Any]],
    key: str,
    value: str,
    label: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        identifier = str(row[key])
        if identifier in result:
            raise RecoveryRepairError(f"duplicate {label} {identifier!r}")
        result[identifier] = str(row[value])
    return result


def _unique_rows(
    rows: list[dict[str, Any]],
    key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row[key])
        if identifier in result:
            raise RecoveryRepairError(f"duplicate {label} {identifier!r}")
        result[identifier] = row
    return result


def _validate_preservation(
    original_polygons: list[dict[str, Any]],
    updated_polygons: list[dict[str, Any]],
    original_documents: list[dict[str, Any]],
    updated_documents: list[dict[str, Any]],
    original_sections: list[dict[str, Any]],
    updated_sections: list[dict[str, Any]],
    original_facts: list[dict[str, Any]],
    updated_facts: list[dict[str, Any]],
    *,
    affected_qids: set[str],
) -> None:
    updated_polygon_map = {str(row["polygon_id"]): row for row in updated_polygons}
    for row in original_polygons:
        if (
            not (set(qids_from_osm_tag(str(row["wikidata"]))) & affected_qids)
            and updated_polygon_map[str(row["polygon_id"])] != row
        ):
            raise RecoveryRepairError(f"healthy polygon changed: {row['polygon_id']}")
    for original, updated, key, label in (
        (original_documents, updated_documents, "document_id", "document"),
        (original_sections, updated_sections, "section_id", "section"),
        (original_facts, updated_facts, "fact_id", "fact"),
    ):
        updated_map = {str(row[key]): row for row in updated}
        for row in original:
            if updated_map.get(str(row[key])) != row:
                raise RecoveryRepairError(f"existing {label} changed: {row[key]}")


def _write_table(
    path: Path,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    schema: pa.Schema,
) -> None:
    normalized = [{column: row.get(column) for column in columns} for row in rows]
    table = pa.Table.from_pylist(normalized, schema=schema)
    pq.write_table(table, path, compression="snappy")  # type: ignore[no-untyped-call]


def _terminal_classifications(
    region: RegionAuditResult,
    links: list[dict[str, Any]],
) -> dict[str, RecoveryClassification]:
    linked_polygon_qids = {(str(link["polygon_id"]), str(link["wikidata"])) for link in links}
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
    data_root: DataRoot,
    stem: str,
    *,
    paths: dict[str, Path],
    staged: dict[str, Path],
    polygons: list[dict[str, Any]],
    links: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    affected_qids: tuple[str, ...],
    affected_polygon_count: int,
) -> None:
    manifest = load_manifest(paths["processed_manifest"])
    manifest_key = f"{stem}.osm.pbf"
    if manifest_key not in manifest:
        raise RecoveryRepairError(f"Processed manifest is missing {manifest_key!r}")
    entry = dict(manifest[manifest_key])
    languages = sorted({str(row["language"]) for row in documents})
    entry.update(
        {
            "polygon_count": len(polygons),
            "unique_wikidata_count": len(
                {qid for row in polygons for qid in qids_from_osm_tag(str(row["wikidata"]))}
            ),
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
    )
    manifest[manifest_key] = entry
    atomic_write_text(staged["processed_manifest"], dumps(manifest) + "\n")

    try:
        augmentation: object = json.loads(
            paths["augmentation_manifest"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryRepairError(f"Augmentation manifest is unreadable: {error}") from error
    if not isinstance(augmentation, dict) or not isinstance(augmentation.get(stem), dict):
        raise RecoveryRepairError(f"Augmentation manifest is missing region {stem!r}")
    augmentation_entry = dict(augmentation[stem])
    counts = augmentation_entry.get("counts")
    if not isinstance(counts, dict):
        raise RecoveryRepairError(f"Augmentation manifest counts are invalid for {stem!r}")
    updated_counts = dict(counts)
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
                str(paths["polygons"]): sha256_file(staged["polygons"]),
                str(paths["documents"]): sha256_file(staged["documents"]),
            },
            "counts": updated_counts,
        }
    )
    augmentation[stem] = augmentation_entry
    atomic_write_text(staged["augmentation_manifest"], dumps(augmentation) + "\n")


__all__ = ["RecoveryRepairError", "RecoveryRepairResult", "repair_wikidata_region"]
