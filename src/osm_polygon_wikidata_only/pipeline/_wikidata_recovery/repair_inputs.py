"""Load and validate the source artifacts for one Wikidata repair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import fact_schema, section_schema
from osm_polygon_wikidata_only.augmentation.steps import sha256_file
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.polygon_document_links import polygon_document_link_schema
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema, polygon_schema
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.checkpoints import (
    RecoveryCheckpointStore,
    recovery_plan_key,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.link_rows import (
    canonical_wikipedia_links_to_legacy as _canonical_wikipedia_links_to_legacy,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.models import (
    RecoveryRepairError,
    RegionAuditResult,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.storage import (
    read_table as _read_table,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.storage import (
    region_paths as _region_paths,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.validation import (
    validate_existing_rows as _validate_existing_rows,
)

from .repair_types import _RepairInputs


def _load_repair_links(
    paths: dict[str, Path],
    *,
    polygons: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    affected_qids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Load either canonical or legacy links and preserve Wikivoyage rows."""
    links_schema = pq.read_schema(paths["links"])
    canonical_links = links_schema.equals(polygon_document_link_schema(), check_metadata=True)
    if canonical_links:
        stored_links = _read_table(paths["links"], polygon_document_link_schema())
        preserved = [dict(row) for row in stored_links if row["project"] == "wikivoyage"]
        links = _canonical_wikipedia_links_to_legacy(
            stored_links,
            documents,
            polygons,
            affected_qids=affected_qids,
        )
        return stored_links, links, preserved, True
    if links_schema.equals(polygon_article_schema(), check_metadata=True):
        stored_links = _read_table(paths["links"], polygon_article_schema())
        return stored_links, stored_links, [], False
    raise RecoveryRepairError(f"Recovery input schema mismatch: {paths['links']}")


def _retain_repair_rows(
    region: RegionAuditResult,
    *,
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str]
]:
    """Drop planned orphan rows and validate that the plan matches its inputs."""
    orphan_fact_ids = set(region.orphan_fact_ids)
    orphan_document_ids = set(region.orphan_document_ids)
    retained_facts = _retain_facts(facts, orphan_fact_ids)
    if len(facts) - len(retained_facts) != len(orphan_fact_ids):
        raise RecoveryRepairError("Recovery plan contains stale or duplicate orphan fact IDs")
    orphan_article_ids = _orphan_article_ids(documents, orphan_document_ids)
    retained_documents = _retain_documents(documents, orphan_document_ids)
    if len(documents) - len(retained_documents) != len(orphan_document_ids):
        raise RecoveryRepairError("Recovery plan contains stale or duplicate orphan document IDs")
    retained_sections = _retain_sections(sections, orphan_document_ids)
    retained_links = _retain_links(links, orphan_article_ids)
    return (
        retained_facts,
        retained_documents,
        retained_sections,
        retained_links,
        orphan_document_ids,
    )


def _retain_facts(facts: list[dict[str, Any]], orphan_fact_ids: set[str]) -> list[dict[str, Any]]:
    """Keep facts not listed as planned orphans."""
    return [row for row in facts if str(row["fact_id"]) not in orphan_fact_ids]


def _orphan_article_ids(documents: list[dict[str, Any]], orphan_document_ids: set[str]) -> set[str]:
    """Find article identities belonging to orphan documents."""
    return {
        str(row["article_id"])
        for row in documents
        if str(row["document_id"]) in orphan_document_ids
    }


def _retain_documents(
    documents: list[dict[str, Any]], orphan_document_ids: set[str]
) -> list[dict[str, Any]]:
    """Keep documents not listed as planned orphans."""
    return [row for row in documents if str(row["document_id"]) not in orphan_document_ids]


def _retain_sections(
    sections: list[dict[str, Any]], orphan_document_ids: set[str]
) -> list[dict[str, Any]]:
    """Keep sections belonging to retained documents."""
    return [row for row in sections if str(row["document_id"]) not in orphan_document_ids]


def _retain_links(
    links: list[dict[str, Any]], orphan_article_ids: set[str]
) -> list[dict[str, Any]]:
    """Keep links belonging to retained articles."""
    return [row for row in links if str(row["article_id"]) not in orphan_article_ids]


def _load_repair_inputs(data_root: DataRoot, region: RegionAuditResult) -> _RepairInputs:
    """Load, normalize, and validate all regional repair inputs."""
    paths = _region_paths(data_root, region.stem)
    polygons = _read_table(paths["polygons"], polygon_schema())
    documents = _read_table(paths["documents"], wikipedia_document_schema())
    sections = _read_table(paths["sections"], section_schema())
    facts = _read_table(paths["facts"], fact_schema())
    stored_links, links, preserved, canonical_links = _load_repair_links(
        paths,
        polygons=polygons,
        documents=documents,
        affected_qids=set(region.affected_qids),
    )
    retained_facts, retained_documents, retained_sections, retained_links, orphan_document_ids = (
        _retain_repair_rows(
            region,
            documents=documents,
            sections=sections,
            facts=facts,
            links=links,
        )
    )
    _validate_existing_rows(
        polygons,
        retained_links,
        retained_documents,
        retained_sections,
        retained_facts,
    )
    return _RepairInputs(
        paths=paths,
        polygons=polygons,
        stored_links=stored_links,
        links=links,
        documents=documents,
        sections=sections,
        facts=facts,
        preserved_wikivoyage_links=preserved,
        canonical_links=canonical_links,
        retained_facts=retained_facts,
        retained_documents=retained_documents,
        retained_sections=retained_sections,
        retained_links=retained_links,
        orphan_document_ids=orphan_document_ids,
    )


def _repair_checkpoint_store(
    data_root: DataRoot,
    region: RegionAuditResult,
    inputs: _RepairInputs,
    settings: Settings,
    affected_qids: tuple[str, ...],
) -> RecoveryCheckpointStore:
    """Create the checkpoint store keyed by all repair inputs and settings."""
    return RecoveryCheckpointStore(
        data_root.cache / "wikidata_recovery" / "checkpoints",
        region.stem,
        recovery_plan_key(
            fingerprints=region.fingerprints,
            affected_qids=affected_qids,
            sections_hash=sha256_file(inputs.paths["sections"]),
            settings_identity=(
                tuple(settings.languages) if settings.languages is not None else None,
                settings.max_articles_per_qid,
                settings.fetch_full_text,
            ),
        ),
    )
