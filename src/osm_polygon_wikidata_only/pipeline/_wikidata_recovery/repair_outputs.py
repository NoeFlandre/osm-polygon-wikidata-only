"""Stage and transactionally publish the outputs of a Wikidata repair."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from osm_polygon_wikidata_only.augmentation.schema import (
    FACT_COLUMNS,
    SECTION_COLUMNS,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.steps import CONTRACT_VERSION, sha256_file
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    WIKIPEDIA_DOCUMENT_COLUMNS,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    CANONICAL_COLUMNS,
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_COLUMNS,
    polygon_article_schema,
    polygon_schema,
)
from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataClient
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.manifest import load_manifest

from .audit import (
    RECOVERY_CONTRACT_VERSION,
    audit_wikidata_integrity,
    record_region_recovery_receipt,
)
from .checkpoints import RecoveryCheckpointStore
from .models import RecoveryRepairError, RecoveryRepairResult, RegionAuditResult
from .repair_types import _RepairInputs, _RepairOutputs
from .storage import write_table as _write_table
from .transaction import commit_replacements, transaction_directory


def _staged_repair_paths(directory: Path) -> dict[str, Path]:
    """Return deterministic temporary paths for repaired regional artifacts."""
    return {
        "polygons": directory / "staged-polygons.parquet",
        "links": directory / "staged-polygon-articles.parquet",
        "documents": directory / "staged-wikipedia-documents.parquet",
        "sections": directory / "staged-wikipedia-sections.parquet",
        "facts": directory / "staged-wikidata-facts.parquet",
        "processed_manifest": directory / "staged-processed-manifest.json",
        "augmentation_manifest": directory / "staged-augmentation-manifest.json",
    }


def _stage_repair_tables(
    stem: str,
    inputs: _RepairInputs,
    outputs: _RepairOutputs,
    *,
    paths: dict[str, Path],
    directory: Path,
) -> dict[str, Path]:
    """Write all repaired tables and manifests into a transaction directory."""
    staged = _staged_repair_paths(directory)
    _write_table(staged["polygons"], outputs.updated_polygons, POLYGON_COLUMNS, polygon_schema())
    if inputs.canonical_links:
        _write_table(
            staged["links"],
            outputs.persisted_links,
            CANONICAL_COLUMNS,
            polygon_document_link_schema(),
        )
    else:
        _write_table(
            staged["links"],
            outputs.persisted_links,
            POLYGON_ARTICLE_COLUMNS,
            polygon_article_schema(),
        )
    _write_table(
        staged["documents"],
        outputs.merged_documents,
        WIKIPEDIA_DOCUMENT_COLUMNS,
        wikipedia_document_schema(),
    )
    _write_table(staged["sections"], outputs.merged_sections, SECTION_COLUMNS, section_schema())
    _write_table(staged["facts"], outputs.merged_facts, FACT_COLUMNS, fact_schema())
    _stage_manifests(
        stem,
        paths=paths,
        staged=staged,
        polygons=outputs.updated_polygons,
        documents=outputs.merged_documents,
        sections=outputs.merged_sections,
        facts=outputs.merged_facts,
        affected_qids=outputs.affected_qids,
        affected_polygon_count=len(outputs.affected_polygon_ids),
    )
    return staged


def persist_repair_outputs(
    data_root: DataRoot,
    region: RegionAuditResult,
    inputs: _RepairInputs,
    outputs: _RepairOutputs,
    checkpoint_store: RecoveryCheckpointStore,
    *,
    transaction_root: Path,
    wikidata_client: WikidataClient,
    settings: Settings,
    before_commit: Callable[[], None] | None,
    audit_fn: Callable[..., Any] = audit_wikidata_integrity,
    record_receipt_fn: Callable[..., Any] = record_region_recovery_receipt,
) -> RecoveryRepairResult:
    """Persist changed repair outputs transactionally and verify convergence."""
    if not outputs.changed:
        record_receipt_fn(data_root, region.stem, outputs.terminal_classifications)
        checkpoint_store.clear()
        return RecoveryRepairResult(
            region.stem,
            False,
            outputs.affected_qids,
            len(outputs.affected_polygon_ids),
            (),
            False,
        )
    directory = transaction_directory(transaction_root, region.stem)
    directory.mkdir(parents=True, exist_ok=False)
    staged = _stage_repair_tables(
        region.stem,
        inputs,
        outputs,
        paths=inputs.paths,
        directory=directory,
    )
    replacements = [(inputs.paths[key], staged[key]) for key in staged]
    commit_replacements(directory, region.stem, replacements, before_commit=before_commit)
    record_receipt_fn(data_root, region.stem, outputs.terminal_classifications)
    post_audit = audit_fn(
        data_root,
        [region.stem],
        wikidata_client,
        batch_size=settings.enrichment_batch_size,
        languages=settings.languages,
        max_articles_per_qid=settings.max_articles_per_qid,
    )
    if post_audit.region(region.stem).affected_qids:
        raise RecoveryRepairError(f"Recovery did not converge for region {region.stem!r}")
    checkpoint_store.clear()
    repaired_paths = tuple(target for target, _ in replacements)
    return RecoveryRepairResult(
        region.stem,
        True,
        outputs.affected_qids,
        len(outputs.affected_polygon_ids),
        repaired_paths,
        outputs.map_inputs_changed,
    )


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
    atomic_write_text(staged, _json_dumps(manifest) + "\n")


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
    from osm_polygon_wikidata_only.enrichment.wikidata.parsing import qids_from_osm_tag

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
    atomic_write_text(staged, _json_dumps(augmentation_mapping) + "\n")


def _json_dumps(value: object) -> str:
    """Use the project's deterministic JSON encoder without importing its facade."""
    from osm_polygon_wikidata_only.utils.json import dumps

    return dumps(value)


__all__ = [
    "_processed_manifest_statistics",
    "_stage_augmentation_manifest",
    "_stage_manifests",
    "_stage_processed_manifest",
    "_stage_repair_tables",
    "_staged_repair_paths",
    "persist_repair_outputs",
]
