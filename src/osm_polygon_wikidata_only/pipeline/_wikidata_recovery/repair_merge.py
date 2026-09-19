"""Merge recovered Wikidata rows and recompute polygon relationships."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .checkpoints import RecoveryBatchArtifacts
from .link_rows import (
    legacy_wikipedia_links_to_canonical as _legacy_wikipedia_links_to_canonical,
)
from .link_rows import (
    merge_links as _merge_links,
)
from .models import RecoveryClassification, RecoveryRepairError, RegionAuditResult
from .repair_fields import recompute_affected_polygon_fields as _recompute_affected_polygon_fields
from .repair_types import RepairInputs as _RepairInputs
from .repair_types import RepairOutputs as _RepairOutputs
from .validation import validate_existing_rows as _validate_existing_rows
from .validation import validate_preservation as _validate_preservation


def _flatten_recovery_batches(
    completed_batches: list[RecoveryBatchArtifacts],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten checkpoint batches without changing their deterministic order."""
    return (
        _flatten_batch_rows(completed_batches, "documents"),
        _flatten_batch_rows(completed_batches, "sections"),
        _flatten_batch_rows(completed_batches, "facts"),
    )


def _flatten_batch_rows(
    completed_batches: list[RecoveryBatchArtifacts], attribute: str
) -> list[dict[str, Any]]:
    """Flatten one named row collection from recovery batches."""
    return [row for batch in completed_batches for row in getattr(batch, attribute)]


def _affected_polygon_ids(
    region: RegionAuditResult,
    affected_qids: tuple[str, ...],
) -> set[str]:
    """Collect polygon IDs whose QID relationships are being repaired."""
    return {
        polygon_id
        for qid, polygon_ids in region.affected_polygon_ids_by_qid
        if qid in affected_qids
        for polygon_id in polygon_ids
    }


def _sort_repair_tables(
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> None:
    """Apply the stable row ordering used by repaired artifacts."""
    documents.sort(key=lambda row: str(row["document_id"]))
    sections.sort(key=lambda row: (str(row["document_id"]), int(row["section_index"])))
    facts.sort(key=lambda row: str(row["fact_id"]))


def _removed_section_ids(sections: list[dict[str, Any]], orphan_document_ids: set[str]) -> set[str]:
    """Return section IDs cascaded from orphaned documents."""
    return {
        str(row["section_id"]) for row in sections if str(row["document_id"]) in orphan_document_ids
    }


def _merge_repair_tables(
    region: RegionAuditResult,
    inputs: _RepairInputs,
    completed_batches: list[RecoveryBatchArtifacts],
    affected_qids: tuple[str, ...],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    set[str],
]:
    """Merge documents, links, sections, and facts from completed batches."""
    batch_documents, batch_sections, batch_facts = _flatten_recovery_batches(completed_batches)
    merged_documents, _ = _merge_rows(
        inputs.retained_documents,
        batch_documents,
        primary_key="document_id",
        label="document_id",
        secondary_key="article_id",
    )
    merged_links = _merge_links(
        inputs.polygons,
        inputs.retained_links,
        merged_documents,
        affected_qids=set(affected_qids),
    )
    updated_polygons, updated_links = _recompute_affected_polygon_fields(
        inputs.polygons,
        merged_links,
        merged_documents,
        affected_qids=set(affected_qids),
    )
    merged_sections, _ = _merge_rows(
        inputs.retained_sections,
        batch_sections,
        primary_key="section_id",
        label="section_id",
    )
    merged_facts, _ = _merge_rows(
        inputs.retained_facts,
        batch_facts,
        primary_key="fact_id",
        label="fact_id",
    )
    _sort_repair_tables(merged_documents, merged_sections, merged_facts)
    return (
        updated_polygons,
        updated_links,
        merged_documents,
        merged_sections,
        merged_facts,
        _affected_polygon_ids(region, affected_qids),
    )


def _validate_merged_repair(
    inputs: _RepairInputs,
    merged: tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        set[str],
    ],
    affected_qids: tuple[str, ...],
) -> None:
    """Validate merged rows and preservation invariants before publication."""
    updated_polygons, updated_links, merged_documents, merged_sections, merged_facts, _ = merged
    _validate_existing_rows(
        updated_polygons,
        updated_links,
        merged_documents,
        merged_sections,
        merged_facts,
    )
    _validate_preservation(
        inputs.polygons,
        updated_polygons,
        inputs.documents,
        merged_documents,
        inputs.sections,
        merged_sections,
        inputs.retained_facts,
        merged_facts,
        affected_qids=set(affected_qids),
        removed_document_ids=inputs.orphan_document_ids,
        removed_section_ids=_removed_section_ids(inputs.sections, inputs.orphan_document_ids),
    )


def _persisted_repair_links(
    updated_links: list[dict[str, Any]],
    merged_documents: list[dict[str, Any]],
    inputs: _RepairInputs,
) -> list[dict[str, Any]]:
    """Restore canonical link rows when the input artifact uses that schema."""
    if not inputs.canonical_links:
        return updated_links
    return _legacy_wikipedia_links_to_canonical(
        updated_links,
        merged_documents,
        inputs.preserved_wikivoyage_links,
    )


def _repair_change_flags(
    inputs: _RepairInputs,
    updated_polygons: list[dict[str, Any]],
    persisted_links: list[dict[str, Any]],
    merged_documents: list[dict[str, Any]],
    merged_sections: list[dict[str, Any]],
    merged_facts: list[dict[str, Any]],
) -> tuple[bool, bool]:
    """Return total-change and map-input-change flags for repaired tables."""
    changed = any(
        before != after
        for before, after in (
            (inputs.polygons, updated_polygons),
            (inputs.stored_links, persisted_links),
            (inputs.documents, merged_documents),
            (inputs.sections, merged_sections),
            (inputs.facts, merged_facts),
        )
    )
    map_inputs_changed = any(
        before != after
        for before, after in (
            (inputs.polygons, updated_polygons),
            (inputs.stored_links, persisted_links),
            (inputs.documents, merged_documents),
        )
    )
    return changed, map_inputs_changed


def _merge_repair_outputs(
    region: RegionAuditResult,
    inputs: _RepairInputs,
    completed_batches: list[RecoveryBatchArtifacts],
) -> _RepairOutputs:
    """Merge recovered rows, recompute affected fields, and validate preservation."""
    affected_qids = tuple(sorted(region.affected_qids))
    merged = _merge_repair_tables(region, inputs, completed_batches, affected_qids)
    _validate_merged_repair(inputs, merged, affected_qids)
    (
        updated_polygons,
        updated_links,
        merged_documents,
        merged_sections,
        merged_facts,
        affected_polygon_ids,
    ) = merged
    persisted_links = _persisted_repair_links(updated_links, merged_documents, inputs)
    changed, map_inputs_changed = _repair_change_flags(
        inputs,
        updated_polygons,
        persisted_links,
        merged_documents,
        merged_sections,
        merged_facts,
    )
    terminal_classifications = _terminal_classifications(region, persisted_links)
    return _RepairOutputs(
        updated_polygons=updated_polygons,
        persisted_links=persisted_links,
        merged_documents=merged_documents,
        merged_sections=merged_sections,
        merged_facts=merged_facts,
        terminal_classifications=terminal_classifications,
        affected_qids=affected_qids,
        affected_polygon_ids=affected_polygon_ids,
        map_inputs_changed=map_inputs_changed,
        changed=changed,
    )


def _validate_merge_rows(
    existing: list[dict[str, Any]],
    *,
    primary_key: str,
    label: str,
    secondary_key: str | None,
) -> tuple[set[str], set[str]]:
    """Validate existing primary and optional secondary identities."""
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
    return primary, secondary


def _append_merge_rows(
    merged: list[dict[str, Any]],
    additions: Iterable[dict[str, Any]],
    *,
    primary: set[str],
    secondary: set[str],
    primary_key: str,
    label: str,
    secondary_key: str | None,
) -> set[str]:
    """Append unseen rows while enforcing merge identity constraints."""
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
    return added


def _merge_rows(
    existing: list[dict[str, Any]],
    additions: Iterable[dict[str, Any]],
    *,
    primary_key: str,
    label: str,
    secondary_key: str | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    merged = [dict(row) for row in existing]
    primary, secondary = _validate_merge_rows(
        existing,
        primary_key=primary_key,
        label=label,
        secondary_key=secondary_key,
    )
    added = _append_merge_rows(
        merged,
        additions,
        primary=primary,
        secondary=secondary,
        primary_key=primary_key,
        label=label,
        secondary_key=secondary_key,
    )
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
        terminal[qid] = _terminal_classification(
            qid,
            state,
            affected_qids=set(region.affected_qids),
            polygon_ids=polygons_by_qid.get(qid, ()),
            linked_polygon_qids=linked_polygon_qids,
        )
    return terminal


def _terminal_classification(
    qid: str,
    state: RecoveryClassification,
    *,
    affected_qids: set[str],
    polygon_ids: tuple[str, ...] | list[str],
    linked_polygon_qids: set[tuple[str, str]],
) -> RecoveryClassification:
    """Classify one repaired QID after checking all affected polygon links."""
    if qid not in affected_qids:
        return state
    if all((polygon_id, qid) in linked_polygon_qids for polygon_id in polygon_ids):
        return RecoveryClassification.CURRENT
    return RecoveryClassification.AUTHORITATIVE_NO_ARTICLE


# Public collaborator spellings used by the recovery facade.
flatten_recovery_batches = _flatten_recovery_batches
flatten_batch_rows = _flatten_batch_rows
affected_polygon_ids = _affected_polygon_ids
sort_repair_tables = _sort_repair_tables
removed_section_ids = _removed_section_ids
merge_repair_tables = _merge_repair_tables
validate_merged_repair = _validate_merged_repair
persisted_repair_links = _persisted_repair_links
repair_change_flags = _repair_change_flags
merge_repair_outputs = _merge_repair_outputs
validate_merge_rows = _validate_merge_rows
append_merge_rows = _append_merge_rows
merge_rows = _merge_rows
terminal_classifications = _terminal_classifications
terminal_classification = _terminal_classification
