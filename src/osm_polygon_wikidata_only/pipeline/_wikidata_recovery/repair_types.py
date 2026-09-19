"""Internal value objects shared by the Wikidata recovery stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import RecoveryClassification


@dataclass(frozen=True, slots=True)
class _RepairInputs:
    """Validated source tables and rows retained for a regional repair."""

    paths: dict[str, Path]
    polygons: list[dict[str, Any]]
    stored_links: list[dict[str, Any]]
    links: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    preserved_wikivoyage_links: list[dict[str, Any]]
    canonical_links: bool
    retained_facts: list[dict[str, Any]]
    retained_documents: list[dict[str, Any]]
    retained_sections: list[dict[str, Any]]
    retained_links: list[dict[str, Any]]
    orphan_document_ids: set[str]


@dataclass(frozen=True, slots=True)
class _RepairOutputs:
    """Merged and validated tables ready for transactional persistence."""

    updated_polygons: list[dict[str, Any]]
    persisted_links: list[dict[str, Any]]
    merged_documents: list[dict[str, Any]]
    merged_sections: list[dict[str, Any]]
    merged_facts: list[dict[str, Any]]
    terminal_classifications: dict[str, RecoveryClassification]
    affected_qids: tuple[str, ...]
    affected_polygon_ids: set[str]
    map_inputs_changed: bool
    changed: bool


RepairInputs = _RepairInputs
RepairOutputs = _RepairOutputs


__all__ = ["RepairInputs", "RepairOutputs"]
