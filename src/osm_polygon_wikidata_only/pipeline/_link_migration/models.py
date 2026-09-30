"""Immutable planning models for polygon-document link migration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

    from osm_polygon_wikidata_only.augmentation.rejection_ledger import IntegrityPlan
    from osm_polygon_wikidata_only.config.paths import DataRoot


class StemClassification(StrEnum):
    MIGRATABLE = "migratable"
    CANONICAL = "canonical"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class StemPlan:
    """Per-stem migration plan entry."""

    stem: str
    classification: StemClassification
    reason: str
    polygons_fingerprint: str
    links_fingerprint: str
    documents_fingerprint: str
    row_count: int
    canonical_digest: str | None


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """Immutable read-only migration plan."""

    processed_dir: Path
    stems: tuple[StemPlan, ...]

    @property
    def is_safe_to_apply(self) -> bool:
        return all(stem.classification != StemClassification.BLOCKED for stem in self.stems)


@dataclass(frozen=True, slots=True)
class StemApplyInputs:
    """Immutable source tables and paths used by one stem transaction."""

    stem_plan: StemPlan
    links_path: Path
    polygons_path: Path
    docs_path: Path
    voyage_documents_path: Path
    voyage_sections_path: Path
    legacy_table: pa.Table
    polygons_table: pa.Table
    docs_table: pa.Table
    data_root: DataRoot


@dataclass(frozen=True, slots=True)
class StemApplyContext:
    """Derived canonical data and integrity plan for one stem."""

    inputs: StemApplyInputs
    integrity_plan: IntegrityPlan | None
    canonical_table: pa.Table


__all__ = [
    "MigrationPlan",
    "StemApplyContext",
    "StemApplyInputs",
    "StemClassification",
    "StemPlan",
]
