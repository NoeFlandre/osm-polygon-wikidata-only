"""Immutable planning models for polygon-document link migration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


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


__all__ = ["MigrationPlan", "StemClassification", "StemPlan"]
