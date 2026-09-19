"""Shared immutable values for the Wikidata integrity audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RegionScan:
    stem: str
    fingerprints: tuple[tuple[str, str], ...]
    polygon_ids_by_qid: tuple[tuple[str, tuple[str, ...]], ...]
    missing_polygon_ids_by_qid: tuple[tuple[str, tuple[str, ...]], ...]
    orphan_fact_ids: tuple[str, ...] = ()
    orphan_document_ids: tuple[str, ...] = ()
    blocked_reason: str = ""


@dataclass(frozen=True, slots=True)
class RegionRows:
    paths: dict[str, Path]
    canonical_links: bool
    polygon_rows: list[dict[str, Any]]
    link_rows: list[dict[str, Any]]
    document_rows: list[dict[str, Any]]
    fact_rows: list[dict[str, Any]]


class ScanError(ValueError):
    """Raised when a region cannot satisfy the recovery input contract."""


__all__ = ["RegionRows", "RegionScan", "ScanError"]
