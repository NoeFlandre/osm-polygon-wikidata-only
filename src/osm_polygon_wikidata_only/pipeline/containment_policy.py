"""Canonical policy for known whole-file Geofabrik containment overlaps."""

from __future__ import annotations

import re
from dataclasses import dataclass

_STEM_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*-latest\Z")


@dataclass(frozen=True, slots=True)
class ContainmentRule:
    """Retain ``parent`` and retire ``children`` after lossless consolidation."""

    parent: str
    children: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TableContract:
    """Canonical per-region table and its source-independent identity."""

    subdir: str
    identity_columns: tuple[str, ...]


CONTAINMENT_RULES = (
    ContainmentRule("brandenburg-latest", ("berlin-latest",)),
    ContainmentRule(
        "china-guangdong-latest",
        ("china-hong-kong-latest", "china-macau-latest"),
    ),
    ContainmentRule(
        "china-hebei-latest",
        ("china-beijing-latest", "china-tianjin-latest"),
    ),
    ContainmentRule("indonesia-nusa-tenggara-latest", ("east-timor-latest",)),
    ContainmentRule("italy-latest", ("centro-latest", "nord-est-latest")),
    ContainmentRule("morocco-latest", ("ceuta-latest", "melilla-latest")),
    ContainmentRule("niedersachsen-latest", ("bremen-latest",)),
)

TABLE_CONTRACTS = (
    TableContract("polygons", ("osm_type", "osm_id")),
    TableContract("polygon_articles", ("osm_type", "osm_id", "article_id")),
    TableContract("wikipedia/documents", ("document_id",)),
    TableContract("wikipedia/sections", ("section_id",)),
    TableContract("wikivoyage/documents", ("document_id",)),
    TableContract("wikivoyage/sections", ("section_id",)),
    TableContract("wikidata/facts", ("fact_id",)),
)


def validate_stem(stem: str) -> str:
    """Return a safe canonical stem or raise before path construction."""
    if not _STEM_PATTERN.fullmatch(stem):
        raise ValueError(f"Invalid containment stem: {stem!r}")
    return stem


def child_stems() -> tuple[str, ...]:
    """Return every retired child stem in deterministic order."""
    return tuple(sorted(child for rule in CONTAINMENT_RULES for child in rule.children))


__all__ = [
    "CONTAINMENT_RULES",
    "TABLE_CONTRACTS",
    "ContainmentRule",
    "TableContract",
    "child_stems",
    "validate_stem",
]
