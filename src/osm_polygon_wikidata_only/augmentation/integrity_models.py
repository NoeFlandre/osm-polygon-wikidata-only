"""Result records shared by the shard integrity checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

INTEGRITY_CONTRACT_VERSION = "join-integrity-v1"

REASON_POLYGON_ARTICLES_MISMATCH = "wikidata_mismatch_with_polygon_master"
REASON_WIKIVOYAGE_ABSENT = "wikidata_absent_from_polygons"


@dataclass(frozen=True, slots=True)
class RejectionRecord:
    """One deterministic rejection entry.

    The tuple ``(shard, source_table, identifier, reason)`` is unique across
    the dataset; ``cascaded_sections`` is non-zero only for
    ``wikivoyage_documents`` rejections.
    """

    shard: str
    source_table: str
    identifier: str
    wikidata: str
    expected: str | None
    reason: str
    cascaded_sections: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolygonArticlesIntegrityResult:
    """Result of :func:`enforce_polygon_articles_integrity`."""

    shard: str
    original_row_count: int
    retained_row_count: int
    rejected_row_count: int
    rewritten: bool
    rejections: tuple[RejectionRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "original_row_count": self.original_row_count,
            "retained_row_count": self.retained_row_count,
            "rejected_row_count": self.rejected_row_count,
            "rewritten": self.rewritten,
            "rejections": [record.to_dict() for record in self.rejections],
        }


@dataclass(frozen=True, slots=True)
class WikivoyageIntegrityResult:
    """Result of :func:`enforce_wikivoyage_integrity`."""

    shard: str
    original_document_count: int
    retained_document_count: int
    rejected_document_count: int
    original_section_count: int
    retained_section_count: int
    cascaded_section_count: int
    rewritten_documents: bool
    rewritten_sections: bool
    rejections: tuple[RejectionRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "original_document_count": self.original_document_count,
            "retained_document_count": self.retained_document_count,
            "rejected_document_count": self.rejected_document_count,
            "original_section_count": self.original_section_count,
            "retained_section_count": self.retained_section_count,
            "cascaded_section_count": self.cascaded_section_count,
            "rewritten_documents": self.rewritten_documents,
            "rewritten_sections": self.rewritten_sections,
            "rejections": [record.to_dict() for record in self.rejections],
        }


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Aggregate result of :func:`enforce_all_regions`."""

    contract_version: str
    polygon_articles: tuple[PolygonArticlesIntegrityResult, ...]
    wikivoyage: tuple[WikivoyageIntegrityResult, ...]
    audit_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "polygon_articles": [result.to_dict() for result in self.polygon_articles],
            "wikivoyage": [result.to_dict() for result in self.wikivoyage],
        }

    @property
    def total_polygon_articles_rejected(self) -> int:
        return sum(result.rejected_row_count for result in self.polygon_articles)

    @property
    def total_wikivoyage_documents_rejected(self) -> int:
        return sum(result.rejected_document_count for result in self.wikivoyage)

    @property
    def total_wikivoyage_sections_cascaded(self) -> int:
        return sum(result.cascaded_section_count for result in self.wikivoyage)
