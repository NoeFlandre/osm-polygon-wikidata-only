from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RecoveryClassification(StrEnum):
    CURRENT = "current"
    AUTHORITATIVE_NO_SITELINK = "authoritative_no_sitelink"
    AUTHORITATIVE_NO_ARTICLE = "authoritative_no_article"
    AUTHORITATIVE_MISSING = "authoritative_missing"
    REPAIR_REQUIRED = "repair_required"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class QidAuditResult:
    qid: str
    state: RecoveryClassification
    regions: tuple[str, ...]
    polygon_ids: tuple[str, ...]
    sitelinks: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class RegionAuditResult:
    stem: str
    fingerprints: tuple[tuple[str, str], ...]
    classifications: tuple[tuple[str, RecoveryClassification], ...]
    polygon_ids_by_qid: tuple[tuple[str, tuple[str, ...]], ...]
    affected_polygon_ids_by_qid: tuple[tuple[str, tuple[str, ...]], ...]
    affected_qids: tuple[str, ...]
    affected_polygon_count: int
    orphan_fact_ids: tuple[str, ...] = ()
    blocked_reason: str = ""
    reused: bool = field(default=False, compare=False)

    @property
    def requires_repair(self) -> bool:
        """Whether this region needs a transactional local correction."""
        return bool(self.affected_qids or self.orphan_fact_ids)


@dataclass(frozen=True, slots=True)
class RecoveryAuditResult:
    regions: tuple[RegionAuditResult, ...]
    qids: tuple[QidAuditResult, ...]
    upstream_validation_count: int = field(compare=False)
    authoritative_cache_hits: int = field(compare=False)

    def region(self, stem: str) -> RegionAuditResult:
        for result in self.regions:
            if result.stem == stem:
                return result
        raise KeyError(stem)

    def qid(self, qid: str) -> QidAuditResult:
        for result in self.qids:
            if result.qid == qid:
                return result
        raise KeyError(qid)


__all__ = [
    "QidAuditResult",
    "RecoveryAuditResult",
    "RecoveryClassification",
    "RegionAuditResult",
]
