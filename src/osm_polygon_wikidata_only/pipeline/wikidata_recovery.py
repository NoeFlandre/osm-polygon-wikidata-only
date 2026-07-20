"""Private exhaustive Wikidata integrity recovery facade."""

from ._wikidata_recovery import (
    RECOVERY_CONTRACT_VERSION,
    QidAuditResult,
    RecoveryAuditResult,
    RecoveryClassification,
    RecoveryRepairError,
    RecoveryRepairResult,
    RegionAuditResult,
    audit_wikidata_integrity,
    repair_wikidata_region,
)

__all__ = [
    "RECOVERY_CONTRACT_VERSION",
    "QidAuditResult",
    "RecoveryAuditResult",
    "RecoveryClassification",
    "RecoveryRepairError",
    "RecoveryRepairResult",
    "RegionAuditResult",
    "audit_wikidata_integrity",
    "repair_wikidata_region",
]
