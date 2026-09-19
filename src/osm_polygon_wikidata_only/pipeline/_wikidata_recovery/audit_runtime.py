"""Small orchestration helpers for the Wikidata recovery audit."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path

from osm_polygon_wikidata_only.enrichment.wikidata.models import WikidataEntity

from .audit_entities import eligible_sitelinks, progress_checkpoint
from .audit_receipts import receipt_from_result, save_receipts
from .audit_types import RegionScan
from .models import QidAuditResult, RecoveryClassification, RegionAuditResult


def validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")


def validation_progress(
    emit: Callable[[str], None],
    batch_size: int,
    started_at: float,
) -> Callable[[int, int], None]:
    def report(completed: int, total: int) -> None:
        if completed <= batch_size or progress_checkpoint(
            completed,
            total,
            every=max(batch_size * 10, 1),
        ):
            emit(
                "Wikidata integrity audit upstream validation "
                f"{completed}/{total} QIDs; {time.monotonic() - started_at:.0f}s elapsed"
            )

    return report


def eligible_sitelinks_by_qid(
    entities: Mapping[str, WikidataEntity | None],
    *,
    languages: tuple[str, ...] | None,
    max_articles_per_qid: int | None,
) -> dict[str, tuple[tuple[str, str], ...]]:
    return {
        qid: eligible_sitelinks(
            entity,
            languages=languages,
            max_articles_per_qid=max_articles_per_qid,
        )
        for qid, entity in entities.items()
    }


def save_changed_receipts(
    index_path: Path,
    receipts: dict[str, object],
    changed: bool,
) -> None:
    if changed:
        save_receipts(index_path, receipts)


def emit_audit_complete(
    emit: Callable[[str], None],
    regions: list[RegionAuditResult],
    qids: list[QidAuditResult],
    cache_hits: int,
    validation_qids: list[str],
    scoped_count: int,
    started_at: float,
) -> None:
    affected_regions, affected_qids, affected_polygons, orphan_facts, orphan_documents = (
        audit_counts(regions, qids)
    )
    emit(
        "Wikidata integrity audit complete: "
        f"regions scanned {len(regions)}/{scoped_count}; "
        f"QIDs examined {len(qids)}; authoritative cache hits {cache_hits}; "
        f"QIDs requiring upstream validation {len(validation_qids)}; "
        f"affected QIDs {affected_qids}; affected polygons {affected_polygons}; "
        f"orphan facts {orphan_facts}; orphan Wikipedia documents {orphan_documents}; "
        f"affected regions {affected_regions}; {time.monotonic() - started_at:.0f}s elapsed"
    )


def audit_counts(
    regions: list[RegionAuditResult],
    qids: list[QidAuditResult],
) -> tuple[int, int, int, int, int]:
    return (
        count_repair_regions(regions),
        count_repair_qids(qids),
        count_affected_polygons(regions),
        count_orphan_facts(regions),
        count_orphan_documents(regions),
    )


def count_repair_regions(regions: list[RegionAuditResult]) -> int:
    return sum(region.requires_repair for region in regions)


def count_repair_qids(qids: list[QidAuditResult]) -> int:
    return sum(result.state is RecoveryClassification.REPAIR_REQUIRED for result in qids)


def count_affected_polygons(regions: list[RegionAuditResult]) -> int:
    return sum(region.affected_polygon_count for region in regions)


def count_orphan_facts(regions: list[RegionAuditResult]) -> int:
    return sum(len(region.orphan_fact_ids) for region in regions)


def count_orphan_documents(regions: list[RegionAuditResult]) -> int:
    return sum(len(region.orphan_document_ids) for region in regions)


def classify_scoped_regions(
    scoped_stems: tuple[str, ...],
    scans: Mapping[str, RegionScan],
    reused_results: Mapping[str, RegionAuditResult],
    entities: Mapping[str, WikidataEntity | None],
    eligible: Mapping[str, tuple[tuple[str, str], ...]],
    receipts: dict[str, object],
    *,
    classify_region_fn: Callable[
        [RegionScan, dict[str, WikidataEntity | None], dict[str, tuple[tuple[str, str], ...]]],
        RegionAuditResult,
    ],
) -> tuple[list[RegionAuditResult], bool]:
    region_results: list[RegionAuditResult] = []
    changed_receipts = False
    for stem in scoped_stems:
        reused = reused_results.get(stem)
        if reused is not None:
            region_results.append(reused)
            continue
        result = classify_region_fn(scans[stem], dict(entities), dict(eligible))
        region_results.append(result)
        changed_receipts |= record_current_receipt(stem, result, receipts)
    return region_results, changed_receipts


def record_current_receipt(
    stem: str,
    result: RegionAuditResult,
    receipts: dict[str, object],
) -> bool:
    if result.blocked_reason or result.requires_repair:
        return False
    receipt = receipt_from_result(result)
    if receipts.get(stem) == receipt:
        return False
    receipts[stem] = receipt
    return True


__all__ = [
    "audit_counts",
    "classify_scoped_regions",
    "count_affected_polygons",
    "count_orphan_documents",
    "count_orphan_facts",
    "count_repair_qids",
    "count_repair_regions",
    "eligible_sitelinks_by_qid",
    "emit_audit_complete",
    "record_current_receipt",
    "save_changed_receipts",
    "validate_batch_size",
    "validation_progress",
]
