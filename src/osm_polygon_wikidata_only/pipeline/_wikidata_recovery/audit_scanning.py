"""Scan finalized regional tables before upstream QID validation."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from osm_polygon_wikidata_only.config.paths import DataRoot

from .audit_receipts import reuse_receipt
from .audit_types import RegionScan
from .models import RegionAuditResult


class ScanHooks:
    """Patchable seams used by the compatibility audit facade."""

    def __init__(
        self,
        *,
        region_fingerprints: Callable[[DataRoot, str], tuple[tuple[str, str], ...]],
        scan_region: Callable[[DataRoot, str, tuple[tuple[str, str], ...]], RegionScan],
        progress_checkpoint: Callable[[int, int], bool],
    ) -> None:
        self.region_fingerprints = region_fingerprints
        self.scan_region = scan_region
        self.progress_checkpoint = progress_checkpoint


def scan_regions(
    data_root: DataRoot,
    scoped_stems: tuple[str, ...],
    receipts: Mapping[str, object],
    *,
    contract_matches: bool,
    started_at: float,
    emit: Callable[[str], None],
    hooks: ScanHooks,
) -> tuple[dict[str, RegionScan], dict[str, RegionAuditResult]]:
    scans: dict[str, RegionScan] = {}
    reused_results: dict[str, RegionAuditResult] = {}
    for region_index, stem in enumerate(scoped_stems, start=1):
        reused, scan = scan_region_for_audit(
            data_root,
            stem,
            receipts.get(stem),
            contract_matches=contract_matches,
            hooks=hooks,
        )
        if reused is not None:
            reused_results[stem] = reused
        else:
            assert scan is not None
            scans[stem] = scan
        emit_scan_progress(
            emit,
            region_index,
            len(scoped_stems),
            started_at,
            progress_checkpoint=hooks.progress_checkpoint,
        )
    return scans, reused_results


def scan_region_for_audit(
    data_root: DataRoot,
    stem: str,
    raw_receipt: object,
    *,
    contract_matches: bool,
    hooks: ScanHooks,
) -> tuple[RegionAuditResult | None, RegionScan | None]:
    try:
        fingerprints = hooks.region_fingerprints(data_root, stem)
        reused = reuse_receipt(stem, fingerprints, raw_receipt)
        if contract_matches and reused is not None:
            return reused, None
        return None, hooks.scan_region(data_root, stem, fingerprints)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return None, RegionScan(
            stem=stem,
            fingerprints=(),
            polygon_ids_by_qid=(),
            missing_polygon_ids_by_qid=(),
            blocked_reason=str(error),
        )


def emit_scan_progress(
    emit: Callable[[str], None],
    completed: int,
    total: int,
    started_at: float,
    *,
    progress_checkpoint: Callable[[int, int], bool],
) -> None:
    if progress_checkpoint(completed, total, every=25):
        emit(
            "Wikidata integrity audit local scan "
            f"{completed}/{total} regions; "
            f"{time.monotonic() - started_at:.0f}s elapsed"
        )


def validation_qids(scans: Mapping[str, RegionScan]) -> list[str]:
    return sorted(
        {
            qid
            for scan in scans.values()
            if not scan.blocked_reason
            for qid, polygon_ids in scan.missing_polygon_ids_by_qid
            if polygon_ids
        }
    )


__all__ = [
    "ScanHooks",
    "emit_scan_progress",
    "scan_region_for_audit",
    "scan_regions",
    "validation_qids",
]
