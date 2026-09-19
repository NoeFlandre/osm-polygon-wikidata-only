"""Read, validate, reuse, and persist recovery audit receipts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps, loads

from .audit_types import RegionScan
from .models import RecoveryClassification, RegionAuditResult

RECOVERY_CONTRACT_VERSION = "wikidata-enrichment-integrity-v2"
INDEX_RELATIVE_PATH = Path("wikidata_recovery/index.json")


def load_receipts(path: Path) -> tuple[dict[str, object], bool]:
    if not path.is_file():
        return {}, False
    try:
        raw = loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, False
    return decode_receipts(raw)


def decode_receipts(raw: object) -> tuple[dict[str, object], bool]:
    if not isinstance(raw, dict) or raw.get("contract_version") != RECOVERY_CONTRACT_VERSION:
        return {}, False
    regions = raw.get("regions")
    if not isinstance(regions, dict):
        return {}, False
    return dict(cast(dict[str, object], regions)), True


def reuse_receipt(
    stem: str,
    fingerprints: tuple[tuple[str, str], ...],
    raw_receipt: object,
) -> RegionAuditResult | None:
    fields = receipt_fields(raw_receipt, fingerprints)
    if fields is None:
        return None
    raw_classifications, raw_polygon_ids = fields
    entries = parse_receipt_entries(raw_classifications, raw_polygon_ids)
    if entries is None:
        return None
    classifications, polygon_ids_by_qid = entries
    if receipt_needs_repair(classifications):
        return None
    return RegionAuditResult(
        stem=stem,
        fingerprints=fingerprints,
        classifications=classifications,
        polygon_ids_by_qid=polygon_ids_by_qid,
        affected_polygon_ids_by_qid=(),
        affected_qids=(),
        affected_polygon_count=0,
        orphan_fact_ids=(),
        orphan_document_ids=(),
        reused=True,
    )


def receipt_fields(
    raw_receipt: object,
    fingerprints: tuple[tuple[str, str], ...],
) -> tuple[dict[object, object], dict[object, object]] | None:
    if not isinstance(raw_receipt, dict):
        return None
    raw_fingerprints = raw_receipt.get("fingerprints")
    if not fingerprints_match(raw_fingerprints, fingerprints):
        return None
    return receipt_maps(cast(Mapping[str, object], raw_receipt))


def fingerprints_match(
    raw_fingerprints: object,
    fingerprints: tuple[tuple[str, str], ...],
) -> bool:
    return isinstance(raw_fingerprints, dict) and dict(fingerprints) == raw_fingerprints


def receipt_maps(
    raw_receipt: Mapping[str, object],
) -> tuple[dict[object, object], dict[object, object]] | None:
    raw_classifications = raw_receipt.get("classifications")
    raw_polygon_ids = raw_receipt.get("polygon_ids")
    if not isinstance(raw_classifications, dict) or not isinstance(raw_polygon_ids, dict):
        return None
    return cast(dict[object, object], raw_classifications), cast(
        dict[object, object], raw_polygon_ids
    )


def parse_receipt_entries(
    raw_classifications: dict[object, object],
    raw_polygon_ids: dict[object, object],
) -> (
    tuple[
        tuple[tuple[str, RecoveryClassification], ...],
        tuple[tuple[str, tuple[str, ...]], ...],
    ]
    | None
):
    classifications = parse_receipt_classifications(raw_classifications)
    polygon_ids_by_qid = parse_receipt_polygon_ids(raw_polygon_ids)
    if classifications is None or polygon_ids_by_qid is None:
        return None
    return classifications, polygon_ids_by_qid


def parse_receipt_classifications(
    raw_classifications: dict[object, object],
) -> tuple[tuple[str, RecoveryClassification], ...] | None:
    try:
        return tuple(
            (str(qid), RecoveryClassification(str(state)))
            for qid, state in sorted(raw_classifications.items())
        )
    except (TypeError, ValueError):
        return None


def parse_receipt_polygon_ids(
    raw_polygon_ids: dict[object, object],
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    parsed: list[tuple[str, tuple[str, ...]]] = []
    for qid, values in sorted(raw_polygon_ids.items()):
        if not isinstance(values, list):
            return None
        parsed.append((str(qid), tuple(sorted(str(value) for value in values))))
    return tuple(parsed)


def receipt_needs_repair(
    classifications: tuple[tuple[str, RecoveryClassification], ...],
) -> bool:
    return any(state is RecoveryClassification.REPAIR_REQUIRED for _, state in classifications)


def receipt_from_result(result: RegionAuditResult) -> dict[str, object]:
    return {
        "fingerprints": dict(result.fingerprints),
        "classifications": {qid: state.value for qid, state in result.classifications},
        "polygon_ids": {qid: list(values) for qid, values in result.polygon_ids_by_qid},
    }


def record_recovery_receipt(
    data_root: DataRoot,
    stem: str,
    classifications: Mapping[str, RecoveryClassification],
    *,
    fingerprints: tuple[tuple[str, str], ...],
    scan: RegionScan,
) -> RegionAuditResult:
    expected_qids = {qid for qid, _ in scan.polygon_ids_by_qid}
    validate_receipt_classifications(stem, classifications, expected_qids)
    result = receipt_result(stem, fingerprints, scan, classifications)
    store_receipt(data_root, stem, result)
    return result


def validate_receipt_classifications(
    stem: str,
    classifications: Mapping[str, RecoveryClassification],
    expected_qids: set[str],
) -> None:
    if set(classifications) != expected_qids:
        raise ValueError(
            f"Recovery receipt classifications do not cover region {stem!r}: "
            f"expected {sorted(expected_qids)}, got {sorted(classifications)}"
        )
    if any(state is RecoveryClassification.REPAIR_REQUIRED for state in classifications.values()):
        raise ValueError("A completed recovery receipt cannot contain repair_required")


def receipt_result(
    stem: str,
    fingerprints: tuple[tuple[str, str], ...],
    scan: RegionScan,
    classifications: Mapping[str, RecoveryClassification],
) -> RegionAuditResult:
    return RegionAuditResult(
        stem=stem,
        fingerprints=fingerprints,
        classifications=tuple(sorted(classifications.items())),
        polygon_ids_by_qid=scan.polygon_ids_by_qid,
        affected_polygon_ids_by_qid=(),
        affected_qids=(),
        affected_polygon_count=0,
        orphan_fact_ids=scan.orphan_fact_ids,
        orphan_document_ids=scan.orphan_document_ids,
    )


def store_receipt(data_root: DataRoot, stem: str, result: RegionAuditResult) -> None:
    index_path = data_root.cache / INDEX_RELATIVE_PATH
    receipts, contract_matches = load_receipts(index_path)
    if not contract_matches:
        receipts = {}
    receipts[stem] = receipt_from_result(result)
    save_receipts(index_path, receipts)


def save_receipts(path: Path, receipts: dict[str, object]) -> None:
    payload = {
        "contract_version": RECOVERY_CONTRACT_VERSION,
        "regions": {stem: receipts[stem] for stem in sorted(receipts)},
    }
    atomic_write_text(path, dumps(payload) + "\n")


__all__ = [
    "INDEX_RELATIVE_PATH",
    "RECOVERY_CONTRACT_VERSION",
    "decode_receipts",
    "fingerprints_match",
    "load_receipts",
    "parse_receipt_classifications",
    "parse_receipt_entries",
    "parse_receipt_polygon_ids",
    "receipt_fields",
    "receipt_from_result",
    "receipt_maps",
    "receipt_needs_repair",
    "receipt_result",
    "record_recovery_receipt",
    "save_receipts",
    "store_receipt",
    "validate_receipt_classifications",
]
