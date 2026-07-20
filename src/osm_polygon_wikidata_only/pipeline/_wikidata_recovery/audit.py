from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import fact_schema
from osm_polygon_wikidata_only.augmentation.steps import sha256_file
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import polygon_article_schema, polygon_schema
from osm_polygon_wikidata_only.enrichment.wikidata.models import (
    BatchWikidataClient,
    WikidataClient,
    WikidataEntity,
)
from osm_polygon_wikidata_only.enrichment.wikidata.parsing import (
    language_from_site,
    qids_from_osm_tag,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps, loads

from .models import (
    QidAuditResult,
    RecoveryAuditResult,
    RecoveryClassification,
    RegionAuditResult,
)

LOGGER = logging.getLogger(__name__)
RECOVERY_CONTRACT_VERSION = "wikidata-enrichment-integrity-v2"
_INDEX_RELATIVE_PATH = Path("wikidata_recovery/index.json")


@dataclass(frozen=True, slots=True)
class _RegionScan:
    stem: str
    fingerprints: tuple[tuple[str, str], ...]
    polygon_ids_by_qid: tuple[tuple[str, tuple[str, ...]], ...]
    missing_polygon_ids_by_qid: tuple[tuple[str, tuple[str, ...]], ...]
    orphan_fact_ids: tuple[str, ...] = ()
    blocked_reason: str = ""


class _ScanError(ValueError):
    pass


def audit_wikidata_integrity(
    data_root: DataRoot,
    stems: list[str] | tuple[str, ...],
    client: WikidataClient,
    *,
    batch_size: int = 50,
    languages: tuple[str, ...] | None = None,
    max_articles_per_qid: int | None = None,
    log: Callable[[str], None] | None = None,
) -> RecoveryAuditResult:
    """Audit every scoped polygon QID using identities and authoritative outcomes."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    emit = log or LOGGER.info
    scoped_stems = tuple(sorted(set(stems)))
    started_at = time.monotonic()
    emit(f"Wikidata integrity audit started: {len(scoped_stems)} finalized regions")
    index_path = data_root.cache / _INDEX_RELATIVE_PATH
    receipts, contract_matches = _load_receipts(index_path)
    reused_results: dict[str, RegionAuditResult] = {}
    scans: dict[str, _RegionScan] = {}

    for region_index, stem in enumerate(scoped_stems, start=1):
        try:
            fingerprints = _region_fingerprints(data_root, stem)
            reused = _reuse_receipt(stem, fingerprints, receipts.get(stem))
            if contract_matches and reused is not None:
                reused_results[stem] = reused
            else:
                scans[stem] = _scan_region(data_root, stem, fingerprints)
        except (OSError, ValueError, TypeError, KeyError) as error:
            scans[stem] = _RegionScan(stem, (), (), (), (), str(error))
        if _progress_checkpoint(region_index, len(scoped_stems), every=25):
            emit(
                "Wikidata integrity audit local scan "
                f"{region_index}/{len(scoped_stems)} regions; "
                f"{time.monotonic() - started_at:.0f}s elapsed"
            )

    validation_qids = sorted(
        {
            qid
            for scan in scans.values()
            if not scan.blocked_reason
            for qid, polygon_ids in scan.missing_polygon_ids_by_qid
            if polygon_ids
        }
    )
    emit(
        "Wikidata integrity audit local scan complete: "
        f"{len(scoped_stems)} regions; {len(validation_qids)} QIDs require upstream validation; "
        f"{time.monotonic() - started_at:.0f}s elapsed"
    )

    def report_validation(completed: int, total: int) -> None:
        if completed <= batch_size or _progress_checkpoint(
            completed,
            total,
            every=max(batch_size * 10, 1),
        ):
            emit(
                "Wikidata integrity audit upstream validation "
                f"{completed}/{total} QIDs; {time.monotonic() - started_at:.0f}s elapsed"
            )

    entities, cache_hits = _resolve_entities(
        client,
        validation_qids,
        batch_size=batch_size,
        progress=report_validation,
    )
    eligible_sitelinks = {
        qid: _eligible_sitelinks(
            entity,
            languages=languages,
            max_articles_per_qid=max_articles_per_qid,
        )
        for qid, entity in entities.items()
    }

    region_results: list[RegionAuditResult] = []
    changed_receipts = False
    for stem in scoped_stems:
        reused = reused_results.get(stem)
        if reused is not None:
            region_results.append(reused)
            continue
        scan = scans[stem]
        result = _classify_region(scan, entities, eligible_sitelinks)
        region_results.append(result)
        if not result.blocked_reason and not result.requires_repair:
            receipt = _receipt_from_result(result)
            if receipts.get(stem) != receipt:
                receipts[stem] = receipt
                changed_receipts = True

    if changed_receipts:
        _save_receipts(index_path, receipts)

    qid_results = _global_qid_results(region_results, eligible_sitelinks)
    affected_regions = sum(region.requires_repair for region in region_results)
    affected_qids = sum(
        result.state is RecoveryClassification.REPAIR_REQUIRED for result in qid_results
    )
    affected_polygons = sum(region.affected_polygon_count for region in region_results)
    orphan_facts = sum(len(region.orphan_fact_ids) for region in region_results)
    emit(
        "Wikidata integrity audit complete: "
        f"regions scanned {len(region_results)}/{len(scoped_stems)}; "
        f"QIDs examined {len(qid_results)}; authoritative cache hits {cache_hits}; "
        f"QIDs requiring upstream validation {len(validation_qids)}; "
        f"affected QIDs {affected_qids}; affected polygons {affected_polygons}; "
        f"orphan facts {orphan_facts}; "
        f"affected regions {affected_regions}; {time.monotonic() - started_at:.0f}s elapsed"
    )
    return RecoveryAuditResult(
        regions=tuple(region_results),
        qids=tuple(qid_results),
        upstream_validation_count=len(validation_qids),
        authoritative_cache_hits=cache_hits,
    )


def _region_paths(data_root: DataRoot, stem: str) -> tuple[tuple[str, Path, bool], ...]:
    return (
        ("polygons", data_root.processed_polygons / f"{stem}.parquet", True),
        ("polygon_articles", data_root.processed_links / f"{stem}.parquet", True),
        (
            "wikipedia_documents",
            data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet",
            True,
        ),
        (
            "wikidata_facts",
            data_root.processed / "wikidata" / "facts" / f"{stem}.parquet",
            True,
        ),
    )


def _region_fingerprints(data_root: DataRoot, stem: str) -> tuple[tuple[str, str], ...]:
    fingerprints: list[tuple[str, str]] = []
    for label, path, required in _region_paths(data_root, stem):
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"Required recovery input is missing: {path}")
            fingerprints.append((label, ""))
            continue
        fingerprints.append((label, sha256_file(path)))
    return tuple(fingerprints)


def _scan_region(
    data_root: DataRoot,
    stem: str,
    fingerprints: tuple[tuple[str, str], ...],
) -> _RegionScan:
    paths = {label: path for label, path, _ in _region_paths(data_root, stem)}
    _require_schema(paths["polygons"], polygon_schema())
    _require_schema(paths["polygon_articles"], polygon_article_schema())
    _require_schema(paths["wikipedia_documents"], wikipedia_document_schema())
    _require_schema(paths["wikidata_facts"], fact_schema())

    polygon_rows = _read_rows(paths["polygons"], ["polygon_id", "wikidata"])
    link_rows = _read_rows(
        paths["polygon_articles"],
        ["polygon_id", "article_id", "wikidata"],
    )
    document_rows = _read_rows(
        paths["wikipedia_documents"],
        ["article_id", "document_id", "wikidata"],
    )
    fact_rows = _read_rows(paths["wikidata_facts"], ["fact_id", "wikidata"])

    polygons: dict[str, tuple[str, ...]] = {}
    polygon_ids_by_qid: dict[str, list[str]] = {}
    for row in polygon_rows:
        polygon_id = _required_string(row, "polygon_id", "polygons")
        raw_qid = _required_string(row, "wikidata", "polygons")
        qids = qids_from_osm_tag(raw_qid)
        if not qids:
            raise _ScanError(f"polygons contains invalid Wikidata identifier {raw_qid!r}")
        if polygon_id in polygons:
            raise _ScanError(f"polygons contains duplicate polygon_id {polygon_id!r}")
        polygons[polygon_id] = qids
        for qid in qids:
            polygon_ids_by_qid.setdefault(qid, []).append(polygon_id)

    documents_by_article: dict[str, dict[str, Any]] = {}
    document_ids: set[str] = set()
    for row in document_rows:
        article_id = _required_string(row, "article_id", "wikipedia documents")
        document_id = _required_string(row, "document_id", "wikipedia documents")
        qid = _required_string(row, "wikidata", "wikipedia documents")
        if article_id in documents_by_article:
            raise _ScanError(f"wikipedia documents contains duplicate article_id {article_id!r}")
        if document_id in document_ids:
            raise _ScanError(f"wikipedia documents contains duplicate document_id {document_id!r}")
        if qid not in polygon_ids_by_qid:
            raise _ScanError(f"wikipedia document {document_id!r} references absent QID {qid!r}")
        documents_by_article[article_id] = row
        document_ids.add(document_id)

    linked_polygon_qids: set[tuple[str, str]] = set()
    link_ids: set[tuple[str, str]] = set()
    for row in link_rows:
        polygon_id = _required_string(row, "polygon_id", "polygon_articles")
        article_id = _required_string(row, "article_id", "polygon_articles")
        qid = _required_string(row, "wikidata", "polygon_articles")
        identity = (polygon_id, article_id)
        if identity in link_ids:
            raise _ScanError(f"polygon_articles contains duplicate identity {identity!r}")
        link_ids.add(identity)
        polygon_qids = polygons.get(polygon_id)
        if polygon_qids is None:
            raise _ScanError(f"polygon_articles references absent polygon_id {polygon_id!r}")
        if qid not in polygon_qids:
            raise _ScanError(
                f"polygon_articles QID {qid!r} disagrees with polygon {polygon_id!r} "
                f"QIDs {polygon_qids!r}"
            )
        document = documents_by_article.get(article_id)
        if document is None:
            raise _ScanError(f"polygon_articles references absent article_id {article_id!r}")
        if str(document.get("wikidata") or "") != qid:
            raise _ScanError(f"article {article_id!r} disagrees with link QID {qid!r}")
        linked_polygon_qids.add((polygon_id, qid))

    normalized_polygon_ids = tuple(
        (qid, tuple(sorted(polygon_ids))) for qid, polygon_ids in sorted(polygon_ids_by_qid.items())
    )
    missing = tuple(
        (
            qid,
            tuple(
                polygon_id
                for polygon_id in polygon_ids
                if (polygon_id, qid) not in linked_polygon_qids
            ),
        )
        for qid, polygon_ids in normalized_polygon_ids
        if any((polygon_id, qid) not in linked_polygon_qids for polygon_id in polygon_ids)
    )
    valid_polygon_qids = set(polygon_ids_by_qid)
    orphan_fact_ids: list[str] = []
    seen_fact_ids: set[str] = set()
    for row in fact_rows:
        fact_id = _required_string(row, "fact_id", "wikidata facts")
        qid = _required_string(row, "wikidata", "wikidata facts")
        if fact_id in seen_fact_ids:
            raise _ScanError(f"wikidata facts contains duplicate fact_id {fact_id!r}")
        seen_fact_ids.add(fact_id)
        if qid not in valid_polygon_qids:
            orphan_fact_ids.append(fact_id)
    return _RegionScan(
        stem,
        fingerprints,
        normalized_polygon_ids,
        missing,
        tuple(sorted(orphan_fact_ids)),
    )


def _require_schema(path: Path, expected: pa.Schema) -> None:
    actual: pa.Schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
    if not actual.equals(expected, check_metadata=True):
        raise _ScanError(f"Recovery input schema mismatch: {path}")


def _read_rows(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    table: pa.Table = pq.read_table(path, columns=columns)  # type: ignore[no-untyped-call]
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def _required_string(row: Mapping[str, Any], key: str, table: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise _ScanError(f"{table} contains an empty or non-string {key}")
    return value


def _resolve_entities(
    client: WikidataClient,
    qids: list[str],
    *,
    batch_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, WikidataEntity | None], int]:
    resolved: dict[str, WikidataEntity | None] = {}
    cache_hits = 0
    if isinstance(client, BatchWikidataClient):
        for start in range(0, len(qids), batch_size):
            chunk = qids[start : start + batch_size]
            results = client.get_entities(chunk)
            if len(results) != len(chunk):
                raise RuntimeError("Wikidata batch client returned the wrong result count")
            resolved.update(zip(chunk, results, strict=True))
            hits = getattr(client, "last_batch_cache_hits", 0)
            if isinstance(hits, int):
                cache_hits += hits
            if progress is not None:
                progress(min(start + len(chunk), len(qids)), len(qids))
    else:
        for index, qid in enumerate(qids, start=1):
            resolved[qid] = client.get_entity(qid)
            if progress is not None:
                progress(index, len(qids))
    return resolved, cache_hits


def _progress_checkpoint(completed: int, total: int, *, every: int) -> bool:
    return total > 0 and (completed == 1 or completed == total or completed % every == 0)


def _eligible_sitelinks(
    entity: WikidataEntity | None,
    *,
    languages: tuple[str, ...] | None,
    max_articles_per_qid: int | None,
) -> tuple[tuple[str, str], ...]:
    if entity is None:
        return ()
    allowed = set(languages) if languages is not None else None
    sitelinks = tuple(
        (site, title)
        for site, title in sorted(entity.sitelinks.items())
        if allowed is None or language_from_site(site) in allowed
    )
    if max_articles_per_qid is None:
        return sitelinks
    return sitelinks[: max(0, max_articles_per_qid)]


def _classify_region(
    scan: _RegionScan,
    entities: dict[str, WikidataEntity | None],
    eligible_sitelinks: dict[str, tuple[tuple[str, str], ...]],
) -> RegionAuditResult:
    if scan.blocked_reason:
        return RegionAuditResult(
            stem=scan.stem,
            fingerprints=scan.fingerprints,
            classifications=(),
            polygon_ids_by_qid=(),
            affected_polygon_ids_by_qid=(),
            affected_qids=(),
            affected_polygon_count=0,
            orphan_fact_ids=scan.orphan_fact_ids,
            blocked_reason=scan.blocked_reason,
        )
    missing = dict(scan.missing_polygon_ids_by_qid)
    classifications: list[tuple[str, RecoveryClassification]] = []
    affected: list[tuple[str, tuple[str, ...]]] = []
    for qid, _polygon_ids in scan.polygon_ids_by_qid:
        if qid not in missing:
            state = RecoveryClassification.CURRENT
        elif entities[qid] is None:
            state = RecoveryClassification.AUTHORITATIVE_MISSING
        elif not eligible_sitelinks[qid]:
            state = RecoveryClassification.AUTHORITATIVE_NO_SITELINK
        else:
            state = RecoveryClassification.REPAIR_REQUIRED
            affected.append((qid, missing[qid]))
        classifications.append((qid, state))
    return RegionAuditResult(
        stem=scan.stem,
        fingerprints=scan.fingerprints,
        classifications=tuple(classifications),
        polygon_ids_by_qid=scan.polygon_ids_by_qid,
        affected_polygon_ids_by_qid=tuple(affected),
        affected_qids=tuple(qid for qid, _ in affected),
        affected_polygon_count=len(
            {polygon_id for _, polygon_ids in affected for polygon_id in polygon_ids}
        ),
        orphan_fact_ids=scan.orphan_fact_ids,
    )


def _global_qid_results(
    regions: list[RegionAuditResult],
    eligible_sitelinks: dict[str, tuple[tuple[str, str], ...]],
) -> list[QidAuditResult]:
    states: dict[str, list[RecoveryClassification]] = {}
    region_names: dict[str, set[str]] = {}
    polygon_ids: dict[str, set[str]] = {}
    for region in regions:
        region_polygon_ids = dict(region.polygon_ids_by_qid)
        for qid, state in region.classifications:
            states.setdefault(qid, []).append(state)
            region_names.setdefault(qid, set()).add(region.stem)
            polygon_ids.setdefault(qid, set()).update(region_polygon_ids.get(qid, ()))
    priority = (
        RecoveryClassification.BLOCKED,
        RecoveryClassification.REPAIR_REQUIRED,
        RecoveryClassification.CURRENT,
        RecoveryClassification.AUTHORITATIVE_NO_ARTICLE,
        RecoveryClassification.AUTHORITATIVE_NO_SITELINK,
        RecoveryClassification.AUTHORITATIVE_MISSING,
    )
    results: list[QidAuditResult] = []
    for qid in sorted(states):
        state = next(candidate for candidate in priority if candidate in states[qid])
        results.append(
            QidAuditResult(
                qid=qid,
                state=state,
                regions=tuple(sorted(region_names[qid])),
                polygon_ids=tuple(sorted(polygon_ids[qid])),
                sitelinks=eligible_sitelinks.get(qid, ()),
            )
        )
    return results


def _load_receipts(path: Path) -> tuple[dict[str, object], bool]:
    if not path.is_file():
        return {}, False
    try:
        raw: object = loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, False
    if not isinstance(raw, dict) or raw.get("contract_version") != RECOVERY_CONTRACT_VERSION:
        return {}, False
    regions = raw.get("regions")
    if not isinstance(regions, dict):
        return {}, False
    return dict(regions), True


def _reuse_receipt(
    stem: str,
    fingerprints: tuple[tuple[str, str], ...],
    raw_receipt: object,
) -> RegionAuditResult | None:
    if not isinstance(raw_receipt, dict):
        return None
    raw_fingerprints = raw_receipt.get("fingerprints")
    if not isinstance(raw_fingerprints, dict) or dict(fingerprints) != raw_fingerprints:
        return None
    raw_classifications = raw_receipt.get("classifications")
    raw_polygon_ids = raw_receipt.get("polygon_ids")
    if not isinstance(raw_classifications, dict) or not isinstance(raw_polygon_ids, dict):
        return None
    try:
        classifications = tuple(
            (str(qid), RecoveryClassification(str(state)))
            for qid, state in sorted(raw_classifications.items())
        )
        polygon_ids_by_qid = tuple(
            (str(qid), tuple(sorted(str(value) for value in values)))
            for qid, values in sorted(raw_polygon_ids.items())
            if isinstance(values, list)
        )
    except ValueError:
        return None
    if len(polygon_ids_by_qid) != len(raw_polygon_ids):
        return None
    if any(state is RecoveryClassification.REPAIR_REQUIRED for _, state in classifications):
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
        reused=True,
    )


def _receipt_from_result(result: RegionAuditResult) -> dict[str, object]:
    return {
        "fingerprints": dict(result.fingerprints),
        "classifications": {qid: state.value for qid, state in result.classifications},
        "polygon_ids": {qid: list(values) for qid, values in result.polygon_ids_by_qid},
    }


def record_region_recovery_receipt(
    data_root: DataRoot,
    stem: str,
    classifications: Mapping[str, RecoveryClassification],
) -> RegionAuditResult:
    fingerprints = _region_fingerprints(data_root, stem)
    scan = _scan_region(data_root, stem, fingerprints)
    expected_qids = {qid for qid, _ in scan.polygon_ids_by_qid}
    if set(classifications) != expected_qids:
        raise ValueError(
            f"Recovery receipt classifications do not cover region {stem!r}: "
            f"expected {sorted(expected_qids)}, got {sorted(classifications)}"
        )
    if any(state is RecoveryClassification.REPAIR_REQUIRED for state in classifications.values()):
        raise ValueError("A completed recovery receipt cannot contain repair_required")
    result = RegionAuditResult(
        stem=stem,
        fingerprints=fingerprints,
        classifications=tuple(sorted(classifications.items())),
        polygon_ids_by_qid=scan.polygon_ids_by_qid,
        affected_polygon_ids_by_qid=(),
        affected_qids=(),
        affected_polygon_count=0,
        orphan_fact_ids=scan.orphan_fact_ids,
    )
    index_path = data_root.cache / _INDEX_RELATIVE_PATH
    receipts, contract_matches = _load_receipts(index_path)
    if not contract_matches:
        receipts = {}
    receipts[stem] = _receipt_from_result(result)
    _save_receipts(index_path, receipts)
    return result


def _save_receipts(path: Path, receipts: dict[str, object]) -> None:
    payload = {
        "contract_version": RECOVERY_CONTRACT_VERSION,
        "regions": {stem: receipts[stem] for stem in sorted(receipts)},
    }
    atomic_write_text(path, dumps(payload) + "\n")


__all__ = [
    "RECOVERY_CONTRACT_VERSION",
    "audit_wikidata_integrity",
    "record_region_recovery_receipt",
]
