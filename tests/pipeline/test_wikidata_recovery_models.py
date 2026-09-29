from __future__ import annotations

import pytest

from osm_polygon_wikidata_only.pipeline._wikidata_recovery.models import (
    QidAuditResult,
    RecoveryAuditResult,
    RecoveryClassification,
    RegionAuditResult,
)


def test_audit_result_lookups_return_matches_and_raise_for_missing_keys() -> None:
    region = RegionAuditResult(
        stem="region-latest",
        fingerprints=(),
        classifications=(),
        polygon_ids_by_qid=(),
        affected_polygon_ids_by_qid=(),
        affected_qids=(),
        affected_polygon_count=0,
    )
    qid = QidAuditResult(
        qid="Q1",
        state=RecoveryClassification.CURRENT,
        regions=(region.stem,),
        polygon_ids=(),
    )
    result = RecoveryAuditResult(
        regions=(region,),
        qids=(qid,),
        upstream_validation_count=0,
        authoritative_cache_hits=0,
    )

    assert result.region(region.stem) is region
    assert result.qid(qid.qid) is qid
    with pytest.raises(KeyError, match="missing-region"):
        result.region("missing-region")
    with pytest.raises(KeyError, match="Q999"):
        result.qid("Q999")
