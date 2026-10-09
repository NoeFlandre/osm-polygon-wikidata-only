from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.pipeline._wikidata_recovery import repair_outputs
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.checkpoints import (
    RecoveryCheckpointStore,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.models import (
    RecoveryClassification,
    RegionAuditResult,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.repair_types import (
    _RepairInputs,
    _RepairOutputs,
)


def test_unchanged_repair_records_receipt_and_clears_checkpoint(tmp_path: Path) -> None:
    data_root = DataRoot(tmp_path)
    region = RegionAuditResult(
        stem="region-latest",
        fingerprints=(),
        classifications=(),
        polygon_ids_by_qid=(),
        affected_polygon_ids_by_qid=(),
        affected_qids=(),
        affected_polygon_count=0,
    )
    outputs = _RepairOutputs(
        updated_polygons=[],
        persisted_links=[],
        merged_documents=[],
        merged_sections=[],
        merged_facts=[],
        terminal_classifications={"Q1": RecoveryClassification.AUTHORITATIVE_NO_ARTICLE},
        affected_qids=("Q1",),
        affected_polygon_ids=set(),
        map_inputs_changed=False,
        changed=False,
    )
    inputs = cast(_RepairInputs, object())
    checkpoint = RecoveryCheckpointStore(tmp_path, region.stem, "plan")
    receipts: list[tuple[str, dict[str, RecoveryClassification]]] = []

    def record_receipt(
        _root: DataRoot, stem: str, classifications: dict[str, RecoveryClassification]
    ) -> None:
        receipts.append((stem, classifications))

    result = repair_outputs.persist_repair_outputs(
        data_root,
        region,
        inputs,
        outputs,
        checkpoint,
        repair_outputs.RepairPersistence(
            transaction_root=tmp_path / "transactions",
            wikidata_client=cast(Any, object()),
            settings=Settings(
                languages=None,
                fetch_full_text=True,
                max_articles_per_qid=None,
                enrichment_batch_size=50,
            ),
            before_commit=None,
            record_receipt_fn=record_receipt,
        ),
    )

    assert result.changed is False
    assert receipts == [(region.stem, outputs.terminal_classifications)]
    assert not (tmp_path / "transactions").exists()
