"""Ownership contracts for the private Wikidata recovery implementation."""

from osm_polygon_wikidata_only.pipeline._wikidata_recovery import repair
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.models import (
    RecoveryRepairError,
    RecoveryRepairResult,
)
from osm_polygon_wikidata_only.pipeline._wikidata_recovery.storage import (
    read_table,
    region_paths,
    write_table,
)


def test_repair_facade_reexports_models_by_identity() -> None:
    assert repair.RecoveryRepairError is RecoveryRepairError
    assert repair.RecoveryRepairResult is RecoveryRepairResult


def test_repair_uses_shared_storage_helpers() -> None:
    assert repair._region_paths is region_paths
    assert repair._read_table is read_table
    assert repair._write_table is write_table
