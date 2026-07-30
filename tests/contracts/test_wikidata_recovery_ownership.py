"""Ownership contracts for the private Wikidata recovery implementation."""

from osm_polygon_wikidata_only.pipeline._wikidata_recovery import link_rows, repair, validation
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


def test_repair_uses_focused_validation_helpers() -> None:
    assert repair._validate_existing_rows is validation.validate_existing_rows
    assert repair._validate_preservation is validation.validate_preservation


def test_repair_uses_focused_link_row_helpers() -> None:
    assert repair._merge_links is link_rows.merge_links
    assert (
        repair._canonical_wikipedia_links_to_legacy is link_rows.canonical_wikipedia_links_to_legacy
    )
    assert (
        repair._legacy_wikipedia_links_to_canonical is link_rows.legacy_wikipedia_links_to_canonical
    )
