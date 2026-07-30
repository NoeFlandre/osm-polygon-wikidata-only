"""Stable facade and focused ownership contracts for link migration."""

from __future__ import annotations

from osm_polygon_wikidata_only.pipeline import link_migration
from osm_polygon_wikidata_only.pipeline._link_migration import (
    conversion,
    models,
    transaction,
)


def test_link_migration_models_have_focused_owner() -> None:
    assert link_migration.StemClassification is models.StemClassification
    assert link_migration.StemPlan is models.StemPlan
    assert link_migration.MigrationPlan is models.MigrationPlan


def test_link_migration_conversion_has_focused_owner() -> None:
    assert link_migration._build_canonical_rows is conversion.build_canonical_rows


def test_link_migration_transaction_has_focused_owner() -> None:
    assert (
        link_migration._commit_ordered_replacements
        is transaction.commit_ordered_replacements
    )
