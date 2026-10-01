"""Stable public API for safe Wikipedia document backfills.

The read-only planner and atomic apply service live in private cohesive modules.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.augmentation._wikipedia_document_migration.application import (
    apply_migration as apply_migration,
)
from osm_polygon_wikidata_only.augmentation._wikipedia_document_migration.models import (
    ApplyResult as ApplyResult,
)
from osm_polygon_wikidata_only.augmentation._wikipedia_document_migration.models import (
    MigrationError as MigrationError,
)
from osm_polygon_wikidata_only.augmentation._wikipedia_document_migration.models import (
    MigrationOperation as MigrationOperation,
)
from osm_polygon_wikidata_only.augmentation._wikipedia_document_migration.models import (
    MigrationPlan as MigrationPlan,
)
from osm_polygon_wikidata_only.augmentation._wikipedia_document_migration.models import (
    StemPlan as StemPlan,
)
from osm_polygon_wikidata_only.augmentation._wikipedia_document_migration.planning import (
    assert_canonical_preserves_legacy as assert_canonical_preserves_legacy,
)
from osm_polygon_wikidata_only.augmentation._wikipedia_document_migration.planning import (
    plan_migration as plan_migration,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    WikipediaDocumentConversionError as WikipediaDocumentConversionError,
)

__all__ = [
    "ApplyResult",
    "MigrationError",
    "MigrationOperation",
    "MigrationPlan",
    "StemPlan",
    "apply_migration",
    "plan_migration",
]
