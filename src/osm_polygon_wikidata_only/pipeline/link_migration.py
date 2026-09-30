"""Public link migration planning and apply API.

Implementation lives in cohesive read-only planning and transactional apply services.
"""

from __future__ import annotations

from osm_polygon_wikidata_only.pipeline._link_migration.application import (
    apply_link_migration,
)
from osm_polygon_wikidata_only.pipeline._link_migration.models import (
    MigrationPlan,
    StemClassification,
    StemPlan,
)
from osm_polygon_wikidata_only.pipeline._link_migration.planning import (
    classify_stem_schema,
    plan_link_migration,
)
from osm_polygon_wikidata_only.pipeline._link_migration.planning import (
    plan_link_migration_normalization_rejections as plan_link_migration_normalization_rejections,
)
from osm_polygon_wikidata_only.pipeline._link_migration.planning import (
    plan_link_migration_normalization_rejections_for_stem as plan_link_migration_normalization_rejections_for_stem,
)

__all__ = [
    "MigrationPlan",
    "StemClassification",
    "StemPlan",
    "apply_link_migration",
    "classify_stem_schema",
    "plan_link_migration",
]
