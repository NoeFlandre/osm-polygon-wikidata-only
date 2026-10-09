"""Remote inventory reconciliation for the ``sync-dir`` pipeline."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from osm_polygon_wikidata_only.augmentation.orchestrator import augmentation_is_current
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.pipeline.local_validation import LocalValidationProgress
from osm_polygon_wikidata_only.pipeline.sync_planner import SyncAction

if TYPE_CHECKING:
    from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
    from osm_polygon_wikidata_only.hf.reconciliation import (
        ReconciliationPlan,
        ReconciliationPlanner,
    )

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


@dataclass(slots=True)
class RemoteReconciliation:
    """Remote inputs computed once before sync-state planning."""

    inventory: RemoteInventory | None
    plan: ReconciliationPlan | None
    augmentation_current: dict[str, bool]
    stems_with_gaps: set[str]
    containment_publications: dict[str, tuple[str, ...]]
    core_repaired: bool


def _containment_publications_for_remote(
    parent_children: dict[str, tuple[str, ...]],
    *,
    inventory: RemoteInventory,
    canonical_region_paths: Callable[[str], dict[str, str]],
) -> dict[str, tuple[str, ...]]:
    """Keep only contained children represented by a remote artifact."""
    publications: dict[str, tuple[str, ...]] = {}
    for parent, children in parent_children.items():
        present = tuple(
            child
            for child in children
            if _remote_child_has_artifact(child, inventory, canonical_region_paths)
        )
        if present:
            publications[parent] = present
    return publications


def _remote_child_has_artifact(
    child: str,
    inventory: RemoteInventory,
    canonical_region_paths: Callable[[str], dict[str, str]],
) -> bool:
    """Return whether any canonical artifact for a child is remote."""
    return any(inventory.contains(path) for path in canonical_region_paths(child).values())


def _reconciliation_gap_counts(
    input_stems: set[str],
    missing: set[tuple[str, str]],
) -> tuple[int, int]:
    """Count regions with missing core files and augmentation files."""
    augmentation_corpora = (
        "wikipedia/documents",
        "wikipedia/sections",
        "wikivoyage/documents",
        "wikivoyage/sections",
        "wikidata/facts",
    )
    core_count = sum(
        (stem, "polygons") in missing or (stem, "polygon_articles") in missing
        for stem in input_stems
    )
    augmentation_count = sum(
        any((stem, corpus) in missing for corpus in augmentation_corpora) for stem in input_stems
    )
    return core_count, augmentation_count


def remote_reconciliation_helpers(
    enabled: bool,
) -> tuple[type[ReconciliationPlanner] | None, Callable[[str], dict[str, str]] | None]:
    if not enabled:
        return None, None
    # Keep these imports lazy so local-only callers do not capture temporary
    # test replacements in later push-enabled runs.
    from osm_polygon_wikidata_only.hf.reconciliation import (  # noqa: PLC0415
        ReconciliationPlanner,
    )
    from osm_polygon_wikidata_only.hf.repo_layout import (  # noqa: PLC0415
        canonical_region_paths,
    )

    return ReconciliationPlanner, canonical_region_paths


@dataclass(frozen=True, slots=True)
class RemoteSource:
    """Hub client or pre-fetched inventory that supplies the remote dataset state."""

    hub: HfHub | None
    inventory_override: RemoteInventory | None


@dataclass(frozen=True, slots=True)
class RemoteHelpers:
    """Injectable remote reconciliation collaborators; None selects the defaults."""

    canonical_region_paths: Callable[[str], dict[str, str]] | None = None
    planner_cls: type[ReconciliationPlanner] | None = None


def prepare_remote_reconciliation(
    *,
    enabled: bool,
    data_root: DataRoot,
    settings: Settings,
    input_stems: set[str],
    source: RemoteSource,
    validate_augmentation: Callable[[DataRoot, list[str]], dict[str, bool]],
    load_retired_parent_children: Callable[[Path], dict[str, tuple[str, ...]]],
    helpers: RemoteHelpers = RemoteHelpers(),
) -> RemoteReconciliation:
    """Prepare remote reconciliation inputs without work on local-only runs."""
    if not enabled:
        return RemoteReconciliation(None, None, {}, set(), {}, False)
    canonical_region_paths, planner_cls = require_remote_helpers(
        helpers.canonical_region_paths, helpers.planner_cls
    )

    augmentation_current = validate_augmentation(data_root, sorted(input_stems))
    inventory = _remote_inventory(
        source.inventory_override,
        repo_id=settings.repo_id,
        hub=source.hub,
        token=settings.hf_token,
    )
    retired_groups = load_retired_parent_children(data_root.processed)
    containment_publications = _containment_publications_for_remote(
        retired_groups,
        inventory=inventory,
        canonical_region_paths=canonical_region_paths,
    )
    reconciliation_plan, stems_with_gaps, core_repaired = _plan_remote_reconciliation(
        planner_cls,
        data_root=data_root,
        inventory=inventory,
        input_stems=input_stems,
        augmentation_current=augmentation_current,
    )
    return RemoteReconciliation(
        inventory,
        reconciliation_plan,
        augmentation_current,
        stems_with_gaps,
        containment_publications,
        core_repaired,
    )


def _plan_remote_reconciliation(
    planner_cls: type[ReconciliationPlanner],
    *,
    data_root: DataRoot,
    inventory: RemoteInventory,
    input_stems: set[str],
    augmentation_current: dict[str, bool],
) -> tuple[ReconciliationPlan, set[str], bool]:
    """Plan remote gaps, log their counts, and report whether core is repaired."""
    reconciliation_plan = planner_cls(
        data_root=data_root,
        inventory=inventory,
        stems=input_stems,
        augmentation_current=augmentation_current,
    ).plan()
    stems_with_gaps = set(reconciliation_plan.stems_to_publish) | set(
        reconciliation_plan.stems_to_augment
    )
    missing = set(reconciliation_plan.missing)
    core_repaired = any(
        core_repair_required(SyncAction.PUBLISH, stem, missing) for stem in stems_with_gaps
    )
    missing_core_count, missing_aug_count = _reconciliation_gap_counts(
        input_stems,
        missing,
    )
    LOGGER.info(
        "Remote reconciliation: %d regions missing core artifacts, %d missing augmentation artifacts",
        missing_core_count,
        missing_aug_count,
    )
    return reconciliation_plan, stems_with_gaps, core_repaired


def require_remote_helpers(
    canonical_region_paths: Callable[[str], dict[str, str]] | None,
    planner_cls: type[ReconciliationPlanner] | None,
) -> tuple[Callable[[str], dict[str, str]], type[ReconciliationPlanner]]:
    if canonical_region_paths is None or planner_cls is None:
        raise RuntimeError("Remote reconciliation helpers are required when push is enabled")
    return canonical_region_paths, planner_cls


def _remote_inventory(
    override: RemoteInventory | None,
    *,
    repo_id: str,
    hub: HfHub | None,
    token: str | None,
) -> RemoteInventory:
    if override is not None:
        return override
    return RemoteInventory.fetch(repo_id=repo_id, hub=hub, token=token)


def core_repair_required(
    action: SyncAction,
    stem: str,
    missing: set[tuple[str, str]],
) -> bool:
    """Return whether a state changes core artifacts requiring refresh."""
    if action is SyncAction.PROCESS:
        return True
    if action not in (SyncAction.PUBLISH, SyncAction.AUGMENT):
        return False
    return (stem, "polygons") in missing or (stem, "polygon_articles") in missing


def log_remote_reconciliation_summary(
    *,
    stems_with_gaps: set[str],
    core_repaired: bool,
    metadata_repaired: bool,
    log: Callable[[str], None] = LOGGER.info,
) -> None:
    """Emit the final remote-reconciliation summary line.

    The summary is derived strictly from signals produced by the
    upload pipeline -- it must never claim maps or README were
    refreshed unless a core or metadata-only publication actually
    refreshed them. Claims are made only after the background
    upload queue has drained successfully.

    The ``log`` parameter is the ``info``-level callable that
    receives the rendered message. Tests pass a recorded logger
    spy to observe emissions without depending on caplog state
    or module-level logger configuration.
    """
    log(
        reconciliation_summary_message(
            len(stems_with_gaps),
            core_repaired,
            metadata_repaired,
        )
    )


def reconciliation_summary_message(
    repaired_regions: int,
    core_repaired: bool,
    metadata_repaired: bool,
) -> str:
    """Build the stable summary text from upload outcomes."""
    maps_refreshed = core_repaired or metadata_repaired
    if maps_refreshed:
        if repaired_regions:
            return (
                f"Remote reconciliation complete: {repaired_regions} "
                "regions repaired; README and maps refreshed"
            )
        return "Remote reconciliation complete: README and maps refreshed"
    if repaired_regions:
        return f"Remote reconciliation complete: {repaired_regions} regions repaired"
    return "Remote reconciliation complete: converged"


def validate_local_augmentation_state(
    data_root: DataRoot,
    stems: list[str],
) -> dict[str, bool]:
    """Validate the local augmentation state for every stem exactly once.

    Wraps :class:`LocalValidationProgress` so the operator gets a
    bounded, periodic startup progress signal even when this phase
    takes several minutes. Each stem is visited exactly once and
    the resulting mapping is returned for downstream planning.
    """
    progress = LocalValidationProgress(
        validator=lambda stem: augmentation_is_current(data_root, stem),
        stems=list(stems),
        log=LOGGER.info,
        clock=time.monotonic,
        progress_interval_s=30.0,
        quiet_threshold=25,
        phase_label="regions",
    )
    return progress.run()
