"""Deep application module for the unified sync execution phase.

``cli.run_sync`` remains the composition root: it parses arguments, performs
local migration, builds runtime clients, reconciles the remote inventory, and
plans states.  This module owns what happens after that plan exists.  Its
single public operation, :meth:`SyncApplication.run`, hides callback wiring,
recovery, augmentation, publication, queue draining, and final metadata
refresh behind explicit injected services.

The service bundle is deliberately explicit.  It keeps this module free of
CLI parsing and network-client construction while preserving the existing
``run_sync.execute`` monkeypatch points and making the lifecycle independently
testable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.pipeline.sync_planner import RegionSyncState, SyncAction

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


@dataclass(frozen=True, slots=True)
class SyncApplicationServices:
    """Injected boundaries used by :class:`SyncApplication`.

    Keeping all side-effectful collaborators here makes the application
    deterministic in tests and prevents the deep module from reaching into
    CLI globals or constructing clients on its own.
    """

    extract_pbf: Callable[[Path], Any]
    process_extracted_pbf: Callable[[Any], Any]
    augment_region: Callable[..., Any]
    load_existing_augmentation: Callable[..., Any]
    recover_region: Callable[[RegionSyncState], Any]
    run_sync: Callable[..., int]
    plan_link_migration: Callable[..., Any]
    apply_link_migration: Callable[..., Any]
    audit_wikidata_integrity: Callable[..., Any]
    ensure_recovery_audit_unblocked: Callable[[Any], None]
    repair_wikidata_region: Callable[..., Any]
    prepare_local_retirement: Callable[..., Any]
    add_pending_publications: Callable[..., Any]
    record_region_recovery_receipt: Callable[..., Any]
    assemble_region_upload: Callable[..., list[PublicationOp]]
    assemble_metadata_only_upload: Callable[..., list[PublicationOp]]
    load_existing_core_for_publication: Callable[..., Any]
    commit_message: Callable[[RegionSyncState], str]
    log_remote_reconciliation_summary: Callable[..., None]
    load_metadata_refresh_marker: Callable[..., Any]
    set_metadata_refresh_marker: Callable[..., Any]
    clear_metadata_refresh_marker: Callable[..., Any]
    augmentation_progress: Callable[[], Any]
    sync_heartbeat: Callable[..., Any]
    logger: Any = LOGGER


@dataclass(slots=True)
class SyncApplicationContext:
    """Planned state and mutable reconciliation facts for one sync run."""

    data_root: DataRoot
    settings: Settings
    runtime: Any
    augmentation_client: Any
    states: list[RegionSyncState]
    push_enabled: bool
    dry_run: bool
    pending_stems: set[str]
    stems_with_gaps: set[str]
    reconciliation_plan: Any
    upload_queue: Any | None
    publish_builder: Callable[..., list[PublicationOp]] | None = None
    core_will_be_repaired: bool = False
    core_repaired: bool = False
    containment_enqueued: bool = False


@dataclass(frozen=True, slots=True)
class SyncApplicationResult:
    """Stable summary returned after the application lifecycle finishes."""

    return_code: int
    core_will_be_repaired: bool
    core_repaired: bool
    metadata_repaired: bool


class SyncApplication:
    """Execute a planned sync while hiding callback and upload lifecycle detail."""

    def __init__(
        self, *, context: SyncApplicationContext, services: SyncApplicationServices
    ) -> None:
        self.context = context
        self.services = services
        self._recovered_stems: set[str] = set()
        self._recovery_map_refresh_stems: set[str] = set()
        self._recovery_classifications: dict[str, dict[Any, Any]] = {}
        self._region_publication_submitted = False
        self._metadata_repair_requested = False
        self._deferred_metadata_stems: dict[str, str] = {}

    def run(self) -> SyncApplicationResult:
        """Run all planned states and finalize successful metadata refreshes."""
        rc, metadata_repaired, failures = self._run_with_cleanup()
        if rc == 0 and not failures:
            metadata_repaired = self._refresh_metadata_marker(metadata_repaired)
        return self._finish(rc, failures, metadata_repaired)

    def _run_with_cleanup(self) -> tuple[int, bool, list[str]]:
        rc = 0
        metadata_repaired = False
        try:
            rc, metadata_repaired = self._execute_plan()
        except Exception as error:
            if self.context.upload_queue is not None:
                self.services.logger.error("Unified sync aborted: %s", error)
            raise
        finally:
            failures = self._close_uploads()
            if failures:
                rc = 1
        return rc, metadata_repaired, failures

    def _execute_plan(self) -> tuple[int, bool]:
        callbacks = self._runner_callbacks()
        rc = self.services.run_sync(self.context.states, **callbacks)
        self.context.core_will_be_repaired = self.context.core_will_be_repaired or bool(
            self._recovered_stems
        )
        self._request_metadata_repair(rc, callbacks["submit_upload"])
        return rc, False

    def _runner_callbacks(self) -> dict[str, Any]:
        if not self.context.push_enabled:
            publish_builder = None
            submit_upload = None
        else:
            publish_builder = self.context.publish_builder or self._build_region_publication
            submit_upload = self._submit_upload
        return {
            "extract_pbf": self.services.extract_pbf,
            "process_extracted_pbf": self.services.process_extracted_pbf,
            "augment_region": self._augment,
            "build_upload_files": publish_builder,
            "commit_message": self.services.commit_message,
            "submit_upload": submit_upload,
            "close_uploads": None,
            "load_existing_augmentation": self._load_existing,
            "recover_region": self._recover,
            "on_complete": self._prepare_publication,
        }

    def _metadata_repair_needed(self, rc: int) -> bool:
        if not self._successful_code(rc):
            return False
        if not self.context.push_enabled:
            return False
        if not self._repository_refresh_requested():
            return False
        return self._no_repair_outputs()

    @staticmethod
    def _successful_code(rc: int) -> bool:
        return rc == 0

    def _repository_refresh_requested(self) -> bool:
        plan = self.context.reconciliation_plan
        return not (plan is None or not plan.repository_refresh)

    def _no_repair_outputs(self) -> bool:
        return not self.context.core_will_be_repaired and not self.context.containment_enqueued

    def _request_metadata_repair(
        self,
        rc: int,
        submit_upload: Callable[[list[PublicationOp], str], None] | None,
    ) -> None:
        """Record that this sync owes a repository-wide metadata repair.

        The repair is not queued behind the regional jobs: the queue keeps
        going after a job exhausts its retries, so a repair riding it could
        publish a manifest, statistics report, map, and card describing
        regional artifacts that never reached the remote. It is published
        once the queue has drained without failures, through the same
        post-drain path the deferred refresh uses.
        """
        if submit_upload is None or not self._metadata_repair_needed(rc):
            return
        self.services.logger.info(
            "Metadata-only repair requested (no region core repair planned); "
            "publishing it after the region queue drains"
        )
        self._metadata_repair_requested = True

    def _refresh_metadata_marker(self, metadata_repaired: bool) -> bool:
        """Publish the repository-wide assets once, when they are still owed.

        This runs only after the upload queue drained without failures, so
        the regional artifacts the assets describe are on the remote. A
        reconciliation repair and a deferred refresh both resolve here, in
        one scan and one upload rather than two.
        """
        if not self.context.push_enabled:
            return metadata_repaired
        marker = self.services.load_metadata_refresh_marker(self.context.data_root)
        if not self._metadata_repair_requested and not self._metadata_refresh_required(marker):
            return metadata_repaired
        self._log_metadata_refresh(marker)
        self._upload_metadata_refresh()
        self._clear_metadata_refresh_state(marker)
        return True

    def _metadata_refresh_required(self, marker: dict[str, Any] | None) -> bool:
        """A migrated region or a deferred regional publication needs the refresh."""
        return marker is not None or self._region_publication_submitted

    def _upload_metadata_refresh(self) -> None:
        """Publish the repository-wide metadata assets synchronously."""
        metadata_ops = self.services.assemble_metadata_only_upload(
            data_root=self.context.data_root,
            repo_id=self.context.settings.repo_id,
            world_land_warning=None,
        )
        if self.context.upload_queue is None:
            raise RuntimeError("Metadata refresh requires an upload queue")
        self.context.upload_queue.upload_synchronously(
            metadata_ops,
            "Repair remote repository metadata and maps",
        )

    def _clear_metadata_refresh_state(self, marker: dict[str, Any] | None) -> None:
        """Retire the marker, if any, and the deferred-publication flag.

        A dry run uploads through the stub hub, so the remote is
        unchanged and the marker stays: retiring it would erase the only
        record that an earlier interrupted run still owes a refresh.
        """
        if marker is not None and not self.context.dry_run:
            self.services.clear_metadata_refresh_marker(self.context.data_root)
        self._region_publication_submitted = False

    def _log_metadata_refresh(self, marker: dict[str, Any] | None) -> None:
        """Record why the repository-wide metadata refresh is running."""
        if marker is None:
            self.services.logger.info("Refreshing repository metadata after unified sync")
            return
        self.services.logger.info(
            "Refreshing repository metadata after %d migrated region(s)",
            len(marker["stems"]),
        )

    def _finish(
        self,
        rc: int,
        failures: list[str],
        metadata_repaired: bool,
    ) -> SyncApplicationResult:
        if rc == 0 and not failures:
            self._log_success(metadata_repaired)
            return self._result(rc, metadata_repaired)
        if rc != 0:
            self.services.logger.error("Unified sync completed with failures (rc=%d)", rc)
        return self._result(rc or 1, metadata_repaired)

    def _log_success(self, metadata_repaired: bool) -> None:
        if not self.context.push_enabled:
            return
        self.context.core_repaired = self.context.core_repaired or bool(self._recovered_stems)
        self.services.log_remote_reconciliation_summary(
            stems_with_gaps=self.context.stems_with_gaps,
            core_repaired=self.context.core_repaired,
            metadata_repaired=metadata_repaired,
            log=self.services.logger.info,
        )

    def _result(self, rc: int, metadata_repaired: bool) -> SyncApplicationResult:
        return SyncApplicationResult(
            return_code=rc,
            core_will_be_repaired=self.context.core_will_be_repaired,
            core_repaired=self.context.core_repaired,
            metadata_repaired=metadata_repaired,
        )

    def _augment(self, state: RegionSyncState) -> Any:
        progress = self.services.augmentation_progress()
        logger = self.services.logger
        logger.info("Sync region %s: augmentation started", state.stem)
        with self._heartbeat(state, progress):
            augmentation_result = self._augment_documents(state, progress)
            augmentation_result = self._audit_augmentation(state, augmentation_result)
        logger.info("Unified sync completed %s: %s", state.stem, augmentation_result.counts)
        return augmentation_result

    def _heartbeat(self, state: RegionSyncState, progress: Any) -> Any:
        actionable = [s for s in self.context.states if s.action is not SyncAction.COMPLETE]
        runtime = self.context.runtime
        return self.services.sync_heartbeat(
            region=state.stem,
            region_index=self._region_index(state, actionable),
            region_total=self._region_total(actionable),
            augmentation_snapshot=progress.snapshot,
            scheduler_snapshot=runtime.scheduler.snapshot,
            auth_snapshot=runtime.session.auth_snapshot,
            log=self.services.logger.info,
        )

    @staticmethod
    def _region_index(state: RegionSyncState, actionable: list[RegionSyncState]) -> int:
        if state not in actionable:
            return 0
        return actionable.index(state) + 1

    def _region_total(self, actionable: list[RegionSyncState]) -> int:
        if actionable:
            return len(actionable)
        return len(self.context.states)

    def _augment_documents(self, state: RegionSyncState, progress: Any) -> Any:
        result = self.services.augment_region(
            self.context.data_root,
            state.stem,
            self.context.augmentation_client,
            progress=progress,
        )
        if not getattr(result, "wikipedia_documents_path", None):
            return result
        current_links = self.services.plan_link_migration(
            self.context.data_root.processed,
            stems={state.stem},
        )
        if self._links_need_migration(current_links):
            self.services.apply_link_migration(
                self.context.data_root.processed,
                stems={state.stem},
            )
        return self.services.load_existing_augmentation(self.context.data_root, state.stem)

    @staticmethod
    def _links_need_migration(current_links: Any) -> bool:
        return bool(
            current_links.stems and current_links.stems[0].classification.value != "canonical"
        )

    def _audit_augmentation(self, state: RegionSyncState, result: Any) -> Any:
        runtime = self.context.runtime
        logger = self.services.logger
        audit = self.services.audit_wikidata_integrity(
            self.context.data_root,
            [state.stem],
            runtime.wikidata,
            batch_size=self.context.settings.enrichment_batch_size,
            languages=self.context.settings.languages,
            max_articles_per_qid=self.context.settings.max_articles_per_qid,
            log=logger.info,
        )
        self.services.ensure_recovery_audit_unblocked(audit)
        region = audit.region(state.stem)
        if not region.requires_repair:
            return result
        repair_result = self.services.repair_wikidata_region(
            self.context.data_root,
            region,
            wikidata_client=runtime.wikidata,
            wikipedia_client=runtime.wikipedia,
            augmentation_client=self.context.augmentation_client,
            settings=self.context.settings,
            log=logger.info,
            scheduler_snapshot=runtime.scheduler.snapshot,
        )
        if not repair_result.changed:
            return result
        self._mark_recovered(state.stem, repair_result.map_inputs_changed)
        return self.services.load_existing_augmentation(self.context.data_root, state.stem)

    def _mark_recovered(self, stem: str, map_inputs_changed: bool) -> None:
        self._recovered_stems.add(stem)
        if map_inputs_changed:
            self._recovery_map_refresh_stems.add(stem)

    def _load_existing(self, state: RegionSyncState) -> Any:
        return self.services.load_existing_augmentation(self.context.data_root, state.stem)

    def _migrate_links_if_needed(self, stem: str) -> bool:
        current = self.services.plan_link_migration(
            self.context.data_root.processed,
            stems={stem},
        )
        if not current.stems:
            return False
        stem_plan = current.stems[0]
        if stem_plan.classification.value == "canonical":
            return False
        if stem_plan.classification.value == "BLOCKED":
            raise RuntimeError(
                f"Unified polygon-document link migration is blocked for {stem}: {stem_plan.reason}"
            )
        self.services.apply_link_migration(self.context.data_root.processed, stems={stem})
        return True

    def _recover(self, state: RegionSyncState) -> Any:
        plan = self._recovery_plan(state)
        if plan.requires_repair:
            return self._recover_repair(state, plan)
        return self._recover_healthy(state, plan)

    def _recovery_plan(self, state: RegionSyncState) -> Any:
        runtime = self.context.runtime
        logger = self.services.logger
        audit = self.services.audit_wikidata_integrity(
            self.context.data_root,
            [state.stem],
            runtime.wikidata,
            batch_size=self.context.settings.enrichment_batch_size,
            languages=self.context.settings.languages,
            max_articles_per_qid=self.context.settings.max_articles_per_qid,
            log=logger.info,
        )
        self.services.ensure_recovery_audit_unblocked(audit)
        return audit.region(state.stem)

    def _recover_healthy(self, state: RegionSyncState, plan: Any) -> Any:
        if self._migrate_links_if_needed(state.stem):
            self._recovery_classifications[state.stem] = dict(plan.classifications)
            self._mark_recovered(state.stem, False)
            if self.context.push_enabled:
                return self._load_existing(state)
            return None
        if self.context.push_enabled and state.stem in self.context.pending_stems:
            return self._load_existing(state)
        return None

    def _recover_repair(self, state: RegionSyncState, plan: Any) -> Any:
        runtime = self.context.runtime
        logger = self.services.logger
        repair_result = self.services.repair_wikidata_region(
            self.context.data_root,
            plan,
            wikidata_client=runtime.wikidata,
            wikipedia_client=runtime.wikipedia,
            augmentation_client=self.context.augmentation_client,
            settings=self.context.settings,
            log=logger.info,
            scheduler_snapshot=runtime.scheduler.snapshot,
        )
        if not repair_result.changed:
            return self._recover_healthy(state, plan)
        self._mark_recovered(state.stem, repair_result.map_inputs_changed)
        self._migrate_links_if_needed(state.stem)
        return self._load_existing(state)

    def _prepare_publication(self, state: RegionSyncState, result: Any) -> None:
        if getattr(result, "wikipedia_documents_path", None) is None:
            return
        self.services.prepare_local_retirement(self.context.data_root, state.stem)
        self.services.add_pending_publications(self.context.data_root, {state.stem})
        classifications = self._recovery_classifications.get(state.stem)
        if classifications is not None:
            self.services.record_region_recovery_receipt(
                self.context.data_root,
                state.stem,
                classifications,
            )

    def _submit_upload(self, ops: list[PublicationOp], message: str) -> None:
        if self.context.upload_queue is not None:
            self.context.upload_queue.submit(ops, message)
            self._region_publication_submitted = True

    def _record_deferred_metadata(self, stem: str) -> None:
        """Persist the intent to refresh metadata once the queue drains.

        A regional commit that defers the repository-wide assets leaves
        them stale until the final refresh succeeds. The marker survives
        a crash or a failed refresh, so the next run repairs them instead
        of finding every path present and scheduling nothing.

        A surviving marker is merged rather than replaced: another run may
        have marked a region this one never reconciles, and dropping it
        here would let this run's regions license a repository-wide
        refresh describing the stranded one.

        A dry run publishes nothing, so it records nothing: durable
        publication intent must describe real uploads only.
        """
        if self.context.dry_run:
            return
        polygons_path = self.context.data_root.processed_polygons / f"{stem}.parquet"
        if not polygons_path.is_file():
            return
        self._deferred_metadata_stems[stem] = sha256_file(polygons_path)
        merged = {**self._recorded_marker_hashes(), **self._deferred_metadata_stems}
        self.services.set_metadata_refresh_marker(
            self.context.data_root,
            sorted(merged),
            merged,
        )

    def _recorded_marker_hashes(self) -> dict[str, str]:
        """Return the stems an existing refresh marker already names."""
        marker = self.services.load_metadata_refresh_marker(self.context.data_root)
        if marker is None:
            return {}
        return {str(stem): str(digest) for stem, digest in marker["fingerprint_hashes"].items()}

    def _build_region_publication(
        self,
        state: object,
        augmentation: object,
        core: object | None,
    ) -> list[PublicationOp]:
        stem = getattr(state, "stem", "")
        core = self._ensure_publication_core(stem, core)
        self._record_deferred_metadata(stem)
        return self.services.assemble_region_upload(
            data_root=self.context.data_root,
            repo_id=self.context.settings.repo_id,
            stem=stem,
            augmentation=augmentation,
            core=core,
            world_land_warning=None,
            refresh_maps=self._refresh_maps(state, stem),
            defer_metadata_assets=True,
        )

    def _needs_existing_core(self, stem: str) -> bool:
        if stem in self._recovered_stems:
            return True
        plan = self.context.reconciliation_plan
        if not self.context.push_enabled or plan is None:
            return False
        return (stem, "polygons") in plan.missing or (stem, "polygon_articles") in plan.missing

    def _ensure_publication_core(self, stem: str, core: object | None) -> object | None:
        if core is None and self._needs_existing_core(stem):
            self.services.logger.info(
                "Repairing remote region %s from finalized local artifacts (no Wikimedia requests)",
                stem,
            )
            core = self._load_publication_core(stem, core)
        if stem in self._recovered_stems and core is None:
            raise RuntimeError(f"Recovered region {stem!r} has no core publication artifacts")
        return core

    def _load_publication_core(self, stem: str, core: object | None) -> object | None:
        try:
            return self.services.load_existing_core_for_publication(
                self.context.data_root,
                stem,
                core,
                required=True,
            )
        except Exception as error:
            self.services.logger.error(
                "Failed to load local core artifacts for %s: %s", stem, error
            )
            raise

    def _refresh_maps(self, state: object, stem: str) -> bool:
        if getattr(state, "action", None) is not SyncAction.RECOVERY:
            return True
        return stem in self._recovery_map_refresh_stems

    def _close_uploads(self) -> list[str]:
        if self.context.upload_queue is None:
            return []
        return self.context.upload_queue.close_and_wait()


__all__ = [
    "SyncApplication",
    "SyncApplicationContext",
    "SyncApplicationResult",
    "SyncApplicationServices",
]
