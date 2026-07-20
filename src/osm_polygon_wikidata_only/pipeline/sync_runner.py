"""Pure-state sync execution: augment-backlog, publish-only repairs, process, complete.

This module is a pure state executor. It receives injectable
collaborators and delegates every collaboration boundary to the
caller.

The runner owns only:

* Per-state ordering: drain the AUGMENT backlog first (each call
  performs Wikimedia sidecar work and may enqueue an atomic
  remote publication on success); drain PUBLISH-only
  reconciliation repairs next (safe, Wikimedia-free uploads of
  finalized local artifacts that the remote is missing -- the
  repair uses the already-loaded local augmentation result and
  enqueues one Hugging Face upload without invoking any
  Wikidata, Wikipedia, or Wikivoyage call); then walk PROCESS
  states (prefetch the next PBF extraction before fully
  enriching the current one -- the one-PBF-ahead invariant).
  Within each bucket, stems are executed alphabetically in the
  order produced by the planner.
* Aggregation: collect PROCESS results in plan order so a
  subsequent PUBLISH can find a stem's local core artifact
  when the planner later classifies it for repair.
* Exception semantics: processing, augmentation and
  publish-load exceptions propagate through the same boundary
  as before; ``on_complete`` is not invoked on a failure path.

Nothing in this module imports from ``cli.*``, ``hf.*``,
``argparse``, :class:`DataRoot`, or :class:`Settings`. There
is no default production collaborator; every callable is
required and provided by the caller. The CLI shell lives in
:mod:`cli.run_sync` and constructs collaborators, then invokes
this runner.

Public identities (:class:`SyncAction`, :class:`RegionSyncState`,
:func:`run_sync_plan`) are re-exported here unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .sync_orchestrator import run_sync_plan
from .sync_planner import RegionSyncState, SyncAction

__all__ = [
    "RegionSyncState",
    "SyncAction",
    "run_sync",
    "run_sync_plan",
]


def run_sync(
    states: list[RegionSyncState],
    *,
    extract_pbf: Callable[[Path], Any],
    process_extracted_pbf: Callable[[Any], Any],
    augment_region: Callable[[RegionSyncState], Any],
    build_upload_files: Callable[..., list[Any]] | None = None,
    commit_message: Callable[[RegionSyncState], str] | None = None,
    submit_upload: Callable[[list[Any], str], None] | None = None,
    close_uploads: Callable[[], list[str]] | None = None,
    on_complete: Callable[[RegionSyncState, Any], None] | None = None,
    load_existing_augmentation: Callable[[RegionSyncState], Any] | None = None,
    recover_region: Callable[[RegionSyncState], Any] | None = None,
) -> int:
    """Execute the unified sync plan as a pure state executor.

    Required collaborators (all injected, no defaults):

    * ``extract_pbf(pbf_path)``: synchronously returns the
      ``ExtractedPbf`` for a single PBF. The CLI shell pre-binds
      ``settings`` (and any other parameters) into this callable.
    * ``process_extracted_pbf(extracted)``: synchronously enriches
      and persists one ``ExtractedPbf``. The CLI shell pre-binds
      ``data_root``, ``wikidata_client``, ``wikipedia_client`` and
      ``settings`` (with ``skip_existing=False``) into this
      callable.
    * ``augment_region(state)``: synchronously augments one
      region. The CLI shell pre-binds ``data_root``,
      ``augmentation_client``, heartbeat, and logger.

    Optional collaborators (default ``None``):

    * ``build_upload_files(state, augmentation, core)``: returns a
      list of ``PublicationOp`` records (one atomic unit of work
      per op, see ``osm_polygon_wikidata_only.hf._uploader.plan``)
      to commit as one atomic upload. ``None`` means no
      publication assembly.
    * ``commit_message(state)``: returns the per-region commit
      message. Defaults to ``f"Sync complete region {state.stem}"``.
    * ``submit_upload(ops, message)``: enqueues one atomic
      commit. ``None`` means no upload submission.
    * ``close_uploads()``: returns a list of failed-job names from
      the upload queue. ``None`` means no queue is open.
    * ``on_complete(state, result)``: invoked once per successful
      augmentation step (PROCESS or AUGMENT).
    * ``load_existing_augmentation(state)``: loads the local
      augmentation result for a PUBLISH-only repair without
      invoking extraction or Wikimedia. Required only when the
      plan contains PUBLISH states.
    * ``recover_region(state)``: audit one finalized region and,
      when necessary, perform surgical Wikidata integrity recovery
      without invoking PBF extraction or unrelated sidecar work.
      Required only when the plan contains RECOVERY states. A
      ``None`` result means the region is healthy and needs no
      publication; a repair result is published before continuing.

    Execution sequence:

    1. Drain RECOVERY states in alphabetical order, one region at a
       time. Each state performs its QID-level audit without PBF
       extraction. Healthy regions store a resumable receipt and
       continue; affected regions are repaired transactionally and
       published immediately before the next region is audited.
    2. Start prefetching the first PROCESS PBF (background thread).
    3. Drain AUGMENT (backlog) states in alphabetical order.
    4. Drain PUBLISH-only reconciliation repairs in alphabetical
       order. Each repair uses ``load_existing_augmentation`` (no
       extraction, no Wikidata, no Wikipedia, no Wikivoyage
       call). Publication is enqueued before PROCESSING for the
       next region begins.
    5. For each PROCESS state in alphabetical order: await
       extraction, immediately prefetch the next PBF, enrich/
       persist the current region, augment it.
    6. After each successful augmentation, if both
       ``build_upload_files`` and ``submit_upload`` are provided,
       assemble and submit one atomic publication.
    """
    if extract_pbf is None or process_extracted_pbf is None or augment_region is None:
        raise RuntimeError(
            "run_sync requires extract_pbf, process_extracted_pbf, and augment_region collaborators"
        )

    process_states = [state for state in states if state.action is SyncAction.PROCESS]
    augment_states = [state for state in states if state.action is SyncAction.AUGMENT]
    publish_states = [state for state in states if state.action is SyncAction.PUBLISH]
    recovery_states = [state for state in states if state.action is SyncAction.RECOVERY]

    extraction_executor = ThreadPoolExecutor(max_workers=1)
    core_results: dict[str, Any] = {}
    failures: list[str] = []
    try:
        # Recovery must finish before extraction begins: recovery may
        # replace finalized core tables that publication snapshots consume.
        extraction_future: Future[Any] | None = None

        # Step 1: audit RECOVERY states one region at a time. Healthy
        # candidates return None. A repaired region is published
        # before the next candidate is audited. PBF extraction starts
        # only after this resumable recovery queue is drained.
        for state in recovery_states:
            if recover_region is None:
                raise RuntimeError(
                    "run_sync requires recover_region collaborator for RECOVERY states"
                )
            recovery_result = recover_region(state)
            if recovery_result is None:
                continue
            if on_complete is not None:
                on_complete(state, recovery_result)
            _maybe_submit(
                state=state,
                augmentation=recovery_result,
                core=core_results.get(state.stem),
                submit_upload=submit_upload,
                build_upload_files=build_upload_files,
                commit_message=commit_message,
            )

        # Step 2: once recovery has converged, restore the established
        # one-PBF-ahead overlap with AUGMENT/PUBLISH work.
        if process_states:
            extraction_future = extraction_executor.submit(extract_pbf, process_states[0].pbf_path)

        # Step 3: drain AUGMENT (backlog) states. Any exception
        # propagates after the executor is shut down in finally.
        for state in augment_states:
            augmentation = augment_region(state)
            if on_complete is not None:
                on_complete(state, augmentation)
            _maybe_submit(
                state=state,
                augmentation=augmentation,
                core=core_results.get(state.stem),
                submit_upload=submit_upload,
                build_upload_files=build_upload_files,
                commit_message=commit_message,
            )

        # Step 3: drain PUBLISH-only reconciliation repairs. These
        # are safe, Wikimedia-free uploads of finalized local
        # artifacts that the remote is missing. Each repair loads the
        # existing augmentation result (no extraction, no Wikidata /
        # Wikipedia / Wikivoyage call) and enqueues one atomic
        # Hugging Face commit before PROCESS begins.
        for state in publish_states:
            if load_existing_augmentation is None:
                raise RuntimeError(
                    "run_sync requires load_existing_augmentation collaborator for PUBLISH states"
                )
            augmentation = load_existing_augmentation(state)
            if on_complete is not None:
                on_complete(state, augmentation)
            _maybe_submit(
                state=state,
                augmentation=augmentation,
                core=core_results.get(state.stem),
                submit_upload=submit_upload,
                build_upload_files=build_upload_files,
                commit_message=commit_message,
            )

        # Step 4+5: walk PROCESS states. For each, await extraction,
        # immediately schedule the next extraction (one-PBF-ahead),
        # then enrich/persist, then augment that same region. Failures
        # in extraction or processing propagate; subsequent PROCESS
        # states are not entered.
        for process_index, state in enumerate(process_states):
            if extraction_future is None:
                raise RuntimeError(f"Missing prefetched extraction for PROCESS state {state.stem}")
            extracted = extraction_future.result()
            next_state = (
                process_states[process_index + 1]
                if process_index + 1 < len(process_states)
                else None
            )
            extraction_future = (
                extraction_executor.submit(extract_pbf, next_state.pbf_path)
                if next_state is not None
                else None
            )
            result = process_extracted_pbf(extracted)
            core_results[state.stem] = result
            augmentation = augment_region(state)
            if on_complete is not None:
                on_complete(state, augmentation)
            _maybe_submit(
                state=state,
                augmentation=augmentation,
                core=result,
                submit_upload=submit_upload,
                build_upload_files=build_upload_files,
                commit_message=commit_message,
            )
    finally:
        extraction_executor.shutdown(wait=True, cancel_futures=True)
        if close_uploads is not None:
            failures.extend(close_uploads())
    if failures:
        return 1
    return 0


def _maybe_submit(
    *,
    state: RegionSyncState,
    augmentation: Any,
    core: Any | None,
    submit_upload: Callable[[list[Any], str], None] | None,
    build_upload_files: Callable[[RegionSyncState, Any, Any | None], list[Any]] | None,
    commit_message: Callable[[RegionSyncState], str] | None,
) -> None:
    if submit_upload is None or build_upload_files is None:
        return
    ops = build_upload_files(state, augmentation, core)
    if not ops:
        return
    message = (
        commit_message(state)
        if commit_message is not None
        else f"Sync complete region {state.stem}"
    )
    submit_upload(ops, message)
