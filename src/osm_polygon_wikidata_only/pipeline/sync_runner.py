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


def _partition_states(
    states: list[RegionSyncState],
) -> tuple[
    list[RegionSyncState],
    list[RegionSyncState],
    list[RegionSyncState],
    list[RegionSyncState],
]:
    """Partition states by action while preserving their input order."""
    buckets: dict[SyncAction, list[RegionSyncState]] = {
        SyncAction.PROCESS: [],
        SyncAction.AUGMENT: [],
        SyncAction.PUBLISH: [],
        SyncAction.RECOVERY: [],
    }
    for state in states:
        bucket = buckets.get(state.action)
        if bucket is not None:
            bucket.append(state)
    return (
        buckets[SyncAction.PROCESS],
        buckets[SyncAction.AUGMENT],
        buckets[SyncAction.PUBLISH],
        buckets[SyncAction.RECOVERY],
    )


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
    _validate_required_collaborators(extract_pbf, process_extracted_pbf, augment_region)

    process_states, augment_states, publish_states, recovery_states = _partition_states(states)

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
        _run_recovery_phase(
            recovery_states,
            recover_region=recover_region,
            core_results=core_results,
            on_complete=on_complete,
            submit_upload=submit_upload,
            build_upload_files=build_upload_files,
            commit_message=commit_message,
        )

        # Step 2: once recovery has converged, restore the established
        # one-PBF-ahead overlap with AUGMENT/PUBLISH work.
        extraction_future = _start_extraction_prefetch(
            process_states,
            extraction_executor=extraction_executor,
            extract_pbf=extract_pbf,
        )

        # Step 3: drain AUGMENT (backlog) states. Any exception
        # propagates after the executor is shut down in finally.
        _run_augment_phase(
            augment_states,
            augment_region=augment_region,
            core_results=core_results,
            on_complete=on_complete,
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
        _run_publish_phase(
            publish_states,
            load_existing_augmentation=load_existing_augmentation,
            core_results=core_results,
            on_complete=on_complete,
            submit_upload=submit_upload,
            build_upload_files=build_upload_files,
            commit_message=commit_message,
        )

        # Step 4+5: walk PROCESS states. For each, await extraction,
        # immediately schedule the next extraction (one-PBF-ahead),
        # then enrich/persist, then augment that same region. Failures
        # in extraction or processing propagate; subsequent PROCESS
        # states are not entered.
        _run_process_phase(
            process_states,
            extraction_future=extraction_future,
            extraction_executor=extraction_executor,
            extract_pbf=extract_pbf,
            process_extracted_pbf=process_extracted_pbf,
            augment_region=augment_region,
            core_results=core_results,
            on_complete=on_complete,
            submit_upload=submit_upload,
            build_upload_files=build_upload_files,
            commit_message=commit_message,
        )
    finally:
        extraction_executor.shutdown(wait=True, cancel_futures=True)
        if close_uploads is not None:
            failures.extend(close_uploads())
    return int(bool(failures))


def _validate_required_collaborators(
    extract_pbf: Callable[[Path], Any] | None,
    process_extracted_pbf: Callable[[Any], Any] | None,
    augment_region: Callable[[RegionSyncState], Any] | None,
) -> None:
    """Validate the callbacks needed by every execution plan."""
    if extract_pbf is None or process_extracted_pbf is None or augment_region is None:
        raise RuntimeError(
            "run_sync requires extract_pbf, process_extracted_pbf, and augment_region collaborators"
        )


def _start_extraction_prefetch(
    states: list[RegionSyncState],
    *,
    extraction_executor: ThreadPoolExecutor,
    extract_pbf: Callable[[Path], Any],
) -> Future[Any] | None:
    """Start the first PROCESS extraction, if any state needs it."""
    if not states:
        return None
    return extraction_executor.submit(extract_pbf, states[0].pbf_path)


def _complete_state(
    state: RegionSyncState,
    result: Any,
    *,
    core: Any | None,
    on_complete: Callable[[RegionSyncState, Any], None] | None,
    submit_upload: Callable[[list[Any], str], None] | None,
    build_upload_files: Callable[[RegionSyncState, Any, Any | None], list[Any]] | None,
    commit_message: Callable[[RegionSyncState], str] | None,
) -> None:
    if on_complete is not None:
        on_complete(state, result)
    _maybe_submit(
        state=state,
        augmentation=result,
        core=core,
        submit_upload=submit_upload,
        build_upload_files=build_upload_files,
        commit_message=commit_message,
    )


def _run_recovery_phase(
    states: list[RegionSyncState],
    *,
    recover_region: Callable[[RegionSyncState], Any] | None,
    core_results: dict[str, Any],
    on_complete: Callable[[RegionSyncState, Any], None] | None,
    submit_upload: Callable[[list[Any], str], None] | None,
    build_upload_files: Callable[[RegionSyncState, Any, Any | None], list[Any]] | None,
    commit_message: Callable[[RegionSyncState], str] | None,
) -> None:
    """Drain recovery states before any PBF extraction starts."""
    if not states:
        return
    if recover_region is None:
        raise RuntimeError("run_sync requires recover_region collaborator for RECOVERY states")
    for state in states:
        result = recover_region(state)
        if result is not None:
            _complete_state(
                state,
                result,
                core=core_results.get(state.stem),
                on_complete=on_complete,
                submit_upload=submit_upload,
                build_upload_files=build_upload_files,
                commit_message=commit_message,
            )


def _run_augment_phase(
    states: list[RegionSyncState],
    *,
    augment_region: Callable[[RegionSyncState], Any],
    core_results: dict[str, Any],
    on_complete: Callable[[RegionSyncState, Any], None] | None,
    submit_upload: Callable[[list[Any], str], None] | None,
    build_upload_files: Callable[[RegionSyncState, Any, Any | None], list[Any]] | None,
    commit_message: Callable[[RegionSyncState], str] | None,
) -> None:
    """Drain the existing augmentation backlog."""
    for state in states:
        _complete_state(
            state,
            augment_region(state),
            core=core_results.get(state.stem),
            on_complete=on_complete,
            submit_upload=submit_upload,
            build_upload_files=build_upload_files,
            commit_message=commit_message,
        )


def _run_publish_phase(
    states: list[RegionSyncState],
    *,
    load_existing_augmentation: Callable[[RegionSyncState], Any] | None,
    core_results: dict[str, Any],
    on_complete: Callable[[RegionSyncState, Any], None] | None,
    submit_upload: Callable[[list[Any], str], None] | None,
    build_upload_files: Callable[[RegionSyncState, Any, Any | None], list[Any]] | None,
    commit_message: Callable[[RegionSyncState], str] | None,
) -> None:
    """Drain publish-only repairs without extraction or Wikimedia calls."""
    if not states:
        return
    if load_existing_augmentation is None:
        raise RuntimeError(
            "run_sync requires load_existing_augmentation collaborator for PUBLISH states"
        )
    for state in states:
        _complete_state(
            state,
            load_existing_augmentation(state),
            core=core_results.get(state.stem),
            on_complete=on_complete,
            submit_upload=submit_upload,
            build_upload_files=build_upload_files,
            commit_message=commit_message,
        )


def _run_process_phase(
    states: list[RegionSyncState],
    *,
    extraction_future: Future[Any] | None,
    extraction_executor: ThreadPoolExecutor,
    extract_pbf: Callable[[Path], Any],
    process_extracted_pbf: Callable[[Any], Any],
    augment_region: Callable[[RegionSyncState], Any],
    core_results: dict[str, Any],
    on_complete: Callable[[RegionSyncState, Any], None] | None,
    submit_upload: Callable[[list[Any], str], None] | None,
    build_upload_files: Callable[[RegionSyncState, Any, Any | None], list[Any]] | None,
    commit_message: Callable[[RegionSyncState], str] | None,
) -> None:
    """Process cores with one PBF extraction prefetched ahead."""
    for index, state in enumerate(states):
        if extraction_future is None:
            raise RuntimeError(f"Missing prefetched extraction for PROCESS state {state.stem}")
        extracted = extraction_future.result()
        if index + 1 < len(states):
            extraction_future = extraction_executor.submit(extract_pbf, states[index + 1].pbf_path)
        else:
            extraction_future = None
        result = process_extracted_pbf(extracted)
        core_results[state.stem] = result
        _complete_state(
            state,
            augment_region(state),
            core=result,
            on_complete=on_complete,
            submit_upload=submit_upload,
            build_upload_files=build_upload_files,
            commit_message=commit_message,
        )


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
