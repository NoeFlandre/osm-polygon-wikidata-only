"""CLI entry point.

Two commands:

- ``process-pbf <path>``: extract + enrich one PBF file.
- ``process-dir <path>``: process every ``*.pbf`` under a directory.

Shared options: ``--push``, ``--repo-id``, ``--data-root``,
``--skip-existing``, ``--force``, ``--languages``, ``--all-languages``,
``--no-full-text``, ``--max-articles-per-qid``, ``--limit``,
``--commit-message``, ``--log-level``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    AugmentationResult,
    augment_region,
    augmentation_is_current,
    completed_region_stems,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    ARTICLE_DESCRIPTIONS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_ARTICLE_DESCRIPTIONS,
    POLYGON_COLUMNS,
    POLYGON_DESCRIPTIONS,
)
from osm_polygon_wikidata_only.hf.coverage_map import (
    ensure_world_land,
    generate_coverage_map,
    load_centroids_from_parquet,
)
from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card
from osm_polygon_wikidata_only.hf.dataset_stats import (
    compute_dataset_stats,
    render_stats_section,
)
from osm_polygon_wikidata_only.hf.repo_layout import (
    REMOTE_ARTICLES_DIR,
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_LINKS_DIR,
    REMOTE_MANIFEST_FILE,
    REMOTE_POLYGONS_DIR,
)
from osm_polygon_wikidata_only.hf.upload_queue import BackgroundUploadQueue
from osm_polygon_wikidata_only.hf.uploader import (
    StubHfHub,
    UploadError,
    resolve_hf_token,
    upload_files,
    verify_hf_token,
    verify_repo_authorization,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.io.run_lock import RunLockError, exclusive_run_lock
from osm_polygon_wikidata_only.pipeline.orchestrator import collect_pbfs, orchestrate
from osm_polygon_wikidata_only.pipeline.processor import (
    ExtractedPbf,
    ProcessResult,
    extract_pbf,
    process_extracted_pbf,
)
from osm_polygon_wikidata_only.pipeline.sync_orchestrator import run_sync_plan
from osm_polygon_wikidata_only.pipeline.sync_planner import (
    RegionSyncState,
    SyncAction,
    plan_sync_states,
)
from osm_polygon_wikidata_only.utils.logging import configure_logging

from .dependencies import build_clients as _build_clients
from .dependencies import build_wikimedia_runtime as _build_wikimedia_runtime
from .dependencies import resolve_cli_data_root as _resolve_data_root
from .parser import build_parser
from .parser import build_settings as _build_settings

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


def _write_readme_snapshot(data_root: DataRoot, repo_id: str, destination: Path) -> None:
    """Render the canonical dataset README from current local artifacts."""
    entries = load_manifest(data_root.processed_manifests / "processed_pbfs.json")
    aggregate = {
        key: sum(int(entry.get(key, 0)) for entry in entries.values())
        for key in ("polygon_count", "article_count", "unique_wikidata_count")
    }
    stats_section = render_stats_section(compute_dataset_stats(data_root.processed))
    atomic_write_text(
        destination,
        render_dataset_card(
            repo_id=repo_id,
            stats=aggregate,
            polygon_columns=list(POLYGON_COLUMNS),
            polygon_descriptions=POLYGON_DESCRIPTIONS,
            article_columns=list(ARTICLE_COLUMNS),
            article_descriptions=ARTICLE_DESCRIPTIONS,
            link_columns=list(POLYGON_ARTICLE_COLUMNS),
            link_descriptions=POLYGON_ARTICLE_DESCRIPTIONS,
            maintainer="Noé Flandre",
            stats_section=stats_section,
        ),
    )


def _augmentation_upload_files(
    result: AugmentationResult, processed_root: Path, readme: Path
) -> list[tuple[Path, str]]:
    artifacts = (
        result.wikipedia_documents_path,
        result.wikipedia_sections_path,
        result.wikivoyage_documents_path,
        result.wikivoyage_sections_path,
        result.wikidata_facts_path,
        result.manifest_path,
    )
    return [
        *((path, str(path.relative_to(processed_root))) for path in artifacts),
        (readme, "README.md"),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level))
    if not args.data_root or not args.data_root.exists():
        # defer to resolver
        pass
    data_root = _resolve_data_root(args)
    data_root.ensure()
    settings = _build_settings(args)

    if args.push and not args.dry_run:
        resolved = resolve_hf_token(settings.hf_token)
        if not resolved:
            env_token = os.environ.get("HF_TOKEN")
            explicit = bool(settings.hf_token)
            if env_token or explicit:
                # Token was supplied but huggingface_hub rejected it (placeholder,
                # expired, revoked, malformed). Give a more actionable hint than
                # the generic "no token" message.
                source = "--hf-token" if explicit else "HF_TOKEN"
                parser.error(
                    f"--push: {source} is set but Hugging Face rejected it as invalid. "
                    "Generate a fresh write token at https://huggingface.co/settings/tokens "
                    "and replace the current value."
                )
            parser.error(
                "--push requires a Hugging Face write token: pass --hf-token, "
                "set HF_TOKEN, or run `huggingface-cli login`."
            )
        try:
            username = verify_hf_token(settings.hf_token)
        except UploadError as error:
            parser.error(str(error))
        try:
            verify_repo_authorization(settings.hf_token, settings.repo_id)
        except UploadError as error:
            parser.error(str(error))
        LOGGER.info("Authenticated to Hugging Face as %s (target: %s)", username, settings.repo_id)

    if args.command == "sync-dir":
        try:
            with exclusive_run_lock(data_root.cache / "sync.lock"):
                return _run_sync_dir(args, data_root, settings)
        except RunLockError as error:
            parser.error(str(error))

    if args.command in {"augment-region", "augment-dir"}:
        augmentation_client = AugmentationWikimediaClient(
            settings,
            JsonFileCache(data_root.cache / "augmentation", contract_version="text-sidecars-v1"),
        )
        stems = (
            [args.stem] if args.command == "augment-region" else completed_region_stems(data_root)
        )
        augmentation_results: list[AugmentationResult] = []
        for stem in stems:
            if settings.skip_existing and augmentation_is_current(data_root, stem):
                LOGGER.info("Skipping augmentation for %s (already current)", stem)
                continue
            augmentation_result = augment_region(data_root, stem, augmentation_client)
            augmentation_results.append(augmentation_result)
            LOGGER.info("Augmented %s: %s", stem, augmentation_result.counts)
            if args.push:
                snapshots = data_root.cache / "augmentation_upload_snapshots"
                readme_snapshot = snapshots / f"{stem}-README.md"
                _write_readme_snapshot(data_root, settings.repo_id, readme_snapshot)
                files = _augmentation_upload_files(
                    augmentation_result, data_root.processed, readme_snapshot
                )
                upload_files(
                    settings.repo_id,
                    files,
                    hub=StubHfHub() if args.dry_run else None,
                    token=settings.hf_token,
                    commit_message=args.commit_message or f"Add text augmentation for {stem}",
                    num_threads=args.upload_threads,
                )
        LOGGER.info("Done. %d region augmentation(s).", len(augmentation_results))
        return 0

    wd, wiki, cache = _build_clients(settings, data_root=data_root)

    inputs: list[Path]
    if args.command == "process-pbf" or args.command == "process-dir":
        inputs = [args.input]
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")
        return 2

    upload_queue: BackgroundUploadQueue | None = None
    hub = StubHfHub() if args.push and args.dry_run else None
    if args.push:

        def upload_job(files: list[tuple[Path, str]], message: str) -> None:
            upload_files(
                settings.repo_id,
                files,
                hub=hub,
                token=settings.hf_token,
                commit_message=message,
                num_threads=args.upload_threads,
            )

        upload_queue = BackgroundUploadQueue(
            upload=upload_job,
            max_pending=2,
            state_dir=data_root.cache / "upload_jobs",
        )
        resumed = upload_queue.resume_pending()
        if resumed:
            LOGGER.info("Resumed %d pending background upload(s)", resumed)

    def enqueue_upload(result: ProcessResult) -> None:
        if upload_queue is None:
            return
        snapshots = data_root.cache / "upload_manifest_snapshots"
        snapshot = snapshots / f"{result.polygons_path.stem}.json"
        atomic_write_text(snapshot, result.manifest_path.read_text(encoding="utf-8"))
        card_snapshot = snapshots / f"{result.polygons_path.stem}-README.md"
        _write_readme_snapshot(data_root, settings.repo_id, card_snapshot)
        map_snapshot = snapshots / f"{result.polygons_path.stem}-coverage_map.png"
        lons, lats = load_centroids_from_parquet(data_root.processed_polygons)
        try:
            land_path = ensure_world_land(data_root.cache)
        except Exception:
            LOGGER.warning("Could not fetch world land data; map will omit continents")
            land_path = None
        generate_coverage_map(lons, lats, map_snapshot, land_geojson_path=land_path)
        upload_queue.submit(
            [
                (result.polygons_path, f"{REMOTE_POLYGONS_DIR}/{result.polygons_path.name}"),
                (result.articles_path, f"{REMOTE_ARTICLES_DIR}/{result.articles_path.name}"),
                (
                    result.polygon_articles_path,
                    f"{REMOTE_LINKS_DIR}/{result.polygon_articles_path.name}",
                ),
                (snapshot, REMOTE_MANIFEST_FILE),
                (card_snapshot, "README.md"),
                (map_snapshot, REMOTE_COVERAGE_MAP_FILE),
            ],
            args.commit_message or f"Update PBF {result.manifest_entry['source_pbf']}",
        )

    upload_failures: list[str] = []
    try:
        results = orchestrate(
            inputs,
            data_root=data_root,
            settings=settings,
            wikidata_client=wd,
            wikipedia_client=wiki,
            cache=cache,
            on_complete=enqueue_upload,
        )
    finally:
        if upload_queue is not None:
            upload_failures = upload_queue.close_and_wait()
    LOGGER.info(
        "Done. %d PBF(s), %d polygons processed.",
        len(results),
        sum(r.polygon_count for r in results),
    )
    for result in results:
        LOGGER.info(
            "Stage timings for %s: %s",
            result.manifest_entry["source_pbf"],
            ", ".join(f"{name}={seconds:.3f}s" for name, seconds in result.stage_timings_s.items()),
        )
    if upload_failures:
        LOGGER.error("%d background upload(s) failed", len(upload_failures))
        return 1
    return 0


def _run_sync_dir(args: argparse.Namespace, data_root: DataRoot, settings: Settings) -> int:
    """Converge every raw PBF to the complete existing artifact contract."""
    runtime = _build_wikimedia_runtime(settings, data_root=data_root)
    augmentation_client = AugmentationWikimediaClient(
        runtime.settings,
        JsonFileCache(data_root.cache / "augmentation", contract_version="text-sidecars-v1"),
        scheduler=runtime.scheduler,
        session=runtime.session,
    )
    pbfs = collect_pbfs([args.input])
    entries = load_manifest(data_root.processed_manifests / "processed_pbfs.json")
    core_stems = {name.removesuffix(".osm.pbf") for name in entries}
    current_augmentation = {stem for stem in core_stems if augmentation_is_current(data_root, stem)}
    states = plan_sync_states(
        pbfs,
        core_stems=core_stems,
        augmentation_stems=current_augmentation,
        force=settings.force or not settings.skip_existing,
    )
    counts = {action: sum(state.action is action for state in states) for action in SyncAction}
    LOGGER.info(
        "Unified sync plan: %d augmentation backlog, %d core missing, %d complete",
        counts[SyncAction.AUGMENT],
        counts[SyncAction.PROCESS],
        counts[SyncAction.COMPLETE],
    )

    hub = StubHfHub() if args.push and args.dry_run else None

    def upload_job(files: list[tuple[Path, str]], message: str) -> None:
        upload_files(
            settings.repo_id,
            files,
            hub=hub,
            token=settings.hf_token,
            commit_message=message,
            num_threads=args.upload_threads,
        )

    upload_queue = (
        BackgroundUploadQueue(
            upload=upload_job,
            max_pending=2,
            state_dir=data_root.cache / "sync_upload_jobs",
        )
        if args.push
        else None
    )
    if upload_queue is not None:
        upload_queue.resume_pending()
    core_results: dict[str, ProcessResult] = {}
    process_states = [state for state in states if state.action is SyncAction.PROCESS]
    extraction_executor = ThreadPoolExecutor(max_workers=1)
    extraction_index = 0
    extraction_future: Future[ExtractedPbf] | None = None
    if process_states:
        extraction_future = extraction_executor.submit(
            extract_pbf, process_states[0].pbf_path, settings=settings
        )

    def process_state(state: RegionSyncState) -> None:
        nonlocal extraction_future, extraction_index
        if extraction_future is None or process_states[extraction_index].stem != state.stem:
            raise RuntimeError(f"Unexpected core processing order for {state.stem}")
        extracted = extraction_future.result()
        extraction_index += 1
        extraction_future = (
            extraction_executor.submit(
                extract_pbf,
                process_states[extraction_index].pbf_path,
                settings=settings,
            )
            if extraction_index < len(process_states)
            else None
        )
        process_settings = replace(settings, skip_existing=False)
        result = process_extracted_pbf(
            extracted,
            data_root=data_root,
            wikidata_client=runtime.wikidata,
            wikipedia_client=runtime.wikipedia,
            settings=process_settings,
            cache=runtime.cache,
        )
        core_results[state.stem] = result

    def augment_state(state: RegionSyncState) -> None:
        augmentation_result = augment_region(data_root, state.stem, augmentation_client)
        LOGGER.info("Unified sync completed %s: %s", state.stem, augmentation_result.counts)
        if upload_queue is not None:
            files = _sync_upload_files(
                data_root,
                settings.repo_id,
                state.stem,
                augmentation_result,
                core_results.get(state.stem),
            )
            upload_queue.submit(files, args.commit_message or f"Sync complete region {state.stem}")

    failures: list[str] = []
    try:
        completed = run_sync_plan(
            states,
            process_region=process_state,
            augment_region=augment_state,
        )
    finally:
        extraction_executor.shutdown(wait=True, cancel_futures=True)
        if upload_queue is not None:
            failures = upload_queue.close_and_wait()
    LOGGER.info("Unified sync completed %d region(s)", len(completed))
    return 1 if failures else 0


def _sync_upload_files(
    data_root: DataRoot,
    repo_id: str,
    stem: str,
    augmentation: AugmentationResult,
    core: ProcessResult | None,
) -> list[tuple[Path, str]]:
    """Snapshot manifests and assemble one complete atomic region upload."""
    snapshots = data_root.cache / "sync_upload_snapshots" / stem
    augmentation_manifest = snapshots / "augmentation_manifest.json"
    atomic_write_text(augmentation_manifest, augmentation.manifest_path.read_text())
    readme = snapshots / "README.md"
    _write_readme_snapshot(data_root, repo_id, readme)
    files = [
        (augmentation.wikipedia_documents_path, f"wikipedia/documents/{stem}.parquet"),
        (augmentation.wikipedia_sections_path, f"wikipedia/sections/{stem}.parquet"),
        (augmentation.wikivoyage_documents_path, f"wikivoyage/documents/{stem}.parquet"),
        (augmentation.wikivoyage_sections_path, f"wikivoyage/sections/{stem}.parquet"),
        (augmentation.wikidata_facts_path, f"wikidata/facts/{stem}.parquet"),
        (augmentation_manifest, "augmentation/manifests/augmentation_manifest.json"),
        (readme, "README.md"),
    ]
    if _coverage_refresh_required(core):
        core_manifest = snapshots / "processed_pbfs.json"
        atomic_write_text(
            core_manifest,
            (data_root.processed_manifests / "processed_pbfs.json").read_text(),
        )
        coverage = snapshots / "coverage_map.png"
        lons, lats = load_centroids_from_parquet(data_root.processed_polygons)
        try:
            land_path = ensure_world_land(data_root.cache)
        except Exception:
            land_path = None
        generate_coverage_map(lons, lats, coverage, land_geojson_path=land_path)
        assert core is not None
        files[:0] = [
            (core.polygons_path, f"{REMOTE_POLYGONS_DIR}/{core.polygons_path.name}"),
            (core.articles_path, f"{REMOTE_ARTICLES_DIR}/{core.articles_path.name}"),
            (core.polygon_articles_path, f"{REMOTE_LINKS_DIR}/{core.polygon_articles_path.name}"),
            (core_manifest, REMOTE_MANIFEST_FILE),
            (coverage, REMOTE_COVERAGE_MAP_FILE),
        ]
    return files


def _coverage_refresh_required(core: object | None) -> bool:
    """Coverage changes only when a core polygon artifact changes."""
    return core is not None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]
