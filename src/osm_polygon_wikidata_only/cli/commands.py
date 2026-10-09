"""CLI entry point.

Two commands:

- ``process-pbf <path>``: extract + enrich one PBF file.
- ``process-dir <path>``: process every ``*.pbf`` under a directory.

Shared options: ``--push``, ``--repo-id``, ``--data-root``,
``--skip-existing``, ``--force``, ``--languages``, ``--all-languages``,
``--no-full-text``, ``--max-articles-per-qid``, ``--limit``,
``--commit-message``, ``--log-level``.

This module owns argument parsing, runtime construction, authentication,
and command dispatch. Core processing publication is delegated to the
:mod:`osm_polygon_wikidata_only.hf.core_publication` service.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

from osm_polygon_wikidata_only.augmentation.mediawiki import AugmentationWikimediaClient
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    AugmentationResult,
    augment_region,
    augmentation_is_current,
    completed_region_stems,
)
from osm_polygon_wikidata_only.config.paths import DataRoot, DataRootError
from osm_polygon_wikidata_only.config.settings import DEFAULT_REPO_ID, Settings
from osm_polygon_wikidata_only.hf.core_publication import run_core_publication
from osm_polygon_wikidata_only.hf.push_authentication import authenticate_push_targets
from osm_polygon_wikidata_only.hf.uploader import StubHfHub
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.run_lock import RunLockError, exclusive_run_lock
from osm_polygon_wikidata_only.pipeline.orchestrator import orchestrate
from osm_polygon_wikidata_only.pipeline.processor import (
    ProcessResult,
)
from osm_polygon_wikidata_only.utils.logging import configure_logging
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID

from .dependencies import build_clients as _build_clients
from .dependencies import resolve_cli_data_root as _resolve_data_root
from .dispatch import parse_and_dispatch
from .errors import CliFailure, report_cli_error
from .parser import build_parser
from .parser import build_settings as _build_settings

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")
PROG = "osm-polygon-wikidata-only"


def _processing_inputs(command: str, input_path: Path) -> list[Path]:
    """Return the processing input list for a parsed process command."""
    del command
    return [input_path]


def _augmentation_stems(
    command: str,
    stem: str | None,
    data_root: DataRoot,
) -> list[str]:
    """Select the region stems addressed by an augmentation command."""
    if command == "augment-region":
        assert stem is not None
        return [stem]
    return completed_region_stems(data_root)


def _prepare_runtime(
    args: argparse.Namespace,
) -> tuple[DataRoot, Settings]:
    """Configure logging and construct the immutable runtime inputs."""
    configure_logging(level=getattr(logging, args.log_level))
    try:
        data_root = _resolve_data_root(args)
        data_root.ensure()
    except OSError as error:
        raise DataRootError(str(error)) from error
    settings = _build_settings(args)
    if getattr(args, "dataset_version", "v1") == "v2" and settings.repo_id == DEFAULT_REPO_ID:
        settings = replace(settings, repo_id=V2_REPO_ID)
    return data_root, settings


def _release_apply_requested(args: argparse.Namespace) -> bool:
    commands = {"release-stats", "publish-language-splits"}
    return getattr(args, "command", None) in commands and bool(getattr(args, "apply", False))


def _push_target_repo_ids(args: argparse.Namespace, settings: Settings) -> list[str]:
    """Return the repositories whose credentials a real push needs, if any."""
    if args.dry_run:
        return []
    if _release_apply_requested(args):
        return [_RELEASE_TARGETS[t] for t in _selected_release_targets(args.dataset_version)]
    return [settings.repo_id] if args.push else []


def _raise_cli_failure(message: str) -> NoReturn:
    """Report a credential-preflight rejection as an expected CLI failure."""
    raise CliFailure(message)


def _authenticate_for_push(args: argparse.Namespace, settings: Settings) -> None:
    """Validate Hugging Face credentials when a real push was requested."""
    repo_ids = _push_target_repo_ids(args, settings)
    if repo_ids:
        authenticate_push_targets(settings, repo_ids, _raise_cli_failure)


def _run_v2_sync(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the v2 sync while holding the shared lock."""
    # Keep the selected workflow lazy so the CLI can expose help without V2 dependencies.
    from osm_polygon_wikidata_only.v2.cli import execute_v2  # noqa: PLC0415

    try:
        with exclusive_run_lock(data_root.cache / "sync.lock"):
            return execute_v2(args, data_root=data_root, settings=settings)
    except RunLockError as error:
        return report_cli_error(parser.prog, error)


def _run_v2_sentence_split(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run V2 sentence sidecars while holding their dedicated lock."""
    # Sentence splitting has optional model dependencies and is loaded only when selected.
    from osm_polygon_wikidata_only.v2.cli import execute_v2_sentence_split  # noqa: PLC0415

    try:
        with exclusive_run_lock(data_root.cache / "sentence-splitting.lock"):
            return execute_v2_sentence_split(args, data_root=data_root, settings=settings)
    except RunLockError as error:
        return report_cli_error(parser.prog, error)


def _run_v1_sync(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the v1 sync while holding the shared lock."""
    # The V1 runner is loaded only after argument dispatch and lock selection.
    from .run_sync import execute as cli_run_sync  # noqa: PLC0415

    try:
        with exclusive_run_lock(data_root.cache / "sync.lock"):
            return cli_run_sync(
                args,
                data_root=data_root,
                settings=settings,
                # No build_upload_files override: the CLI shell owns the
                # production region-publication builder via
                # hf.publication.assemble_region_upload.
                build_upload_files=None,
            )
    except RunLockError as error:
        return report_cli_error(parser.prog, error)


def _run_sync_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Dispatch the selected sync dataset version."""
    if getattr(args, "dataset_version", "v1") == "v2":
        return _run_v2_sync(parser, args, data_root=data_root, settings=settings)
    return _run_v1_sync(parser, args, data_root=data_root, settings=settings)


def _load_augmentation_result(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    stem: str,
    augmentation_client: AugmentationWikimediaClient,
) -> AugmentationResult | None:
    """Load a current result or perform one region augmentation."""
    # These imports are command collaborators; keeping them local avoids importing
    # the full augmentation and migration stacks for unrelated CLI commands.
    from osm_polygon_wikidata_only.augmentation.orchestrator import (  # noqa: PLC0415
        load_existing_augmentation_result,
    )
    from osm_polygon_wikidata_only.pipeline.link_migration import (  # noqa: PLC0415
        apply_link_migration,
        plan_link_migration,
    )

    if args.skip_existing and augmentation_is_current(data_root, stem):
        migration = plan_link_migration(data_root.processed, stems={stem})
        if not migration.stems or migration.stems[0].classification.value == "canonical":
            LOGGER.info("Skipping augmentation for %s (already current)", stem)
            return None
        apply_link_migration(data_root.processed, plan=migration)
        LOGGER.info(
            "Migrated %s to unified polygon-document links without Wikimedia requests",
            stem,
        )
    else:
        augment_region(data_root, stem, augmentation_client)
        apply_link_migration(data_root.processed, stems={stem})
    return load_existing_augmentation_result(data_root, stem)


def _publish_augmentation(
    args: argparse.Namespace,
    settings: Settings,
    *,
    data_root: DataRoot,
    stem: str,
    result: AugmentationResult,
) -> None:
    """Publish one augmentation result when requested."""
    if not args.push:
        return
    from osm_polygon_wikidata_only.hf.augmentation_publication import (  # noqa: PLC0415
        publish_augmentation,
    )

    publish_augmentation(
        data_root=data_root,
        repo_id=settings.repo_id,
        token=settings.hf_token,
        dry_run=args.dry_run,
        commit_message=args.commit_message or f"Add text augmentation for {stem}",
        upload_threads=args.upload_threads,
        result=result,
    )


def _run_augmentation_command(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run augmentation for one region or all completed regions."""
    augmentation_client = AugmentationWikimediaClient(
        settings,
        JsonFileCache(data_root.cache / "augmentation", contract_version="text-sidecars-v1"),
    )
    stems = _augmentation_stems(args.command, getattr(args, "stem", None), data_root)
    augmentation_results: list[AugmentationResult] = []
    for stem in stems:
        result = _load_augmentation_result(
            args,
            data_root=data_root,
            stem=stem,
            augmentation_client=augmentation_client,
        )
        if result is None:
            continue
        augmentation_results.append(result)
        LOGGER.info("Augmented %s: %s", stem, result.counts)
        _publish_augmentation(args, settings, data_root=data_root, stem=stem, result=result)
    LOGGER.info("Done. %d region augmentation(s).", len(augmentation_results))
    return 0


def _log_process_results(results: Sequence[ProcessResult]) -> None:
    """Log the stable summary emitted after core processing completes."""
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


def _run_processing_command(
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the core PBF processor through the HF publication service."""
    wd, wiki, cache = _build_clients(settings, data_root=data_root)
    inputs = _processing_inputs(args.command, args.input)
    defer_metadata_assets = args.command == "process-dir"

    def process(on_complete: Callable[[ProcessResult], None]) -> list[ProcessResult]:
        return orchestrate(
            inputs,
            data_root=data_root,
            settings=settings,
            wikidata_client=wd,
            wikipedia_client=wiki,
            cache=cache,
            on_complete=on_complete,
        )

    outcome = run_core_publication(
        process,
        args,
        settings,
        data_root=data_root,
        defer_metadata_assets=defer_metadata_assets,
    )
    _log_process_results(outcome.results)
    if outcome.upload_failures:
        return report_cli_error(
            PROG,
            CliFailure(f"{len(outcome.upload_failures)} background upload(s) failed"),
        )
    return 0


_RELEASE_TARGETS: dict[str, str] = {"v1": DEFAULT_REPO_ID, "v2": V2_REPO_ID}


def _selected_release_targets(dataset_version: str) -> tuple[str, ...]:
    """Return the ordered contracts a release run publishes."""
    if dataset_version == "both":
        return ("v1", "v2")
    return (dataset_version,)


def _release_confirmations(
    args: argparse.Namespace,
    targets: tuple[str, ...],
) -> dict[str, str]:
    """Map each released contract to its required exact repository id."""
    supplied = list(getattr(args, "confirm_repo", None) or [])
    expected = [_RELEASE_TARGETS[target] for target in targets]
    if sorted(supplied) != sorted(expected):
        raise CliFailure(
            "release-stats requires one --confirm-repo per released dataset: " + ", ".join(expected)
        )
    return dict(zip(targets, expected, strict=True))


def _run_release_stats(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
) -> int:
    """Recompute and publish only the card and statistics report."""
    # Release-only dependencies are intentionally isolated from normal sync startup.
    from osm_polygon_wikidata_only.hf.stats_release import (  # noqa: PLC0415
        StatsReleaseError,
        release_v1_polygon_stats,
        release_v2_polygon_stats,
    )

    targets = _selected_release_targets(args.dataset_version)
    try:
        confirmations = _release_confirmations(args, targets)
    except CliFailure as failure:
        return report_cli_error(parser.prog, failure)
    releases = {"v1": release_v1_polygon_stats, "v2": release_v2_polygon_stats}
    hub = StubHfHub() if args.dry_run else None
    reports = []
    for target in targets:
        try:
            report = releases[target](
                data_root,
                confirm_repo=confirmations[target],
                generated_on=getattr(args, "generated_on", None),
                apply=args.apply,
                hub=hub,
                token=getattr(args, "hf_token", None),
                source_revision=getattr(args, "source_revision", None),
                data_revision=getattr(args, "data_revision", None),
            )
        except StatsReleaseError as error:
            return report_cli_error(parser.prog, error)
        reports.append(report)
        print(json.dumps(report.to_payload(), sort_keys=True))
    return 0


def _run_language_splits(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
) -> int:
    """Plan or generate both language-split contracts without publication."""
    # Language generation is an explicit command and may load large optional readers.
    from osm_polygon_wikidata_only.hf.language_split_release import (  # noqa: PLC0415
        LanguageSplitReleaseError,
        run_language_split_release,
    )

    try:
        result = run_language_split_release(
            data_root,
            dataset_version=args.dataset_version,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    except LanguageSplitReleaseError as error:
        return report_cli_error(parser.prog, error)
    print(result.to_json())
    return 0


def _run_publish_language_splits(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
) -> int:
    """Generate and publish exact-target language partitions."""
    # Keep publication and its network client out of non-publication CLI paths.
    from osm_polygon_wikidata_only.hf.language_split_publication import (  # noqa: PLC0415
        LanguagePublicationError,
        run_language_split_publication,
    )

    try:
        result = run_language_split_publication(
            data_root,
            dataset_version=args.dataset_version,
            batch_size=args.batch_size,
            confirm_repos=tuple(args.confirm_repo or ()),
            apply=args.apply,
            dry_run=args.dry_run,
            token=args.hf_token,
        )
    except LanguagePublicationError as error:
        return report_cli_error(parser.prog, error)
    print(result.to_json())
    return 0


def _dispatch_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    data_root: DataRoot,
    settings: Settings,
) -> int:
    """Run the selected command handler."""
    dispatch = {
        "language-splits": lambda: _run_language_splits(parser, args, data_root=data_root),
        "publish-language-splits": lambda: _run_publish_language_splits(
            parser, args, data_root=data_root
        ),
        "split-v2-sentences": lambda: _run_v2_sentence_split(
            parser, args, data_root=data_root, settings=settings
        ),
        "release-stats": lambda: _run_release_stats(parser, args, data_root=data_root),
        "sync-dir": lambda: _run_sync_command(parser, args, data_root=data_root, settings=settings),
        "augment-region": lambda: _run_augmentation_command(
            args, data_root=data_root, settings=settings
        ),
        "augment-dir": lambda: _run_augmentation_command(
            args, data_root=data_root, settings=settings
        ),
    }
    handler = dispatch.get(args.command)
    if handler is not None:
        return handler()
    return _run_processing_command(args, data_root=data_root, settings=settings)


def run_parsed(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run an already-parsed processing or augmentation command.

    Tool subcommands are dispatched by :func:`parse_and_dispatch`, never here.
    Expected failures are reported once on stderr with exit status 1.
    """
    try:
        data_root, settings = _prepare_runtime(args)
        _authenticate_for_push(args, settings)
        return _dispatch_command(parser, args, data_root=data_root, settings=settings)
    except (CliFailure, DataRootError) as failure:
        return report_cli_error(parser.prog, failure)


def main(argv: Sequence[str] | None = None) -> int:
    return parse_and_dispatch(build_parser(), argv, run_parsed)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]
