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

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    upload_files,
)
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.manifest import load_manifest
from osm_polygon_wikidata_only.pipeline.orchestrator import orchestrate
from osm_polygon_wikidata_only.utils.logging import configure_logging

from .dependencies import build_clients as _build_clients
from .dependencies import resolve_cli_data_root as _resolve_data_root
from .parser import build_parser
from .parser import build_settings as _build_settings

LOGGER = logging.getLogger("osm_polygon_wikidata_only.cli")


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

    def enqueue_upload(result: Any) -> None:
        if upload_queue is None:
            return
        snapshots = data_root.cache / "upload_manifest_snapshots"
        snapshot = snapshots / f"{result.polygons_path.stem}.json"
        atomic_write_text(snapshot, result.manifest_path.read_text(encoding="utf-8"))
        entries = load_manifest(result.manifest_path)
        aggregate = {
            key: sum(int(entry.get(key, 0)) for entry in entries.values())
            for key in ("polygon_count", "article_count", "unique_wikidata_count")
        }
        dataset_stats = compute_dataset_stats(data_root.processed)
        stats_section = render_stats_section(dataset_stats)
        card_snapshot = snapshots / f"{result.polygons_path.stem}-README.md"
        atomic_write_text(
            card_snapshot,
            render_dataset_card(
                repo_id=settings.repo_id,
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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]
