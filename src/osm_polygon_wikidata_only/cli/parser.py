"""Argument parsing and conversion to immutable runtime settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from osm_polygon_wikidata_only.config.settings import (
    DEFAULT_REPO_ID,
    DEFAULT_USER_AGENT,
    Settings,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the stable processing and augmentation CLI parser."""
    parser = argparse.ArgumentParser(
        prog="osm-polygon-wikidata-only",
        description="Build a Hugging Face dataset of OSM polygons linked to Wikidata + Wikipedia.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, default=None, help="Data root directory")
    common.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id (org/name)"
    )
    common.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="Wikipedia/Wikidata UA")
    common.add_argument("--languages", default=None, help="Optional comma-separated language codes")
    common.add_argument(
        "--all-languages", action="store_true", help="Fetch all available sitelinks"
    )
    common.add_argument(
        "--no-full-text", action="store_true", help="Skip Wikipedia full-text fetch"
    )
    common.add_argument("--max-articles-per-qid", type=int, default=None)
    common.add_argument("--enrichment-batch-size", type=int, default=50)
    common.add_argument("--enrichment-site-workers", type=int, default=8)
    common.add_argument("--limit", type=int, default=None, help="Cap number of polygons per PBF")
    common.add_argument("--skip-existing", action="store_true")
    common.add_argument("--force", action="store_true")
    common.add_argument("--push", action="store_true", help="Push artifacts to Hugging Face")
    common.add_argument("--commit-message", default=None)
    common.add_argument(
        "--upload-threads",
        type=int,
        default=5,
        help="Concurrent Hugging Face upload workers per atomic commit",
    )
    common.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    common.add_argument("--dry-run", action="store_true", help="Use a stub HF client (no network)")
    p_pbf = sub.add_parser("process-pbf", parents=[common], help="Process one PBF file")
    p_pbf.add_argument("input", type=Path, help="Path to a .osm.pbf file")
    p_dir = sub.add_parser("process-dir", parents=[common], help="Process every PBF in a directory")
    p_dir.add_argument("input", type=Path, help="Directory containing *.osm.pbf files")
    p_augment = sub.add_parser(
        "augment-region", parents=[common], help="Augment one completed region without reading PBF"
    )
    p_augment.add_argument("stem", help="Completed region stem, e.g. andorra-latest")
    sub.add_parser("augment-dir", parents=[common], help="Augment every completed core region")
    return parser


def parse_languages(value: str) -> tuple[str, ...]:
    """Normalize an explicit comma-separated language allow-list."""
    return tuple(sorted({item.strip() for item in value.split(",") if item.strip()}))


def build_settings(args: argparse.Namespace) -> Settings:
    """Convert parsed CLI arguments into immutable pipeline settings."""
    languages = (
        None if args.all_languages or args.languages is None else parse_languages(args.languages)
    )
    return Settings(
        repo_id=args.repo_id,
        user_agent=args.user_agent,
        languages=languages,
        fetch_full_text=not args.no_full_text,
        max_articles_per_qid=args.max_articles_per_qid,
        enrichment_batch_size=args.enrichment_batch_size,
        enrichment_site_workers=args.enrichment_site_workers,
        cache_ttl_s=86_400,
        skip_existing=args.skip_existing,
        force=args.force,
        limit=args.limit,
    )


__all__ = ["build_parser", "build_settings", "parse_languages"]
