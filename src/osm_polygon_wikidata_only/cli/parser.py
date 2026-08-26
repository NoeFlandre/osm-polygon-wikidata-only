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
        description=(
            "Build a Hugging Face dataset of OSM polygons with Wikidata, Wikipedia, and Wikivoyage."
        ),
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
        "--hf-token",
        default=None,
        help=(
            "Hugging Face write token. Defaults to the HF_TOKEN env var "
            "or the saved `huggingface-cli login` token."
        ),
    )
    common.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    common.add_argument("--dry-run", action="store_true", help="Use a stub HF client (no network)")
    sentence_common = argparse.ArgumentParser(add_help=False)
    sentence_common.add_argument("--data-root", type=Path, default=None, help="Data root directory")
    sentence_common.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id (org/name)"
    )
    sentence_common.add_argument("--batch-size", type=int, default=256)
    sentence_common.add_argument("--inference-batch-size", type=int, default=16)
    sentence_common.add_argument(
        "--push", action="store_true", help="Push artifacts to Hugging Face"
    )
    sentence_common.add_argument("--commit-message", default=None)
    sentence_common.add_argument("--upload-threads", type=int, default=5)
    sentence_common.add_argument("--hf-token", default=None)
    sentence_common.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    sentence_common.add_argument(
        "--dry-run", action="store_true", help="Use a stub HF client (no network)"
    )
    p_pbf = sub.add_parser("process-pbf", parents=[common], help="Process one PBF file")
    p_pbf.add_argument("input", type=Path, help="Path to a .osm.pbf file")
    p_dir = sub.add_parser("process-dir", parents=[common], help="Process every PBF in a directory")
    p_dir.add_argument("input", type=Path, help="Directory containing *.osm.pbf files")
    p_sync = sub.add_parser(
        "sync-dir", parents=[common], help="Converge core and augmentation for every PBF"
    )
    p_sync.add_argument("input", type=Path, help="Directory containing *.osm.pbf files")
    p_sync.add_argument(
        "--dataset-version",
        choices=("v1", "v2"),
        default="v1",
        help="Select the isolated dataset contract (default: v1)",
    )
    p_sentence = sub.add_parser(
        "split-v2-sentences",
        parents=[sentence_common],
        help="Materialize resumable V2 sentence sidecars",
    )
    p_sentence.set_defaults(dataset_version="v2")
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
    all_languages = getattr(args, "all_languages", False)
    language_value = getattr(args, "languages", None)
    languages = None if all_languages or language_value is None else parse_languages(language_value)
    return Settings(
        repo_id=getattr(args, "repo_id", DEFAULT_REPO_ID),
        user_agent=getattr(args, "user_agent", DEFAULT_USER_AGENT),
        languages=languages,
        fetch_full_text=not getattr(args, "no_full_text", False),
        max_articles_per_qid=getattr(args, "max_articles_per_qid", None),
        enrichment_batch_size=getattr(args, "enrichment_batch_size", 50),
        enrichment_site_workers=getattr(args, "enrichment_site_workers", 8),
        cache_ttl_s=86_400,
        skip_existing=getattr(args, "skip_existing", False),
        force=getattr(args, "force", False),
        limit=getattr(args, "limit", None),
        hf_token=getattr(args, "hf_token", None),
    )


__all__ = ["build_parser", "build_settings", "parse_languages"]
