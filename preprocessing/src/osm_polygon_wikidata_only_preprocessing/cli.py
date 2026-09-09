import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .deduplication.articles import DeduplicationError
from .deduplication.pipeline import deduplicate_region


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="osm-polygon-wikidata-only-preprocessing",
        description="Preprocess OSM polygons using Wikidata-linked features.",
    )
    commands = parser.add_subparsers(dest="command")
    deduplicate = commands.add_parser(
        "deduplicate-articles",
        help="Deduplicate article revisions while preserving entity links.",
    )
    deduplicate.add_argument("--raw-root", type=Path, required=True)
    deduplicate.add_argument("--output-root", type=Path, required=True)
    deduplicate.add_argument("--region-stem", required=True)

    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 0

    try:
        stats = deduplicate_region(
            arguments.raw_root,
            arguments.output_root,
            arguments.region_stem,
        )
    except (DeduplicationError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    duplicate_noun = (
        "duplicate" if stats.duplicate_articles_removed == 1 else "duplicates"
    )
    print(
        f"Articles: {stats.input_articles} -> {stats.output_articles} "
        f"({stats.duplicate_articles_removed} {duplicate_noun} removed)"
    )
    print(f"Entity mappings: {stats.entity_mappings}")
    print(f"Polygon links: {stats.input_links} -> {stats.output_links}")
    return 0
