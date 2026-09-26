"""osm-polygon-wikidata-only: polygon-only PBF → multi-table dataset pipeline.

Subpackages:

* :mod:`cli` — command-line interface (thin layer).
* :mod:`config` — paths and runtime settings.
* :mod:`domain` — pure domain models, schema, geometry, filters, analysis.
* :mod:`enrichment` — Wikidata + Wikipedia clients, parsers, linkers.
* :mod:`hf` — Hugging Face dataset card, uploader, repo layout.
* :mod:`io` — PBF reader, parquet writer, manifest, cache.
* :mod:`pipeline` — extractor, processor, orchestrator, stats.
* :mod:`utils` — small utilities (JSON, time, logging, retry).
"""

from __future__ import annotations

import sys

__all__ = ["__version__", "main"]

__version__ = "0.1.0"
# Public spelling used by implementation modules that need the package version
# without importing the dunder attribute across module boundaries.
VERSION = __version__


def main() -> int:
    """Handle the console version flag before importing the CLI package."""
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        sys.stdout.write(f"osm-polygon-wikidata-only {__version__}\n")
        return 0

    from .cli.app import run  # noqa: PLC0415

    return run()
