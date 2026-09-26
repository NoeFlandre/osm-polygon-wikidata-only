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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

__all__ = ["__version__"]

try:
    # ``pyproject.toml`` is the single source of truth for the version.
    __version__ = _distribution_version("osm-polygon-wikidata-only")
except PackageNotFoundError:  # pragma: no cover - only without installed metadata
    __version__ = "0.0.0+unknown"
# Public spelling used by implementation modules that need the package version
# without importing the dunder attribute across module boundaries.
VERSION = __version__
