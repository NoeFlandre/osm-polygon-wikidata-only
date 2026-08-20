"""Multi-PBF orchestrator.

Iterates over a directory of PBFs (or a list of paths) and calls
:func:`processor.process_pbf` for each one. Honors ``skip_existing``
and ``force`` flags.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata_client import WikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia_client import WikipediaClient
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.manifest import load_manifest

from .processor import ProcessResult, extract_pbf, process_extracted_pbf

LOGGER = logging.getLogger(__name__)


def _collect_input(input_path: Path) -> list[Path]:
    """Expand one file or directory input."""
    if input_path.is_dir():
        return sorted(path for path in input_path.iterdir() if path.suffix == ".pbf")
    if input_path.is_file():
        return [input_path]
    return []


def collect_pbfs(inputs: Iterable[Path]) -> list[Path]:
    """Expand a list of file/directory paths into concrete PBF files."""
    out: list[Path] = []
    for p in inputs:
        out.extend(_collect_input(p))
    return out


def already_processed(manifest_path: Path, source_pbf: str) -> bool:
    entries = load_manifest(manifest_path)
    return source_pbf in entries


def _select_unprocessed(
    pbfs: list[Path],
    processed_entries: dict[str, dict[str, object]],
) -> list[Path]:
    """Filter already processed PBFs while preserving input order."""
    selected: list[Path] = []
    for pbf in pbfs:
        if pbf.name in processed_entries:
            LOGGER.info("Skipping %s (already processed, --skip-existing)", pbf.name)
            continue
        selected.append(pbf)
    return selected


def _process_selected(
    selected: list[Path],
    *,
    data_root: DataRoot,
    settings: Settings,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    cache: JsonFileCache | None,
    on_complete: Callable[[ProcessResult], None] | None,
) -> list[ProcessResult]:
    """Run extraction and enrichment for selected PBFs."""
    results: list[ProcessResult] = []
    for pbf in selected:
        extracted = extract_pbf(pbf, settings=settings)
        result = process_extracted_pbf(
            extracted,
            data_root=data_root,
            wikidata_client=wikidata_client,
            wikipedia_client=wikipedia_client,
            settings=settings,
            cache=cache,
        )
        results.append(result)
        if on_complete is not None:
            on_complete(result)
    return results


def orchestrate(
    inputs: Iterable[Path],
    *,
    data_root: DataRoot,
    settings: Settings,
    wikidata_client: WikidataClient,
    wikipedia_client: WikipediaClient,
    cache: JsonFileCache | None = None,
    on_complete: Callable[[ProcessResult], None] | None = None,
) -> list[ProcessResult]:
    """Process every input PBF, honoring ``skip_existing`` and ``force``."""
    pbfs = collect_pbfs(inputs)
    if not pbfs:
        LOGGER.warning("No PBF inputs to process")
        return []
    LOGGER.info("Orchestrating over %d PBF(s)", len(pbfs))

    processed_entries = (
        load_manifest(data_root.processed_manifests / "processed_pbfs.json")
        if settings.skip_existing and not settings.force
        else {}
    )
    selected = _select_unprocessed(pbfs, processed_entries)
    if not selected:
        return []
    return _process_selected(
        selected,
        data_root=data_root,
        settings=settings,
        wikidata_client=wikidata_client,
        wikipedia_client=wikipedia_client,
        cache=cache,
        on_complete=on_complete,
    )


__all__ = ["already_processed", "collect_pbfs", "orchestrate"]
