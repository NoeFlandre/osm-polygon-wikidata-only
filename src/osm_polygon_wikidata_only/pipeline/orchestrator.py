"""Multi-PBF orchestrator.

Iterates over a directory of PBFs (or a list of paths) and calls
:func:`processor.process_pbf` for each one. Honors ``skip_existing``
and ``force`` flags.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.enrichment.wikidata_client import WikidataClient
from osm_polygon_wikidata_only.enrichment.wikipedia_client import WikipediaClient
from osm_polygon_wikidata_only.io.cache import JsonFileCache
from osm_polygon_wikidata_only.io.manifest import load_manifest

from .processor import ExtractedPbf, ProcessResult, extract_pbf, process_extracted_pbf

LOGGER = logging.getLogger(__name__)


def collect_pbfs(inputs: Iterable[Path]) -> list[Path]:
    """Expand a list of file/directory paths into concrete PBF files."""
    out: list[Path] = []
    for p in inputs:
        if p.is_dir():
            out.extend(sorted(x for x in p.iterdir() if x.suffix == ".pbf"))
        elif p.is_file():
            out.append(p)
    return out


def already_processed(manifest_path: Path, source_pbf: str) -> bool:
    entries = load_manifest(manifest_path)
    return source_pbf in entries


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

    selected: list[Path] = []
    processed_entries = (
        load_manifest(data_root.processed_manifests / "processed_pbfs.json")
        if settings.skip_existing and not settings.force
        else {}
    )
    for pbf in pbfs:
        if pbf.name in processed_entries:
            LOGGER.info("Skipping %s (already processed, --skip-existing)", pbf.name)
            continue
        selected.append(pbf)
    if not selected:
        return []

    results: list[ProcessResult] = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pbf-extraction") as executor:
        current = executor.submit(extract_pbf, selected[0], settings=settings)
        for index in range(len(selected)):
            extracted: ExtractedPbf = current.result()
            if index + 1 < len(selected):
                current = executor.submit(extract_pbf, selected[index + 1], settings=settings)
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


__all__ = ["already_processed", "collect_pbfs", "orchestrate"]
