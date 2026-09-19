"""Sentence-sidecar metrics for V2 cards."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc

from osm_polygon_wikidata_only.hf._geographic.polygon_identities import (
    PolygonIdentity,
    PolygonIndex,
)
from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.card_models import SentenceCardStats as _SentenceCardStats
from osm_polygon_wikidata_only.v2.card_scanning import load_polygon_index as _load_polygon_index


def _compute_sentence_stats(processed_v2: Path) -> _SentenceCardStats | None:
    sentence_paths = {
        project: sorted((processed_v2 / project / "sentences").glob("*.parquet"))
        for project in ("wikipedia", "wikivoyage")
    }
    manifest_path = processed_v2 / "manifests" / "sentence_splitting.json"
    if not manifest_path.is_file() or not any(sentence_paths.values()):
        return None

    regions, supported_languages = _load_sentence_manifest(manifest_path)
    total_rows, unsupported_rows = _sentence_manifest_totals(regions, manifest_path)
    document_ids = {
        project: _sentence_document_ids(paths) for project, paths in sentence_paths.items()
    }
    return _SentenceCardStats(
        total_rows=total_rows,
        split_rows=total_rows - unsupported_rows,
        unsupported_rows=unsupported_rows,
        polygon_count=_sentence_polygon_count(processed_v2, document_ids),
        supported_language_count=len(supported_languages),
        wikipedia_sidecars=len(sentence_paths["wikipedia"]),
        wikivoyage_sidecars=len(sentence_paths["wikivoyage"]),
    )

def _load_sentence_manifest(manifest_path: Path) -> tuple[list[object], list[object]]:
    raw_manifest = json_loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, Mapping):
        raise ValueError(f"Invalid sentence manifest: {manifest_path}")
    regions = raw_manifest.get("regions")
    supported_languages = raw_manifest.get("supported_languages")
    if not isinstance(regions, list) or not isinstance(supported_languages, list):
        raise ValueError(f"Invalid sentence manifest: {manifest_path}")
    return cast(list[object], regions), cast(list[object], supported_languages)

def _sentence_manifest_totals(regions: Iterable[object], manifest_path: Path) -> tuple[int, int]:
    total_rows = 0
    unsupported_rows = 0
    for region in regions:
        region_total_rows, region_unsupported_rows = _sentence_region_totals(region, manifest_path)
        total_rows += region_total_rows
        unsupported_rows += region_unsupported_rows
    if unsupported_rows > total_rows:
        raise ValueError(f"Invalid sentence row totals: {manifest_path}")
    return total_rows, unsupported_rows

def _sentence_region_totals(region: object, manifest_path: Path) -> tuple[int, int]:
    if not isinstance(region, Mapping):
        raise ValueError(f"Invalid sentence manifest region: {manifest_path}")
    values = cast(Mapping[str, object], region)
    # Keep the historical permissive int coercion for legacy JSON values.
    return int(cast(Any, values.get("sentence_rows", 0))), int(
        cast(Any, values.get("unsplit_sections", 0))
    )

def _sentence_document_ids(paths: Iterable[Path]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        _update_sentence_document_ids(path, values)
    return values

def _update_sentence_document_ids(path: Path, values: set[str]) -> None:
    with open_parquet(path) as parquet_file:
        if "document_id" not in parquet_file.schema_arrow.names:
            return
        for batch in iter_record_batches(parquet_file, columns=["document_id"], batch_size=65_536):
            values.update(_sentence_document_ids_batch(batch))

def _sentence_document_ids_batch(batch: pa.RecordBatch) -> set[str]:
    return {str(value) for value in _compute_array("unique", batch.column(0)).to_pylist() if value}

def _sentence_polygon_count(
    processed_v2: Path,
    document_ids: Mapping[str, set[str]],
) -> int:
    value_sets = {
        project: pa.array(sorted(values), type=pa.string())
        for project, values in document_ids.items()
        if values
    }
    if not value_sets:
        return 0

    polygon_index = _load_polygon_index(sorted((processed_v2 / "polygons").glob("*.parquet")))
    polygon_ids: set[PolygonIdentity] = set()
    for path in sorted((processed_v2 / "polygon_document_links").glob("*.parquet")):
        _collect_sentence_polygon_ids(path, value_sets, polygon_index, polygon_ids)
    return len(polygon_ids)

def _collect_sentence_polygon_ids(
    path: Path,
    value_sets: Mapping[str, pa.Array],
    polygon_index: PolygonIndex,
    polygon_ids: set[PolygonIdentity],
) -> None:
    with open_parquet(path) as parquet_file:
        columns = {"polygon_id", "document_id", "project"}
        if not columns.issubset(parquet_file.schema_arrow.names):
            return
        for batch in iter_record_batches(
            parquet_file,
            columns=["polygon_id", "document_id", "project"],
            batch_size=65_536,
        ):
            polygon_ids.update(_sentence_polygon_ids_from_batch(batch, value_sets, polygon_index))

def _sentence_polygon_ids_from_batch(
    batch: pa.RecordBatch,
    value_sets: Mapping[str, pa.Array],
    polygon_index: PolygonIndex,
) -> set[PolygonIdentity]:
    polygon_column, document_column, project_column = batch.columns
    polygon_ids: set[PolygonIdentity] = set()
    for project, value_set in value_sets.items():
        matches = _compute_array(
            "and",
            _compute_array("equal", project_column, project),
            _compute_array(
                "is_in",
                document_column,
                options=pc.SetLookupOptions(value_set),
            ),
        )
        polygon_ids.update(
            polygon_identity
            for value in _compute_array("filter", polygon_column, matches).to_pylist()
            if value
            and (polygon_identity := polygon_index.by_polygon_id.get(str(value))) is not None
        )
    return polygon_ids

def _compute_array(function: str, *arguments: Any, options: Any = None) -> Any:
    return pc.call_function(function, list(arguments), options=options)


# Public collaborator spellings keep sentence metrics independent of the card
# compatibility facade.
compute_sentence_stats = _compute_sentence_stats
load_sentence_manifest = _load_sentence_manifest
sentence_manifest_totals = _sentence_manifest_totals
sentence_region_totals = _sentence_region_totals
sentence_document_ids = _sentence_document_ids
update_sentence_document_ids = _update_sentence_document_ids
sentence_document_ids_batch = _sentence_document_ids_batch
sentence_polygon_count = _sentence_polygon_count
collect_sentence_polygon_ids = _collect_sentence_polygon_ids
sentence_polygon_ids_from_batch = _sentence_polygon_ids_from_batch
compute_array = _compute_array
