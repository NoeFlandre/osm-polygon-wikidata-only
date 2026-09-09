"""Evaluate a geographic NER pilot with deterministic silver signals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.io.atomic import atomic_write_json
from osm_polygon_wikidata_only.ner.pipeline import INPUT_COLUMNS, WIKINEURAL_LANGUAGES
from osm_polygon_wikidata_only.ner.silver_validation import audit_entities

Identity = tuple[str, str, str, str]
Polygon = tuple[str, str]
Document = tuple[str, str]


@dataclass(frozen=True)
class ReferenceIndex:
    names: dict[Document, tuple[str, ...]]
    polygons_by_document: dict[Document, frozenset[Polygon]]
    polygons: frozenset[Polygon]


def evaluate(
    data_root: Path, pilot_dir: Path, primary_output: Path, secondary_output: Path
) -> dict[str, Any]:
    """Return automatic quality signals for one primary/secondary pilot pair."""
    processed = _processed_root(Path(data_root))
    pilot_dir, primary_output, secondary_output = map(
        Path, (pilot_dir, primary_output, secondary_output)
    )
    source_rows = _read_rows(pilot_dir / "input.parquet")
    selection = _read_json(pilot_dir / "selection.json")
    references = _build_references(processed, selection.get("source_files"))
    source = {_identity(row) for row in source_rows}
    primary = _prediction_rows(primary_output, source)
    secondary = _prediction_rows(secondary_output, source)
    totals = Counter()
    by_language: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_project: defaultdict[str, Counter[str]] = defaultdict(Counter)
    entity_polygons: set[Polygon] = set()
    reference_rows = 0
    documents = set()
    for row in source_rows:
        key = _identity(row)
        document = (row["project"], row["document_id"])
        documents.add(document)
        names = references.names.get(document, ())
        if names:
            reference_rows += 1
        metrics = audit_entities(_entities(primary[key]), _entities(secondary[key]), names)
        _add_metrics(totals, metrics)
        _add_group_metrics(by_language[row["language"]], metrics)
        _add_group_metrics(by_project[row["project"]], metrics)
        if metrics["primary_entities"] > metrics["primary_artifacts"]:
            entity_polygons.update(references.polygons_by_document.get(document, ()))
    return {
        "status": "silver_unvalidated",
        "sample_size": len(source_rows),
        "unique_documents": len(documents),
        "unique_polygons": len(references.polygons),
        "reference_rows": reference_rows,
        "entity_bearing_polygons": len(entity_polygons),
        **dict(totals),
        "secondary_supported_languages": list(WIKINEURAL_LANGUAGES),
        "by_language": _sorted_groups(by_language),
        "by_project": _sorted_groups(by_project),
    }


def _processed_root(data_root: Path) -> Path:
    processed = data_root / "processed_v2"
    if processed.is_dir():
        return processed
    if data_root.name == "processed_v2" and data_root.is_dir():
        return data_root
    raise ValueError(f"processed_v2 directory is missing: {processed}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_rows(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing Parquet input: {path}")
    selected = columns or pq.ParquetFile(path).schema_arrow.names
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=list(selected)):
        rows.extend(batch.to_pylist())
    return rows


def _prediction_rows(path: Path, expected: set[Identity]) -> dict[Identity, dict[str, Any]]:
    rows: dict[Identity, dict[str, Any]] = {}
    for batch in sorted(path.glob("batch-*.parquet")):
        for row in _read_rows(batch):
            key = _identity(row)
            if key in rows:
                raise ValueError(f"Duplicate prediction identity: {key}")
            rows[key] = row
    if set(rows) != expected:
        raise ValueError(f"Prediction identities do not match pilot input: {path}")
    return rows


def _identity(row: Mapping[str, Any]) -> Identity:
    return (
        str(row[INPUT_COLUMNS[0]]),
        str(row[INPUT_COLUMNS[1]]),
        str(row[INPUT_COLUMNS[2]]),
        str(row[INPUT_COLUMNS[3]]),
    )


def _entities(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = row.get("entities")
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError("Prediction entities must be a list of objects")
    return cast(list[Mapping[str, Any]], value)


def _build_references(processed: Path, raw_files: object) -> ReferenceIndex:
    names: defaultdict[Document, set[str]] = defaultdict(set)
    polygons_by_document: defaultdict[Document, set[Polygon]] = defaultdict(set)
    polygons: set[Polygon] = set()
    for project, document_names, polygon_names, links in _reference_parts(processed, raw_files):
        for link in links:
            _merge_link_reference(
                project,
                link,
                document_names,
                polygon_names,
                names,
                polygons_by_document,
                polygons,
            )
    return ReferenceIndex(
        names={key: tuple(sorted(values)) for key, values in names.items()},
        polygons_by_document={key: frozenset(value) for key, value in polygons_by_document.items()},
        polygons=frozenset(polygons),
    )


def _reference_parts(
    processed: Path, raw_files: object
) -> Iterable[
    tuple[str, dict[str, tuple[str, ...]], dict[Polygon, tuple[str, ...]], list[dict[str, str]]]
]:
    if not isinstance(raw_files, list) or any(not isinstance(value, str) for value in raw_files):
        raise ValueError("Selection source_files must be a list of strings")
    for relative in cast(list[str], raw_files):
        project, stem = _source_identity(relative)
        document_names, polygon_names, links = _reference_part(processed, project, stem)
        yield project, document_names, polygon_names, links


def _reference_part(
    processed: Path, project: str, stem: str
) -> tuple[dict[str, tuple[str, ...]], dict[Polygon, tuple[str, ...]], list[dict[str, str]]]:
    document_path = processed / project / "documents" / f"{stem}.parquet"
    link_path = processed / "polygon_document_links" / f"{stem}.parquet"
    polygon_path = processed / "polygons" / f"{stem}.parquet"
    return _document_names(document_path), _polygon_names(polygon_path), _read_link_rows(link_path)


def _merge_link_reference(
    project: str,
    link: Mapping[str, str],
    document_names: Mapping[str, Sequence[str]],
    polygon_names: Mapping[Polygon, Sequence[str]],
    names: defaultdict[Document, set[str]],
    polygons_by_document: defaultdict[Document, set[Polygon]],
    polygons: set[Polygon],
) -> None:
    document = (project, link["document_id"])
    polygon = (link["osm_type"], link["osm_id"])
    names[document].update(document_names.get(document[1], ()))
    names[document].update(polygon_names.get(polygon, ()))
    polygons.add(polygon)
    polygons_by_document[document].add(polygon)


def _source_identity(relative: str) -> tuple[str, str]:
    path = Path(relative)
    if len(path.parts) != 3 or path.parts[1] != "sentences" or path.suffix != ".parquet":
        raise ValueError(f"Invalid selected sentence path: {relative}")
    return path.parts[0], path.stem


def _document_names(path: Path) -> dict[str, tuple[str, ...]]:
    columns = _document_columns(path)
    result: defaultdict[str, set[str]] = defaultdict(set)
    for row in _read_rows(path, sorted(columns)):
        if _successful_document(row):
            result[str(row["document_id"])].update(_document_values(row, columns))
    return {key: tuple(sorted(values)) for key, values in result.items()}


def _document_columns(path: Path) -> set[str]:
    columns = _available(
        path,
        ("document_id", "title", "wikidata_label", "wikidata_aliases", "fetch_status", "full_text"),
    )
    if not {"document_id", "fetch_status", "full_text"}.issubset(columns):
        raise ValueError(f"Document file is missing silver reference columns: {path}")
    return columns


def _successful_document(row: Mapping[str, Any]) -> bool:
    return row.get("fetch_status") == "ok" and bool(str(row.get("full_text") or "").strip())


def _document_values(row: Mapping[str, Any], columns: set[str]) -> tuple[str, ...]:
    values = [row.get(column) for column in ("title", "wikidata_label") if column in columns]
    aliases = _aliases(row.get("wikidata_aliases")) if "wikidata_aliases" in columns else ()
    return _clean_names((*values, *aliases))


def _clean_names(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(value for value in values if isinstance(value, str) and value.strip())


def _aliases(value: object) -> tuple[str, ...]:
    return _string_aliases(_decode_aliases(value) if isinstance(value, str) else value)


def _string_aliases(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return ()
    return tuple(cast(str, item) for item in value)


def _decode_aliases(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ()


def _read_link_rows(path: Path) -> list[dict[str, str]]:
    columns = _available(path, ("document_id", "osm_type", "osm_id"))
    required = {"document_id", "osm_type", "osm_id"}
    if not required.issubset(columns):
        raise ValueError(f"Link file is missing polygon identity columns: {path}")
    return [
        {
            "document_id": str(row["document_id"]),
            "osm_type": str(row["osm_type"]),
            "osm_id": str(row["osm_id"]),
        }
        for row in _read_rows(path, sorted(required))
    ]


def _polygon_names(path: Path) -> dict[Polygon, tuple[str, ...]]:
    columns = _available(path, ("osm_type", "osm_id", "name"))
    required = {"osm_type", "osm_id", "name"}
    if not required.issubset(columns):
        raise ValueError(f"Polygon file is missing silver reference columns: {path}")
    result: dict[Polygon, tuple[str, ...]] = {}
    for row in _read_rows(path, sorted(required)):
        name = row.get("name")
        if isinstance(name, str) and name.strip():
            result[(str(row["osm_type"]), str(row["osm_id"]))] = (name,)
    return result


def _available(path: Path, requested: Iterable[str]) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names) & set(requested)


def _add_metrics(target: Counter[str], values: Mapping[str, int]) -> None:
    target.update(values)


def _add_group_metrics(target: Counter[str], values: Mapping[str, int]) -> None:
    target["rows"] += 1
    target.update(values)


def _sorted_groups(groups: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(sorted(values.items())) for key, values in sorted(groups.items())}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--primary-output", type=Path, required=True)
    parser.add_argument("--secondary-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(args.data_root, args.pilot_dir, args.primary_output, args.secondary_output)
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
