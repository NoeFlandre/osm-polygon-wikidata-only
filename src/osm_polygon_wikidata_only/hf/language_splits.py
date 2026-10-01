"""Shared row-level language policy and validated V1/V2 inventories.

This module defines the contract consumed by the future partition generator.
It does not create partitions or publish to the Hugging Face Hub.  Language
values are read from textual/document rows; polygon-level preference fields
are intentionally outside this contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from osm_polygon_wikidata_only.augmentation.schema import (
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.domain.polygon_document_links import (
    polygon_document_link_schema,
)
from osm_polygon_wikidata_only.domain.schema import polygon_schema
from osm_polygon_wikidata_only.hf.language_artifact_manifest import (
    artifact_fingerprint,
    file_sha256,
    load_manifest_references,
    relative_path,
    require_manifest_references,
)
from osm_polygon_wikidata_only.hf.language_inventory_models import (
    DATASET_V1_ID,
    DATASET_V2_ID,
    LANGUAGE_SPLIT_CONTRACT_VERSION,
    LANGUAGE_SPLIT_PREFIX,
    UNKNOWN_LANGUAGE,
    DatasetContract,
    LanguageBucket,
    LanguageDisposition,
    LanguageInventory,
    LanguageInventoryError,
    LanguageResolution,
    LanguageTable,
    LanguageTableInventory,
    LanguageTableSpec,
    ValidatedArtifact,
    language_split_name,
    normalize_language,
)
from osm_polygon_wikidata_only.io.parquet_scan import iter_record_batches, open_parquet
from osm_polygon_wikidata_only.v2.schema import (
    polygon_document_link_v2_schema,
    polygon_v2_schema,
    wikipedia_document_v2_schema,
)

_BATCH_SIZE = 65_536


@dataclass(frozen=True, slots=True)
class _ArtifactSpec:
    table: str
    relative_dir: str
    schema_factory: Callable[[], pa.Schema]
    language_spec: LanguageTableSpec | None = None
    required: bool = False


def language_config_name(table: LanguageTable | str) -> str:
    """Return the additive configuration name for a language-bearing table."""
    table_value = _coerce_table(table).value
    return f"{table_value}_by_language"


def language_table_specs(dataset: DatasetContract | str) -> tuple[LanguageTableSpec, ...]:
    """Return the exact language-bearing table specs for one dataset."""
    return _language_specs_for(_coerce_dataset(dataset))


def build_language_inventory(
    processed_root: Path,
    dataset: DatasetContract | str = DatasetContract.V1,
    *,
    manifest_path: Path | None = None,
) -> LanguageInventory:
    """Validate artifacts and derive language counts from their row values.

    ``processed_root`` is the V1 ``processed/`` directory or the V2
    ``processed_v2/`` directory, never the repository checkout. Only the
    language column is read into batches; all other columns remain on disk.
    """
    contract = _coerce_dataset(dataset)
    root = Path(processed_root).resolve()
    if not root.is_dir():
        raise LanguageInventoryError(f"Processed artifact root is not a directory: {root}")
    manifest = (
        Path(manifest_path) if manifest_path is not None else root / "manifests/processed_pbfs.json"
    ).resolve()
    referenced = load_manifest_references(root, manifest, contract)

    tables, artifacts, source_paths = _collect_inventories(
        root,
        _artifact_specs_for(contract),
        referenced,
    )

    if not tables:
        raise LanguageInventoryError(f"No language-bearing artifacts found under {root}")
    fingerprint_paths = [manifest, *source_paths]
    return LanguageInventory(
        contract=contract,
        dataset_id=_dataset_id(contract),
        contract_version=LANGUAGE_SPLIT_CONTRACT_VERSION,
        source_manifest=relative_path(manifest, root),
        source_manifest_sha256=file_sha256(manifest),
        artifact_fingerprint=artifact_fingerprint(fingerprint_paths, root),
        tables=tuple(tables),
        validated_artifacts=tuple(artifacts),
    )


@dataclass(slots=True)
class _BucketCounter:
    row_count: int = 0
    canonical_rows: int = 0
    legacy_alias_rows: int = 0
    missing_rows: int = 0
    blank_rows: int = 0
    malformed_rows: int = 0
    legacy_unusable_rows: int = 0

    def observe(self, resolution: LanguageResolution, count: int = 1) -> None:
        """Record ``count`` rows that all resolved the same way."""
        self.row_count += count
        _increment_disposition(self, resolution.disposition, count)


_DISPOSITION_COUNTER_FIELDS: dict[LanguageDisposition, str] = {
    LanguageDisposition.CANONICAL: "canonical_rows",
    LanguageDisposition.LEGACY_ALIAS: "legacy_alias_rows",
    LanguageDisposition.MISSING: "missing_rows",
    LanguageDisposition.BLANK: "blank_rows",
    LanguageDisposition.MALFORMED: "malformed_rows",
    LanguageDisposition.LEGACY_UNUSABLE: "legacy_unusable_rows",
}


def _increment_disposition(
    counter: _BucketCounter,
    disposition: LanguageDisposition,
    count: int = 1,
) -> None:
    """Add ``count`` to the reason-specific field for one resolution."""
    field = _DISPOSITION_COUNTER_FIELDS[disposition]
    setattr(counter, field, getattr(counter, field) + count)


def _collect_inventories(
    root: Path,
    specs: Sequence[_ArtifactSpec],
    referenced: set[Path],
) -> tuple[list[LanguageTableInventory], list[ValidatedArtifact], list[Path]]:
    tables: list[LanguageTableInventory] = []
    artifacts: list[ValidatedArtifact] = []
    source_paths: list[Path] = []
    for spec in specs:
        inventory = _inventory_for_spec(root, spec, referenced)
        if inventory is None:
            continue
        artifact, table, paths = inventory
        artifacts.append(artifact)
        source_paths.extend(paths)
        if table is not None:
            tables.append(table)
    return tables, artifacts, source_paths


def _inventory_for_spec(
    root: Path,
    spec: _ArtifactSpec,
    referenced: set[Path],
) -> tuple[ValidatedArtifact, LanguageTableInventory | None, tuple[Path, ...]] | None:
    paths = _artifact_paths(root, spec)
    if not paths:
        if spec.required:
            raise LanguageInventoryError(
                f"Required {spec.table} artifacts are missing under {root / spec.relative_dir}"
            )
        return None
    if spec.required:
        require_manifest_references(paths, referenced, spec.table)

    table_files = tuple(relative_path(path, root) for path in paths)
    row_count, buckets = _read_artifact_rows(paths, spec)
    artifact = ValidatedArtifact(
        table=spec.table,
        source_files=table_files,
        row_count=row_count,
    )
    table = _build_language_table_inventory(spec, table_files, row_count, buckets)
    return artifact, table, paths


def _read_artifact_rows(
    paths: tuple[Path, ...],
    spec: _ArtifactSpec,
) -> tuple[int, dict[str, _BucketCounter]]:
    row_count = 0
    buckets: dict[str, _BucketCounter] = {}
    for path in paths:
        file_rows = _validate_schema_and_count(path, spec.schema_factory())
        row_count += file_rows
        if spec.language_spec is not None:
            _scan_language_file(path, spec.language_spec, buckets, file_rows)
    return row_count, buckets


def _build_language_table_inventory(
    spec: _ArtifactSpec,
    table_files: tuple[str, ...],
    row_count: int,
    buckets: dict[str, _BucketCounter],
) -> LanguageTableInventory | None:
    language_spec = spec.language_spec
    if language_spec is None:
        return None
    return LanguageTableInventory(
        table=language_spec.table,
        configuration=language_spec.configuration,
        language_column=language_spec.language_column,
        identity_columns=language_spec.identity_columns,
        source_files=table_files,
        row_count=row_count,
        buckets=_freeze_buckets(buckets),
    )


def _coerce_dataset(value: DatasetContract | str) -> DatasetContract:
    try:
        return DatasetContract(value)
    except ValueError as error:
        raise ValueError(f"Unknown dataset contract: {value!r}") from error


def _coerce_table(value: LanguageTable | str) -> LanguageTable:
    try:
        return LanguageTable(value)
    except ValueError as error:
        raise ValueError(f"Table is not language-bearing: {value!r}") from error


def _dataset_id(dataset: DatasetContract) -> str:
    return DATASET_V1_ID if dataset is DatasetContract.V1 else DATASET_V2_ID


def _language_specs_for(dataset: DatasetContract) -> tuple[LanguageTableSpec, ...]:
    if dataset is DatasetContract.V1:
        return (
            LanguageTableSpec(
                LanguageTable.POLYGON_ARTICLES,
                "polygon_articles",
                "language",
                ("polygon_id", "project", "document_id"),
                polygon_document_link_schema,
                language_config_name(LanguageTable.POLYGON_ARTICLES),
            ),
            LanguageTableSpec(
                LanguageTable.WIKIPEDIA_DOCUMENTS,
                "wikipedia/documents",
                "language",
                ("document_id",),
                wikipedia_document_schema,
                language_config_name(LanguageTable.WIKIPEDIA_DOCUMENTS),
            ),
            LanguageTableSpec(
                LanguageTable.WIKIPEDIA_SECTIONS,
                "wikipedia/sections",
                "language",
                ("section_id",),
                section_schema,
                language_config_name(LanguageTable.WIKIPEDIA_SECTIONS),
            ),
            LanguageTableSpec(
                LanguageTable.WIKIVOYAGE_DOCUMENTS,
                "wikivoyage/documents",
                "language",
                ("document_id",),
                document_schema,
                language_config_name(LanguageTable.WIKIVOYAGE_DOCUMENTS),
            ),
            LanguageTableSpec(
                LanguageTable.WIKIVOYAGE_SECTIONS,
                "wikivoyage/sections",
                "language",
                ("section_id",),
                section_schema,
                language_config_name(LanguageTable.WIKIVOYAGE_SECTIONS),
            ),
        )
    return (
        LanguageTableSpec(
            LanguageTable.POLYGON_DOCUMENT_LINKS,
            "polygon_document_links",
            "language",
            ("polygon_id", "project", "document_id"),
            polygon_document_link_v2_schema,
            language_config_name(LanguageTable.POLYGON_DOCUMENT_LINKS),
        ),
        LanguageTableSpec(
            LanguageTable.WIKIPEDIA_DOCUMENTS,
            "wikipedia/documents",
            "language",
            ("document_id",),
            wikipedia_document_v2_schema,
            language_config_name(LanguageTable.WIKIPEDIA_DOCUMENTS),
        ),
        LanguageTableSpec(
            LanguageTable.WIKIPEDIA_SECTIONS,
            "wikipedia/sections",
            "language",
            ("section_id",),
            section_schema,
            language_config_name(LanguageTable.WIKIPEDIA_SECTIONS),
        ),
    )


def _artifact_specs_for(dataset: DatasetContract) -> tuple[_ArtifactSpec, ...]:
    language_specs = _language_specs_for(dataset)
    if dataset is DatasetContract.V1:
        return (
            _ArtifactSpec("polygons", "polygons", polygon_schema, required=True),
            *(
                _ArtifactSpec(
                    spec.table.value,
                    spec.relative_dir,
                    spec.schema_factory,
                    spec,
                    spec.table in {LanguageTable.POLYGON_ARTICLES},
                )
                for spec in language_specs
            ),
            _ArtifactSpec("wikidata_facts", "wikidata/facts", fact_schema),
        )
    return (
        _ArtifactSpec("polygons", "polygons", polygon_v2_schema, required=True),
        *(
            _ArtifactSpec(
                spec.table.value, spec.relative_dir, spec.schema_factory, spec, required=True
            )
            for spec in language_specs
        ),
    )


def _artifact_paths(root: Path, spec: _ArtifactSpec) -> tuple[Path, ...]:
    directory = root / spec.relative_dir
    if not directory.is_dir():
        return ()
    return tuple(sorted(path.resolve() for path in directory.glob("*.parquet") if path.is_file()))


def _validate_schema_and_count(path: Path, expected: pa.Schema) -> int:
    try:
        with open_parquet(path) as parquet_file:
            actual = parquet_file.schema_arrow
            metadata = parquet_file.metadata
            row_count = 0 if metadata is None else metadata.num_rows
    except Exception as error:
        raise LanguageInventoryError(f"Could not read artifact {path}: {error}") from error
    if not actual.equals(expected, check_metadata=True):
        raise LanguageInventoryError(f"schema mismatch for artifact {path}")
    return row_count


def _scan_language_file(
    path: Path,
    spec: LanguageTableSpec,
    buckets: dict[str, _BucketCounter],
    expected_rows: int,
) -> None:
    observed_rows = 0
    try:
        with open_parquet(path) as parquet_file:
            for batch in iter_record_batches(
                parquet_file,
                columns=[spec.language_column],
                batch_size=_BATCH_SIZE,
            ):
                observed_rows += batch.num_rows
                _observe_language_batch(batch.column(0), buckets)
    except Exception as error:
        raise LanguageInventoryError(
            f"Could not scan language column {spec.language_column!r} in {path}: {error}"
        ) from error
    if observed_rows != expected_rows:
        raise LanguageInventoryError(
            f"row count changed while scanning {path}: metadata={expected_rows}, observed={observed_rows}"
        )


def partition_row_indices(batch: pa.RecordBatch, language_index: int) -> dict[str, pa.Array]:
    """Group a batch's row indices by normalized language partition.

    ``normalize_language`` depends only on the value, so it runs once per
    distinct value rather than once per row -- the same insight that
    :func:`_observe_language_batch` relies on. Rows are grouped with one stable
    sort, so the cost does not grow with the number of languages in the batch,
    and the index arrays can be handed straight to ``RecordBatch.take``.

    Indices within a partition are ascending and partitions come back in order
    of first occurrence, matching a row-by-row grouping exactly.
    """
    column = batch.column(language_index)
    if len(column) == 0:
        return {}
    names_by_code, codes = _partition_names_by_code(column)
    ids_by_name: dict[str, int] = {}
    id_of_code = [ids_by_name.setdefault(name, len(ids_by_name)) for name in names_by_code]
    partition_ids = _partition_ids(id_of_code, codes)
    grouped = _grouped_row_indices(partition_ids, ids_by_name)
    return dict(sorted(grouped.items(), key=lambda item: item[1][0].as_py()))


def _partition_names_by_code(column: pa.Array) -> tuple[list[str], pa.Array]:
    """Resolve each distinct language value once, keyed by dictionary code.

    Null values are folded in as one extra code so grouping never has to
    special-case them.
    """
    encoded = column.dictionary_encode()
    codes = encoded.indices
    names = [normalize_language(value).partition for value in encoded.dictionary.to_pylist()]
    if codes.null_count:
        names.append(normalize_language(None).partition)
        codes = pc.fill_null(codes, len(names) - 1)
    return names, codes


def _grouped_row_indices(
    partition_ids: pa.Array, ids_by_name: dict[str, int]
) -> dict[str, pa.Array]:
    """Group row indices by partition id using one stable sort.

    Sorting by partition id puts every partition's rows in one contiguous run
    while preserving their relative order, so each run is already the ascending
    index array that ``take`` needs.
    """
    order = pc.call_function("array_sort_indices", [partition_ids])
    counts = pc.call_function("value_counts", [partition_ids])
    name_by_id = {identifier: name for name, identifier in ids_by_name.items()}
    runs = sorted((entry["values"], entry["counts"]) for entry in counts.to_pylist())
    grouped: dict[str, pa.Array] = {}
    offset = 0
    for identifier, count in runs:
        grouped[name_by_id[identifier]] = order.slice(offset, count)
        offset += count
    return grouped


def _partition_ids(id_of_code: list[int], codes: pa.Array) -> pa.Array:
    """Map each row's dictionary code to its dense partition id.

    The ids are held as ``int32``: one value per row, and a batch never carries
    more partitions than an ``int32`` can address.
    """
    lookup = pa.array(id_of_code, type=pa.int32())
    ids: pa.Array = pc.call_function("take", [lookup, codes])
    return ids


def _observe_language_batch(column: pa.Array, buckets: dict[str, _BucketCounter]) -> None:
    """Count one batch by distinct language value rather than row by row.

    A batch holds tens of thousands of rows but only a handful of distinct
    language values, so the resolution runs once per distinct value and the
    row count is added in bulk. The result is identical to resolving every
    row, because ``normalize_language`` depends only on the value.
    """
    counts = pc.call_function("value_counts", [column])
    for value, count in zip(
        counts.field("values").to_pylist(), counts.field("counts").to_pylist(), strict=True
    ):
        resolution = normalize_language(value)
        buckets.setdefault(resolution.partition, _BucketCounter()).observe(resolution, count)


def _freeze_buckets(counters: dict[str, _BucketCounter]) -> tuple[LanguageBucket, ...]:
    def sort_key(language: str) -> tuple[bool, str]:
        return language == UNKNOWN_LANGUAGE, language

    return tuple(
        LanguageBucket(
            language=language,
            row_count=counter.row_count,
            canonical_rows=counter.canonical_rows,
            legacy_alias_rows=counter.legacy_alias_rows,
            missing_rows=counter.missing_rows,
            blank_rows=counter.blank_rows,
            malformed_rows=counter.malformed_rows,
            legacy_unusable_rows=counter.legacy_unusable_rows,
        )
        for language, counter in sorted(counters.items(), key=lambda item: sort_key(item[0]))
    )


__all__ = [
    "DATASET_V1_ID",
    "DATASET_V2_ID",
    "LANGUAGE_SPLIT_CONTRACT_VERSION",
    "LANGUAGE_SPLIT_PREFIX",
    "UNKNOWN_LANGUAGE",
    "DatasetContract",
    "LanguageBucket",
    "LanguageDisposition",
    "LanguageInventory",
    "LanguageInventoryError",
    "LanguageResolution",
    "LanguageTable",
    "LanguageTableInventory",
    "LanguageTableSpec",
    "ValidatedArtifact",
    "build_language_inventory",
    "language_config_name",
    "language_split_name",
    "language_table_specs",
    "normalize_language",
]
