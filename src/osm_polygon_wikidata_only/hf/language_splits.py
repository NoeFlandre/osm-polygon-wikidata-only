"""Shared row-level language policy and validated V1/V2 inventories.

This module defines the contract consumed by the future partition generator.
It does not create partitions or publish to the Hugging Face Hub.  Language
values are read from textual/document rows; polygon-level preference fields
are intentionally outside this contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

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
from osm_polygon_wikidata_only.v2.config import V2_CONTRACT_VERSION
from osm_polygon_wikidata_only.v2.schema import (
    polygon_document_link_v2_schema,
    polygon_v2_schema,
    wikipedia_document_v2_schema,
)

DATASET_V1_ID = "NoeFlandre/osm-polygon-wikidata-only"
DATASET_V2_ID = "NoeFlandre/osm-polygon-wikidata-and-wikipedia"
LANGUAGE_SPLIT_CONTRACT_VERSION = "language-splits-v1"
LANGUAGE_SPLIT_PREFIX = "lang-"
UNKNOWN_LANGUAGE = "unknown"

_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$", re.IGNORECASE)
_LEGACY_ALIASES: dict[str, str] = {"be_x_old": "be-tarask"}
# These are observed V1 project labels, not language inventory members.  They
# are outside the repository's structural language rule and remain addressable
# through the explicit unknown partition.
_LEGACY_UNUSABLE_VALUES = frozenset({"abstract", "simple"})
_BATCH_SIZE = 65_536


class DatasetContract(StrEnum):
    """The two published dataset contracts covered by this shared policy."""

    V1 = "v1"
    V2 = "v2"


class LanguageTable(StrEnum):
    """Logical tables that have a row-level language column."""

    POLYGON_ARTICLES = "polygon_articles"
    POLYGON_DOCUMENT_LINKS = "polygon_document_links"
    WIKIPEDIA_DOCUMENTS = "wikipedia_documents"
    WIKIPEDIA_SECTIONS = "wikipedia_sections"
    WIKIVOYAGE_DOCUMENTS = "wikivoyage_documents"
    WIKIVOYAGE_SECTIONS = "wikivoyage_sections"


class LanguageDisposition(StrEnum):
    """Reason a raw value was assigned to its language partition."""

    CANONICAL = "canonical"
    LEGACY_ALIAS = "legacy_alias"
    MISSING = "missing"
    BLANK = "blank"
    MALFORMED = "malformed"
    LEGACY_UNUSABLE = "legacy_unusable"


@dataclass(frozen=True, slots=True)
class LanguageResolution:
    """Normalized partition key and classification for one raw value."""

    canonical: str | None
    partition: str
    disposition: LanguageDisposition


@dataclass(frozen=True, slots=True)
class LanguageTableSpec:
    """Schema and path contract for one language-bearing table."""

    table: LanguageTable
    relative_dir: str
    language_column: str
    identity_columns: tuple[str, ...]
    schema_factory: Callable[[], pa.Schema]
    configuration: str


@dataclass(frozen=True, slots=True)
class LanguageBucket:
    """Per-partition counts for one language-bearing table."""

    language: str
    row_count: int
    canonical_rows: int
    legacy_alias_rows: int
    missing_rows: int
    blank_rows: int
    malformed_rows: int
    legacy_unusable_rows: int

    @property
    def split(self) -> str:
        """Return the Hugging Face split name for this bucket."""
        return language_split_name(self.language)

    def to_dict(self) -> dict[str, object]:
        """Serialize this bucket with stable field names and values."""
        return {
            "language": self.language,
            "split": self.split,
            "row_count": self.row_count,
            "canonical_rows": self.canonical_rows,
            "legacy_alias_rows": self.legacy_alias_rows,
            "missing_rows": self.missing_rows,
            "blank_rows": self.blank_rows,
            "malformed_rows": self.malformed_rows,
            "legacy_unusable_rows": self.legacy_unusable_rows,
        }


@dataclass(frozen=True, slots=True)
class LanguageTableInventory:
    """Validated row inventory for one language-bearing table."""

    table: LanguageTable
    configuration: str
    language_column: str
    identity_columns: tuple[str, ...]
    source_files: tuple[str, ...]
    row_count: int
    buckets: tuple[LanguageBucket, ...]

    def bucket(self, language: str) -> LanguageBucket:
        """Return the bucket for a canonical language or ``unknown``."""
        partition = (
            UNKNOWN_LANGUAGE
            if language == UNKNOWN_LANGUAGE
            else normalize_language(language).partition
        )
        for bucket in self.buckets:
            if bucket.language == partition:
                return bucket
        raise KeyError(f"No {self.table.value} bucket for {language!r}")

    def to_dict(self) -> dict[str, object]:
        """Serialize this table inventory deterministically."""
        return {
            "table": self.table.value,
            "configuration": self.configuration,
            "language_column": self.language_column,
            "identity_columns": list(self.identity_columns),
            "source_files": list(self.source_files),
            "row_count": self.row_count,
            "buckets": [bucket.to_dict() for bucket in self.buckets],
        }


@dataclass(frozen=True, slots=True)
class ValidatedArtifact:
    """A schema-validated artifact, including language-neutral tables."""

    table: str
    source_files: tuple[str, ...]
    row_count: int

    def to_dict(self) -> dict[str, object]:
        """Serialize this validation record deterministically."""
        return {
            "table": self.table,
            "source_files": list(self.source_files),
            "row_count": self.row_count,
        }


@dataclass(frozen=True, slots=True)
class LanguageInventory:
    """Validated, runtime-derived inventory for exactly one dataset contract."""

    contract: DatasetContract
    dataset_id: str
    contract_version: str
    source_manifest: str
    source_manifest_sha256: str
    artifact_fingerprint: str
    tables: tuple[LanguageTableInventory, ...]
    validated_artifacts: tuple[ValidatedArtifact, ...]

    @property
    def languages(self) -> tuple[str, ...]:
        """Return all discovered non-unknown languages in sorted order."""
        return tuple(
            sorted(
                {
                    bucket.language
                    for table in self.tables
                    for bucket in table.buckets
                    if bucket.language != UNKNOWN_LANGUAGE
                }
            )
        )

    def table(
        self,
        table: LanguageTable | str,
        *,
        allow_non_language: bool = False,
    ) -> LanguageTableInventory | ValidatedArtifact:
        """Return a language table, or a neutral artifact when requested."""
        table_name = table.value if isinstance(table, LanguageTable) else table
        for inventory in self.tables:
            if inventory.table.value == table_name:
                return inventory
        if allow_non_language:
            for artifact in self.validated_artifacts:
                if artifact.table == table_name:
                    return artifact
        raise KeyError(f"No inventory for table {table_name!r}")

    def to_dict(self) -> dict[str, object]:
        """Serialize the inventory as a deterministic JSON-compatible object."""
        return {
            "contract": self.contract.value,
            "dataset_id": self.dataset_id,
            "contract_version": self.contract_version,
            "source_manifest": self.source_manifest,
            "source_manifest_sha256": self.source_manifest_sha256,
            "artifact_fingerprint": self.artifact_fingerprint,
            "languages": list(self.languages),
            "tables": [table.to_dict() for table in self.tables],
            "validated_artifacts": [artifact.to_dict() for artifact in self.validated_artifacts],
        }


class LanguageInventoryError(ValueError):
    """Raised when source artifacts cannot be validated for inventory."""


@dataclass(frozen=True, slots=True)
class _ArtifactSpec:
    table: str
    relative_dir: str
    schema_factory: Callable[[], pa.Schema]
    language_spec: LanguageTableSpec | None = None
    required: bool = False


def normalize_language(value: object) -> LanguageResolution:
    """Normalize one row language using the existing repository rule.

    The result always has a partition. Values outside the rule are assigned to
    ``unknown`` with a reason so callers can conserve every source row without
    inventing a language.
    """
    if value is None:
        return _unknown(LanguageDisposition.MISSING)
    if not isinstance(value, str):
        return _unknown(LanguageDisposition.MALFORMED)
    stripped = value.strip()
    if not stripped:
        return _unknown(LanguageDisposition.BLANK)

    lowered = stripped.lower()
    legacy = _LEGACY_ALIASES.get(lowered)
    if legacy is not None:
        return LanguageResolution(legacy, legacy, LanguageDisposition.LEGACY_ALIAS)
    candidate = lowered.replace("_", "-")
    if candidate in _LEGACY_UNUSABLE_VALUES:
        return _unknown(LanguageDisposition.LEGACY_UNUSABLE)
    if _LANGUAGE_RE.fullmatch(candidate) is None:
        return _unknown(LanguageDisposition.MALFORMED)
    return LanguageResolution(candidate, candidate, LanguageDisposition.CANONICAL)


def language_split_name(value: object) -> str:
    """Return the safe Hugging Face split name for one raw language value."""
    return f"{LANGUAGE_SPLIT_PREFIX}{normalize_language(value).partition}"


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
    referenced = _load_and_validate_manifest(root, manifest, contract)

    tables: list[LanguageTableInventory] = []
    artifacts: list[ValidatedArtifact] = []
    source_paths: list[Path] = []
    for spec in _artifact_specs_for(contract):
        paths = _artifact_paths(root, spec)
        if not paths:
            if spec.required:
                raise LanguageInventoryError(
                    f"Required {spec.table} artifacts are missing under {root / spec.relative_dir}"
                )
            continue
        if spec.required:
            _require_manifest_references(paths, referenced, spec.table)

        table_files = tuple(_relative_path(path, root) for path in paths)
        row_count = 0
        buckets: dict[str, _BucketCounter] = {}
        for path in paths:
            file_rows = _validate_schema_and_count(path, spec.schema_factory())
            row_count += file_rows
            source_paths.append(path)
            if spec.language_spec is not None:
                _scan_language_file(path, spec.language_spec, buckets, file_rows)
        artifacts.append(
            ValidatedArtifact(
                table=spec.table,
                source_files=table_files,
                row_count=row_count,
            )
        )
        if spec.language_spec is not None:
            language_spec = spec.language_spec
            tables.append(
                LanguageTableInventory(
                    table=language_spec.table,
                    configuration=language_spec.configuration,
                    language_column=language_spec.language_column,
                    identity_columns=language_spec.identity_columns,
                    source_files=table_files,
                    row_count=row_count,
                    buckets=_freeze_buckets(buckets),
                )
            )

    if not tables:
        raise LanguageInventoryError(f"No language-bearing artifacts found under {root}")
    fingerprint_paths = [manifest, *source_paths]
    return LanguageInventory(
        contract=contract,
        dataset_id=_dataset_id(contract),
        contract_version=LANGUAGE_SPLIT_CONTRACT_VERSION,
        source_manifest=_relative_path(manifest, root),
        source_manifest_sha256=_sha256_file(manifest),
        artifact_fingerprint=_artifact_fingerprint(fingerprint_paths, root),
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

    def observe(self, resolution: LanguageResolution) -> None:
        self.row_count += 1
        if resolution.disposition is LanguageDisposition.CANONICAL:
            self.canonical_rows += 1
        elif resolution.disposition is LanguageDisposition.LEGACY_ALIAS:
            self.legacy_alias_rows += 1
        elif resolution.disposition is LanguageDisposition.MISSING:
            self.missing_rows += 1
        elif resolution.disposition is LanguageDisposition.BLANK:
            self.blank_rows += 1
        elif resolution.disposition is LanguageDisposition.MALFORMED:
            self.malformed_rows += 1
        elif resolution.disposition is LanguageDisposition.LEGACY_UNUSABLE:
            self.legacy_unusable_rows += 1


def _unknown(disposition: LanguageDisposition) -> LanguageResolution:
    return LanguageResolution(None, UNKNOWN_LANGUAGE, disposition)


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
        with pq.ParquetFile(path) as parquet_file:
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
        with pq.ParquetFile(path) as parquet_file:
            for batch in parquet_file.iter_batches(
                columns=[spec.language_column],
                batch_size=_BATCH_SIZE,
            ):
                values = batch.column(0).to_pylist()
                observed_rows += len(values)
                for value in values:
                    resolution = normalize_language(value)
                    buckets.setdefault(resolution.partition, _BucketCounter()).observe(resolution)
    except Exception as error:
        raise LanguageInventoryError(
            f"Could not scan language column {spec.language_column!r} in {path}: {error}"
        ) from error
    if observed_rows != expected_rows:
        raise LanguageInventoryError(
            f"row count changed while scanning {path}: metadata={expected_rows}, observed={observed_rows}"
        )


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


def _load_and_validate_manifest(
    root: Path,
    manifest: Path,
    dataset: DatasetContract,
) -> set[Path]:
    if not manifest.is_file():
        raise LanguageInventoryError(f"Processed manifest is missing: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LanguageInventoryError(
            f"Could not read processed manifest {manifest}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise LanguageInventoryError(f"Processed manifest must be a JSON object: {manifest}")

    entries = _manifest_entries(payload, dataset, manifest)
    fields = (
        (("polygons_path", "polygons"), ("polygon_articles_path", "polygon_articles"))
        if dataset is DatasetContract.V1
        else (
            ("polygons_path", "polygons"),
            ("documents_path", "wikipedia/documents"),
            ("sections_path", "wikipedia/sections"),
            ("links_path", "polygon_document_links"),
        )
    )
    referenced: set[Path] = set()
    for key, raw_entry in sorted(entries.items()):
        if not isinstance(key, str) or not isinstance(raw_entry, dict):
            raise LanguageInventoryError(f"Malformed manifest entry {key!r} in {manifest}")
        entry = cast(dict[str, object], raw_entry)
        for field, directory in fields:
            raw_path = entry.get(field)
            if not isinstance(raw_path, str):
                raise LanguageInventoryError(
                    f"Manifest entry {key!r} is missing string field {field!r}"
                )
            referenced.add(_manifest_path(root, raw_path, directory, key, field))
    if not entries:
        raise LanguageInventoryError(f"Processed manifest has no entries: {manifest}")
    return referenced


def _manifest_entries(
    payload: dict[str, object],
    dataset: DatasetContract,
    manifest: Path,
) -> dict[object, object]:
    if dataset is DatasetContract.V1:
        return cast(dict[object, object], payload)
    if payload.get("contract_version") != V2_CONTRACT_VERSION:
        raise LanguageInventoryError(
            f"V2 manifest contract version mismatch in {manifest}: expected {V2_CONTRACT_VERSION!r}"
        )
    entries = payload.get("regions")
    if not isinstance(entries, dict):
        raise LanguageInventoryError(f"V2 manifest regions must be an object: {manifest}")
    return cast(dict[object, object], entries)


def _manifest_path(
    root: Path,
    raw_path: str,
    expected_directory: str,
    key: object,
    field: str,
) -> Path:
    relative = Path(raw_path)
    if (
        relative.is_absolute()
        or not raw_path
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.suffix != ".parquet"
    ):
        raise LanguageInventoryError(
            f"Manifest entry {key!r} field {field!r} has unsafe path {raw_path!r}"
        )
    candidate = (root / relative).resolve()
    expected = (root / expected_directory).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise LanguageInventoryError(
            f"Manifest entry {key!r} field {field!r} escapes processed root"
        ) from error
    if candidate.parent != expected:
        raise LanguageInventoryError(
            f"Manifest entry {key!r} field {field!r} must be under {expected_directory!r}"
        )
    if not candidate.is_file():
        raise LanguageInventoryError(
            f"Manifest entry {key!r} field {field!r} points to missing artifact {raw_path!r}"
        )
    return candidate


def _require_manifest_references(
    paths: tuple[Path, ...],
    referenced: set[Path],
    table: str,
) -> None:
    missing = [path for path in paths if path not in referenced]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise LanguageInventoryError(f"{table} artifact is not referenced by the manifest: {names}")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise LanguageInventoryError(f"Artifact is outside processed root: {path}") from error


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise LanguageInventoryError(f"Could not hash artifact {path}: {error}") from error
    return digest.hexdigest()


def _artifact_fingerprint(paths: list[Path], root: Path) -> str:
    digest = sha256()
    for path in sorted(
        {path.resolve() for path in paths}, key=lambda item: _relative_path(item, root)
    ):
        digest.update(_relative_path(path, root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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
