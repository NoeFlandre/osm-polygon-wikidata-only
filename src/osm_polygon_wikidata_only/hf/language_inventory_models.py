"""Value types for language-split inventories."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import pyarrow as pa

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
        match = _find_named_inventory(self.tables, table_name)
        if match is None and allow_non_language:
            match = _find_named_inventory(self.validated_artifacts, table_name)
        if match is None:
            raise KeyError(f"No inventory for table {table_name!r}")
        return match

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


def _find_named_inventory(
    records: Sequence[LanguageTableInventory | ValidatedArtifact],
    table_name: str,
) -> LanguageTableInventory | ValidatedArtifact | None:
    """Return the first inventory record with the requested logical name."""
    for record in records:
        record_name = (
            record.table.value if isinstance(record.table, LanguageTable) else record.table
        )
        if record_name == table_name:
            return record
    return None


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
    return _normalize_language_text(stripped)


def _normalize_language_text(value: str) -> LanguageResolution:
    """Classify one non-blank language string after outer validation."""
    lowered = value.lower()
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


def _unknown(disposition: LanguageDisposition) -> LanguageResolution:
    return LanguageResolution(None, UNKNOWN_LANGUAGE, disposition)
