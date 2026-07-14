"""Audit engine for characterizing and auditing the articles vs wikipedia/documents overlap."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    section_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    POLYGON_ARTICLE_COLUMNS,
    article_schema,
    polygon_article_schema,
)

# Order matching actual canonical ARTICLE_COLUMNS layout
CANONICAL_UPGRADE_COLUMNS = (
    "wikidata_label",
    "wikidata_description",
    "wikidata_aliases",
    "lead_text",
    "extract",
    "thumbnail_url",
    "thumbnail_width",
    "thumbnail_height",
    "categories",
)
CANONICAL_UPGRADE_SET = set(CANONICAL_UPGRADE_COLUMNS)


@dataclass(frozen=True)
class StemAuditResult:
    stem: str
    state: str
    discrepancies: tuple[str, ...] = field(default_factory=tuple)
    article_rows: int = 0
    document_rows: int = 0
    section_rows: int = 0
    link_rows: int = 0
    shared_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "state": self.state,
            "discrepancies": sorted(list(self.discrepancies)),
            "article_rows": self.article_rows,
            "document_rows": self.document_rows,
            "section_rows": self.section_rows,
            "link_rows": self.link_rows,
            "shared_rows": self.shared_rows,
        }


@dataclass(frozen=True)
class AuditReport:
    """Frozen container holding the results of the audit.

    Note: while the top-level dataclass instance is frozen, the collection payloads
    (aggregate_counts, byte_totals, per_stem, and schema_overlap_summary) are mutable dictionaries,
    which allows for convenient JSON serialization.
    """

    safe_to_migrate: bool
    blocking_reasons: tuple[str, ...]
    aggregate_counts: dict[str, Any]
    byte_totals: dict[str, int]
    per_stem: dict[str, Any]
    schema_overlap_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe_to_migrate": self.safe_to_migrate,
            "blocking_reasons": sorted(list(self.blocking_reasons)),
            "aggregate_counts": self.aggregate_counts,
            "byte_totals": self.byte_totals,
            "per_stem": self.per_stem,
            "schema_overlap_summary": self.schema_overlap_summary,
        }


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sanitize_error(e: Exception, data_root: Path) -> str:
    """Sanitize error messages to remove absolute path leakage."""
    err_str = f"{type(e).__name__}: {e}"
    abs_root = str(data_root.resolve())
    rel_root = str(data_root)
    if abs_root in err_str:
        err_str = err_str.replace(abs_root, "DATA_ROOT")
    if rel_root in err_str:
        err_str = err_str.replace(rel_root, "DATA_ROOT")
    # Mask absolute user and system paths
    err_str = re.sub(r"/Users/[^/\s\)\']+", "USER_HOME", err_str)
    err_str = re.sub(r"/private/var/folders/[^/\s\)\']+", "TEMP_DIR", err_str)
    err_str = re.sub(r"/var/folders/[^/\s\)\']+", "TEMP_DIR", err_str)
    return err_str


def capture_dataset_fingerprint(data_root: Path) -> dict[str, dict[str, Any]]:
    """Capture relative paths, sizes, mtimes, and sha256 hashes of all opened files."""
    processed = data_root / "processed"
    if not processed.exists():
        return {}

    fingerprint = {}
    paths_to_check = [
        processed / "articles",
        processed / "wikipedia" / "documents",
        processed / "wikipedia" / "sections",
        processed / "polygon_articles",
    ]

    for dir_path in paths_to_check:
        if not dir_path.exists():
            continue
        for p in dir_path.glob("*.parquet"):
            rel_path = p.relative_to(data_root)
            stat = p.stat()
            fingerprint[str(rel_path)] = {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": compute_sha256(p),
            }

    return dict(sorted(fingerprint.items()))


def validate_parquet_schema(
    pq_file: pq.ParquetFile,
    expected_cols: tuple[str, ...],
    expected_schema_fn: Any,
    table_type: str,
    allow_canonical_upgrades: bool = False,
) -> list[str]:
    """Validate Parquet file schema against expected columns and PyArrow types."""
    discrepancies = []
    actual_cols = pq_file.schema.names
    expected_schema = expected_schema_fn()

    # Missing columns
    missing_cols = sorted(list(set(expected_cols) - set(actual_cols)))
    if missing_cols:
        discrepancies.append(f"Missing column(s) in {table_type}: {missing_cols}")

    # Extra columns
    extra_cols = set(actual_cols) - set(expected_cols)
    if extra_cols:
        if allow_canonical_upgrades:
            unknown_cols = sorted(list(extra_cols - CANONICAL_UPGRADE_SET))
            if unknown_cols:
                discrepancies.append(f"Unknown extra column(s) in {table_type}: {unknown_cols}")
        else:
            unknown_cols = sorted(list(extra_cols))
            discrepancies.append(f"Unknown extra column(s) in {table_type}: {unknown_cols}")

    # PyArrow Type Mismatches
    arrow_schema = pq_file.schema.to_arrow_schema()
    for col in expected_cols:
        if col in actual_cols:
            actual_type = arrow_schema.field(col).type
            expected_type = expected_schema.field(col).type
            if actual_type != expected_type:
                discrepancies.append(
                    f"Schema type mismatch in {table_type} for column '{col}': "
                    f"expected {expected_type}, got {actual_type}"
                )

    # Check canonical column types if present
    if allow_canonical_upgrades:
        for col in CANONICAL_UPGRADE_COLUMNS:
            if col in actual_cols:
                actual_type = arrow_schema.field(col).type
                expected_type = article_schema().field(col).type
                if actual_type != expected_type:
                    discrepancies.append(
                        f"Schema type mismatch in {table_type} for canonical upgrade column '{col}': "
                        f"expected {expected_type}, got {actual_type}"
                    )

    return discrepancies


def run_audit(data_root: Path) -> dict[str, Any]:
    processed = data_root / "processed"
    if not processed.exists():
        raise RuntimeError(f"Processed directory not found under {data_root}")

    # Gather all stems across the four directories
    stems: set[str] = set()
    dirs = [
        processed / "articles",
        processed / "wikipedia" / "documents",
        processed / "wikipedia" / "sections",
        processed / "polygon_articles",
    ]
    for d in dirs:
        if d.exists():
            for p in d.glob("*.parquet"):
                stems.add(p.name[:-8])  # strip .parquet

    sorted_stems = sorted(list(stems))
    per_stem_results: dict[str, Any] = {}

    shared_cols_set = set(ARTICLE_COLUMNS) & set(DOCUMENT_COLUMNS)
    shared_columns = sorted(list(shared_cols_set))

    # Metrics
    article_files_count = 0
    wikipedia_document_files_count = 0
    wikipedia_section_files_count = 0
    polygon_article_link_files_count = 0

    total_article_rows = 0
    total_wikipedia_document_rows = 0
    total_wikipedia_section_rows = 0
    total_polygon_article_link_rows = 0
    total_shared_rows = 0

    articles_bytes_total = 0
    wikipedia_documents_bytes_total = 0

    stems_by_state = {
        "articles_only": 0,
        "documents_only": 0,
        "both_equivalent": 0,
        "both_needing_schema_upgrade": 0,
        "conflicting": 0,
        "orphaned": 0,
    }

    blocking_reasons: list[str] = []
    total_unresolved_links = 0
    total_unresolved_sections = 0
    unreadable_file_count = 0
    duplicate_primary_id_count = 0
    total_discrepancies = 0

    for stem in sorted_stems:
        art_path = processed / "articles" / f"{stem}.parquet"
        doc_path = processed / "wikipedia" / "documents" / f"{stem}.parquet"
        sec_path = processed / "wikipedia" / "sections" / f"{stem}.parquet"
        link_path = processed / "polygon_articles" / f"{stem}.parquet"

        has_art = art_path.exists()
        has_doc = doc_path.exists()
        has_sec = sec_path.exists()
        has_link = link_path.exists()

        has_art_file = has_art
        has_doc_file = has_doc
        has_sec_file = has_sec
        has_link_file = has_link

        stem_discrepancies: list[str] = []

        art_rows = 0
        doc_rows = 0
        sec_rows = 0
        link_rows = 0
        shared_rows_count = 0

        art_ids_set: set[str] = set()
        doc_ids_set: set[str] = set()
        doc_art_ids_set: set[str] = set()
        doc_art_ids: list[str] = []
        doc_ids: list[str] = []

        present_canonical_cols: list[str] = []
        has_canonical_mismatches = False

        if has_art:
            article_files_count += 1
            articles_bytes_total += art_path.stat().st_size
            try:
                art_file = pq.ParquetFile(art_path)
                art_rows = art_file.metadata.num_rows
                total_article_rows += art_rows

                # Schema validation for articles
                stem_discrepancies.extend(
                    validate_parquet_schema(
                        art_file,
                        ARTICLE_COLUMNS,
                        article_schema,
                        "articles",
                        allow_canonical_upgrades=False,
                    )
                )

                # Check duplicates in articles
                art_ids_table = art_file.read(columns=["article_id"])
                art_ids = art_ids_table.column("article_id").to_pylist()
                art_ids_set = set(art_ids)
                if len(art_ids_set) != len(art_ids):
                    stem_discrepancies.append(
                        "Duplicate article_id values found in articles Parquet"
                    )
                    duplicate_primary_id_count += len(art_ids) - len(art_ids_set)
            except Exception as e:
                stem_discrepancies.append(
                    f"Unreadable articles Parquet file: {sanitize_error(e, data_root)}"
                )
                unreadable_file_count += 1
                has_art = False

        if has_doc:
            wikipedia_document_files_count += 1
            wikipedia_documents_bytes_total += doc_path.stat().st_size
            try:
                doc_file = pq.ParquetFile(doc_path)
                doc_rows = doc_file.metadata.num_rows
                total_wikipedia_document_rows += doc_rows

                # Schema validation for documents (allow canonical upgrades)
                stem_discrepancies.extend(
                    validate_parquet_schema(
                        doc_file,
                        DOCUMENT_COLUMNS,
                        document_schema,
                        "documents",
                        allow_canonical_upgrades=True,
                    )
                )

                present_canonical_cols = [
                    col for col in CANONICAL_UPGRADE_COLUMNS if col in doc_file.schema.names
                ]

                # Check duplicates in documents
                doc_ids_table = doc_file.read(columns=["document_id", "article_id"])
                doc_art_ids = doc_ids_table.column("article_id").to_pylist()
                doc_art_ids_set = set(doc_art_ids)
                if len(doc_art_ids_set) != len(doc_art_ids):
                    stem_discrepancies.append(
                        "Duplicate article_id values found in documents Parquet"
                    )
                    duplicate_primary_id_count += len(doc_art_ids) - len(doc_art_ids_set)

                doc_ids = doc_ids_table.column("document_id").to_pylist()
                doc_ids_set = set(doc_ids)
                if len(doc_ids_set) != len(doc_ids):
                    stem_discrepancies.append(
                        "Duplicate document_id values found in documents Parquet"
                    )
                    duplicate_primary_id_count += len(doc_ids) - len(doc_ids_set)
            except Exception as e:
                stem_discrepancies.append(
                    f"Unreadable documents Parquet file: {sanitize_error(e, data_root)}"
                )
                unreadable_file_count += 1
                has_doc = False

        if has_sec:
            wikipedia_section_files_count += 1
            try:
                sec_file = pq.ParquetFile(sec_path)
                sec_rows = sec_file.metadata.num_rows
                total_wikipedia_section_rows += sec_rows

                # Schema validation for sections
                stem_discrepancies.extend(
                    validate_parquet_schema(
                        sec_file,
                        SECTION_COLUMNS,
                        section_schema,
                        "sections",
                        allow_canonical_upgrades=False,
                    )
                )
            except Exception as e:
                stem_discrepancies.append(
                    f"Unreadable sections Parquet file: {sanitize_error(e, data_root)}"
                )
                unreadable_file_count += 1
                has_sec = False

        if has_link:
            polygon_article_link_files_count += 1
            try:
                link_file = pq.ParquetFile(link_path)
                link_rows = link_file.metadata.num_rows
                total_polygon_article_link_rows += link_rows

                # Schema validation for links
                stem_discrepancies.extend(
                    validate_parquet_schema(
                        link_file,
                        POLYGON_ARTICLE_COLUMNS,
                        polygon_article_schema,
                        "links",
                        allow_canonical_upgrades=False,
                    )
                )
            except Exception as e:
                stem_discrepancies.append(
                    f"Unreadable polygon_articles Parquet file: {sanitize_error(e, data_root)}"
                )
                unreadable_file_count += 1
                has_link = False

        # Equivalence and value checks when both exist and are readable
        if has_art and has_doc:
            if art_rows != doc_rows:
                stem_discrepancies.append(
                    f"Row count mismatch: articles has {art_rows}, documents has {doc_rows}"
                )

            if art_ids_set != doc_art_ids_set:
                diff = sorted(list(art_ids_set ^ doc_art_ids_set))
                stem_discrepancies.append(
                    f"Article ID set mismatch: articles has {len(art_ids_set)} unique, "
                    f"documents has {len(doc_art_ids_set)} unique, symmetric difference: {diff}"
                )

            try:
                art_cols_to_read = shared_columns + [
                    c for c in present_canonical_cols if c not in shared_columns
                ]
                doc_cols_to_read = [*shared_columns, "document_id", "project"] + [
                    c for c in present_canonical_cols if c not in shared_columns
                ]

                art_table = art_file.read(columns=art_cols_to_read)
                doc_table = doc_file.read(columns=doc_cols_to_read)

                # Deterministic document_id & project validation in C++
                expected_doc_ids = pc.binary_join_element_wise(
                    doc_table.column("wikidata"),
                    pa.scalar("wikipedia"),
                    doc_table.column("language"),
                    pc.cast(doc_table.column("page_id"), pa.string()),
                    pc.cast(doc_table.column("revision_id"), pa.string()),
                    ":",
                )
                if not doc_table.column("document_id").equals(expected_doc_ids):
                    for act, exp, a_id in zip(
                        doc_table.column("document_id").to_pylist(),
                        expected_doc_ids.to_pylist(),
                        doc_table.column("article_id").to_pylist(),
                    ):
                        if act != exp:
                            stem_discrepancies.append(
                                f"Deterministic document_id mismatch for article_id '{a_id}': "
                                f"expected '{exp}', got '{act}'"
                            )

                proj_eq = pc.equal(doc_table.column("project"), pa.scalar("wikipedia"))
                if not pc.all(proj_eq).as_py():
                    for proj, a_id in zip(
                        doc_table.column("project").to_pylist(),
                        doc_table.column("article_id").to_pylist(),
                    ):
                        if proj != "wikipedia":
                            stem_discrepancies.append(
                                f"Project column is not 'wikipedia' for article_id '{a_id}': got '{proj}'"
                            )

                # Compare shared columns
                common_ids = sorted(list(art_ids_set & doc_art_ids_set))
                shared_rows_count = len(common_ids)
                total_shared_rows += shared_rows_count

                if art_ids_set == doc_art_ids_set:
                    art_sorted = art_table.select(shared_columns).sort_by(
                        [("article_id", "ascending")]
                    )
                    doc_sorted = doc_table.select(shared_columns).sort_by(
                        [("article_id", "ascending")]
                    )
                    if not art_sorted.equals(doc_sorted):
                        art_list = art_table.to_pylist()
                        doc_list = doc_table.to_pylist()
                        art_by_id = {r["article_id"]: r for r in art_list}
                        doc_by_id = {r["article_id"]: r for r in doc_list}
                        for a_id in common_ids:
                            a_row = art_by_id[a_id]
                            d_row = doc_by_id[a_id]
                            for col in shared_columns:
                                val_art = a_row[col]
                                val_doc = d_row[col]
                                if val_art != val_doc:
                                    stem_discrepancies.append(
                                        f"Value mismatch for article_id '{a_id}' in column '{col}': "
                                        f"articles has {val_art!r}, documents has {val_doc!r}"
                                    )
                else:
                    art_list = art_table.to_pylist()
                    doc_list = doc_table.to_pylist()
                    art_by_id = {r["article_id"]: r for r in art_list}
                    doc_by_id = {r["article_id"]: r for r in doc_list}
                    for a_id in common_ids:
                        a_row = art_by_id[a_id]
                        d_row = doc_by_id[a_id]
                        for col in shared_columns:
                            val_art = a_row[col]
                            val_doc = d_row[col]
                            if val_art != val_doc:
                                stem_discrepancies.append(
                                    f"Value mismatch for article_id '{a_id}' in column '{col}': "
                                    f"articles has {val_art!r}, documents has {val_doc!r}"
                                )

                # Lossless comparison of all present canonical upgrade columns
                if present_canonical_cols:
                    if art_ids_set == doc_art_ids_set:
                        cols_to_select = ["article_id", *present_canonical_cols]
                        art_sorted_canon = art_table.select(cols_to_select).sort_by(
                            [("article_id", "ascending")]
                        )
                        doc_sorted_canon = doc_table.select(cols_to_select).sort_by(
                            [("article_id", "ascending")]
                        )
                        if not art_sorted_canon.equals(doc_sorted_canon):
                            has_canonical_mismatches = True
                            art_list = art_table.to_pylist()
                            doc_list = doc_table.to_pylist()
                            art_by_id = {r["article_id"]: r for r in art_list}
                            doc_by_id = {r["article_id"]: r for r in doc_list}
                            for a_id in common_ids:
                                a_row = art_by_id[a_id]
                                d_row = doc_by_id[a_id]
                                for col in present_canonical_cols:
                                    val_art = a_row[col]
                                    val_doc = d_row[col]
                                    if val_art != val_doc:
                                        stem_discrepancies.append(
                                            f"Value mismatch for article_id '{a_id}' in column '{col}': "
                                            f"articles has {val_art!r}, documents has {val_doc!r}"
                                        )
                    else:
                        has_canonical_mismatches = True
                        art_list = art_table.to_pylist()
                        doc_list = doc_table.to_pylist()
                        art_by_id = {r["article_id"]: r for r in art_list}
                        doc_by_id = {r["article_id"]: r for r in doc_list}
                        for a_id in common_ids:
                            a_row = art_by_id[a_id]
                            d_row = doc_by_id[a_id]
                            for col in present_canonical_cols:
                                val_art = a_row[col]
                                val_doc = d_row[col]
                                if val_art != val_doc:
                                    stem_discrepancies.append(
                                        f"Value mismatch for article_id '{a_id}' in column '{col}': "
                                        f"articles has {val_art!r}, documents has {val_doc!r}"
                                    )
            except Exception as e:
                stem_discrepancies.append(
                    f"Error during equivalence checks: {sanitize_error(e, data_root)}"
                )

        # Complete section referential & identity integrity checks
        if has_sec:
            try:
                sec_table = sec_file.read(
                    columns=[
                        "section_id",
                        "article_id",
                        "document_id",
                        "wikidata",
                        "language",
                        "page_id",
                        "revision_id",
                    ]
                )
                # C++ unique section_id check
                sec_ids = sec_table.column("section_id")
                sec_ids_unique = pc.unique(sec_ids)
                if len(sec_ids_unique) != len(sec_ids):
                    stem_discrepancies.append(
                        "Duplicate section_id values found in sections Parquet"
                    )
                    duplicate_primary_id_count += len(sec_ids) - len(sec_ids_unique)

                # Check resolutions in articles
                if has_art:
                    in_art = pc.is_in(
                        sec_table.column("article_id"), value_set=pa.array(list(art_ids_set))
                    )
                    if not pc.all(in_art).as_py():
                        unresolved_mask = pc.invert(in_art)
                        unresolved_table = sec_table.filter(unresolved_mask)
                        for s_id, s_art_id in zip(
                            unresolved_table.column("section_id").to_pylist(),
                            unresolved_table.column("article_id").to_pylist(),
                        ):
                            stem_discrepancies.append(
                                f"Section '{s_id}' references unresolved article_id '{s_art_id}' in articles"
                            )
                            total_unresolved_sections += 1

                # Check resolutions in documents
                if has_doc:
                    in_doc = pc.is_in(
                        sec_table.column("document_id"), value_set=pa.array(list(doc_ids_set))
                    )
                    if not pc.all(in_doc).as_py():
                        unresolved_mask = pc.invert(in_doc)
                        unresolved_table = sec_table.filter(unresolved_mask)
                        for s_id, s_doc_id in zip(
                            unresolved_table.column("section_id").to_pylist(),
                            unresolved_table.column("document_id").to_pylist(),
                        ):
                            stem_discrepancies.append(
                                f"Section '{s_id}' references unresolved document_id '{s_doc_id}' in documents"
                            )
                            total_unresolved_sections += 1

                # Match identities at PyArrow C++ level to avoid Python loops
                if has_doc:
                    # 1. Check section article_id matching document_id (strip project wikipedia)
                    expected_art_ids = pc.replace_substring(
                        sec_table.column("document_id"), pattern=":wikipedia:", replacement=":"
                    )
                    art_id_match = pc.equal(sec_table.column("article_id"), expected_art_ids)
                    if not pc.all(art_id_match).as_py():
                        mismatched_rows = sec_table.filter(pc.invert(art_id_match))
                        for row in mismatched_rows.to_pylist():
                            s_id = row["section_id"]
                            s_art_id = row["article_id"]
                            s_doc_id = row["document_id"]
                            stem_discrepancies.append(
                                f"Section '{s_id}' identity mismatch: document_id '{s_doc_id}' "
                                f"does not represent the same identity as article_id '{s_art_id}'"
                            )

                    # 2. Check section row fields match document_id parts (build expected document_id in C++)
                    expected_doc_ids = pc.binary_join_element_wise(
                        sec_table.column("wikidata"),
                        pa.scalar("wikipedia"),
                        sec_table.column("language"),
                        pc.cast(sec_table.column("page_id"), pa.string()),
                        pc.cast(sec_table.column("revision_id"), pa.string()),
                        ":",
                    )
                    doc_id_match = pc.equal(sec_table.column("document_id"), expected_doc_ids)
                    if not pc.all(doc_id_match).as_py():
                        mismatched_rows = sec_table.filter(pc.invert(doc_id_match))
                        for row in mismatched_rows.to_pylist():
                            s_id = row["section_id"]
                            s_doc_id = row["document_id"]
                            stem_discrepancies.append(
                                f"Section '{s_id}' fields mismatch: values do not match document_id '{s_doc_id}'"
                            )
            except Exception as e:
                stem_discrepancies.append(
                    f"Error checking sections referential integrity: {sanitize_error(e, data_root)}"
                )

        if has_link:
            try:
                link_table = link_file.read(columns=["polygon_id", "article_id"])
                poly_ids_col = link_table.column("polygon_id")
                link_art_ids_col = link_table.column("article_id")
                if has_art:
                    in_art = pc.is_in(link_art_ids_col, value_set=pa.array(list(art_ids_set)))
                    if not pc.all(in_art).as_py():
                        unresolved_mask = pc.invert(in_art)
                        unresolved_table = link_table.filter(unresolved_mask)
                        for poly_id, l_art_id in zip(
                            unresolved_table.column("polygon_id").to_pylist(),
                            unresolved_table.column("article_id").to_pylist(),
                        ):
                            stem_discrepancies.append(
                                f"Link (polygon '{poly_id}') references unresolved article_id '{l_art_id}' in articles"
                            )
                            total_unresolved_links += 1
                if has_doc:
                    in_doc = pc.is_in(link_art_ids_col, value_set=pa.array(list(doc_art_ids_set)))
                    if not pc.all(in_doc).as_py():
                        unresolved_mask = pc.invert(in_doc)
                        unresolved_table = link_table.filter(unresolved_mask)
                        for poly_id, l_art_id in zip(
                            unresolved_table.column("polygon_id").to_pylist(),
                            unresolved_table.column("article_id").to_pylist(),
                        ):
                            stem_discrepancies.append(
                                f"Link (polygon '{poly_id}') references unresolved article_id '{l_art_id}' in documents"
                            )
                            total_unresolved_links += 1

                # Check duplicate links
                unique_links_count = len(
                    link_table.group_by(["polygon_id", "article_id"]).aggregate([])
                )
                if unique_links_count != link_table.num_rows:
                    poly_ids = poly_ids_col.to_pylist()
                    link_art_ids = link_art_ids_col.to_pylist()
                    seen_links = set()
                    duplicate_links_count = 0
                    for poly_id, l_art_id in zip(poly_ids, link_art_ids):
                        link_key = (poly_id, l_art_id)
                        if link_key in seen_links:
                            duplicate_links_count += 1
                        seen_links.add(link_key)
                    if duplicate_links_count > 0:
                        stem_discrepancies.append(
                            f"Duplicate polygon-article links found: {duplicate_links_count} duplicates"
                        )
            except Exception as e:
                stem_discrepancies.append(
                    f"Error checking links referential integrity: {sanitize_error(e, data_root)}"
                )

        # Deterministic Classification Rules
        if has_art_file and has_doc_file:
            if len(stem_discrepancies) > 0 or has_canonical_mismatches:
                state = "conflicting"
            elif len(present_canonical_cols) == len(CANONICAL_UPGRADE_COLUMNS):
                state = "both_equivalent"
            else:
                state = "both_needing_schema_upgrade"
        elif has_art_file and not has_doc_file:
            if has_sec_file:
                state = "orphaned"
            else:
                state = "articles_only"
        elif has_doc_file and not has_art_file:
            if has_sec_file or has_link_file:
                state = "orphaned"
            else:
                state = "documents_only"
        else:  # not has_art_file and not has_doc_file
            state = "orphaned"

        stems_by_state[state] += 1
        total_discrepancies += len(stem_discrepancies)

        if len(stem_discrepancies) > 0:
            blocking_reasons.append(
                f"Discrepancies in stem '{stem}': {', '.join(stem_discrepancies)}"
            )

        per_stem_results[stem] = StemAuditResult(
            stem=stem,
            state=state,
            discrepancies=tuple(stem_discrepancies),
            article_rows=art_rows,
            document_rows=doc_rows,
            section_rows=sec_rows,
            link_rows=link_rows,
            shared_rows=shared_rows_count,
        ).to_dict()

    # Determine safe_to_migrate (fail closed)
    has_conflicts = stems_by_state["conflicting"] > 0
    has_orphans = stems_by_state["orphaned"] > 0
    has_docs_only = stems_by_state["documents_only"] > 0
    safe_to_migrate = not (
        has_conflicts
        or has_orphans
        or has_docs_only
        or total_unresolved_links > 0
        or total_unresolved_sections > 0
        or unreadable_file_count > 0
        or duplicate_primary_id_count > 0
        or total_discrepancies > 0
    )

    if has_conflicts:
        blocking_reasons.append(f"Found {stems_by_state['conflicting']} conflicting stems.")
    if has_orphans:
        blocking_reasons.append(f"Found {stems_by_state['orphaned']} orphaned stems.")
    if has_docs_only:
        blocking_reasons.append(
            f"Found {stems_by_state['documents_only']} documents_only stems (missing source articles)."
        )
    if total_unresolved_links > 0:
        blocking_reasons.append(f"Found {total_unresolved_links} unresolved polygon-article links.")
    if total_unresolved_sections > 0:
        blocking_reasons.append(f"Found {total_unresolved_sections} unresolved Wikipedia sections.")
    if unreadable_file_count > 0:
        blocking_reasons.append(f"Found {unreadable_file_count} unreadable Parquet files.")
    if duplicate_primary_id_count > 0:
        blocking_reasons.append(f"Found {duplicate_primary_id_count} duplicate primary IDs.")
    if total_discrepancies > 0 and len(blocking_reasons) == 0:
        blocking_reasons.append(f"Found {total_discrepancies} schema or metadata discrepancies.")

    # Sort blocking reasons deterministically
    blocking_reasons = sorted(list(set(blocking_reasons)))

    schema_overlap_summary = {
        "articles_columns_count": len(ARTICLE_COLUMNS),
        "documents_columns_count": len(DOCUMENT_COLUMNS),
        "shared_columns_count": len(shared_cols_set),
        "articles_only_columns": sorted(list(set(ARTICLE_COLUMNS) - set(DOCUMENT_COLUMNS))),
        "documents_only_columns": sorted(list(set(DOCUMENT_COLUMNS) - set(ARTICLE_COLUMNS))),
    }

    aggregate_counts = {
        "article_files": article_files_count,
        "wikipedia_document_files": wikipedia_document_files_count,
        "wikipedia_section_files": wikipedia_section_files_count,
        "polygon_article_link_files": polygon_article_link_files_count,
        "article_rows": total_article_rows,
        "wikipedia_document_rows": total_wikipedia_document_rows,
        "wikipedia_section_rows": total_wikipedia_section_rows,
        "polygon_article_link_rows": total_polygon_article_link_rows,
        "shared_article_document_rows": total_shared_rows,
        "stems_by_state": stems_by_state,
        "conflicting_stem_count": stems_by_state["conflicting"],
        "discrepancy_count": total_discrepancies,
        "unreadable_file_count": unreadable_file_count,
        "duplicate_primary_id_count": duplicate_primary_id_count,
        "total_unresolved_links": total_unresolved_links,
        "total_unresolved_sections": total_unresolved_sections,
    }

    byte_totals = {
        "articles_bytes": articles_bytes_total,
        "wikipedia_documents_bytes": wikipedia_documents_bytes_total,
    }

    report = AuditReport(
        safe_to_migrate=safe_to_migrate,
        blocking_reasons=tuple(blocking_reasons),
        aggregate_counts=aggregate_counts,
        byte_totals=byte_totals,
        per_stem=per_stem_results,
        schema_overlap_summary=schema_overlap_summary,
    )

    return report.to_dict()


def write_audit_report(report_data: dict, output_path: Path) -> None:
    """Write the audit report in a deterministic sorted JSON format."""
    serialized = json.dumps(report_data, indent=2, sort_keys=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
