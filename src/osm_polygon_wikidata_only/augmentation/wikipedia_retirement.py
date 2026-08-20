"""Fail-closed retirement of locally staged legacy Wikipedia articles."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    build_wikipedia_document_table,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps

from .steps import sha256_file
from .wikipedia_document_migration import (
    MigrationError,
    _assert_canonical_preserves_legacy,
)


def _assert_legacy_rows_preserved(
    canonical_path: Path,
    legacy_path: Path,
    stem: str,
) -> None:
    """Require canonical documents to contain every converted legacy row."""
    canonical = pq.read_table(canonical_path)  # type: ignore[no-untyped-call]
    expected = build_wikipedia_document_table(pq.read_table(legacy_path))  # type: ignore[no-untyped-call]
    try:
        _assert_canonical_preserves_legacy(canonical, expected, stem)
    except MigrationError as error:
        raise MigrationError(f"Stem {stem!r} is not safe to retire: {error}") from error


def _assert_references_resolve(data_root: DataRoot, stem: str) -> None:
    documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    links_path = data_root.processed_links / f"{stem}.parquet"
    sections_path = data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet"
    documents = pq.read_table(documents_path, columns=["article_id", "document_id"])  # type: ignore[no-untyped-call]
    article_ids = set(documents["article_id"].to_pylist())
    document_ids = set(documents["document_id"].to_pylist())
    _assert_link_references_resolve(links_path, article_ids, document_ids, stem)
    _assert_section_references_resolve(sections_path, document_ids, stem)


def _assert_link_references_resolve(
    links_path: Path,
    article_ids: set[object],
    document_ids: set[object],
    stem: str,
) -> None:
    if not links_path.exists():
        return
    link_schema = pq.read_schema(links_path)  # type: ignore[no-untyped-call]
    if "article_id" in link_schema.names:
        unresolved = _unresolved_article_links(links_path, article_ids)
    else:
        unresolved = _unresolved_document_links(links_path, document_ids)
    if unresolved:
        raise MigrationError(f"Stem {stem!r} has polygon links unresolved by canonical documents")


def _unresolved_article_links(links_path: Path, article_ids: set[object]) -> bool:
    links = pq.read_table(links_path, columns=["article_id"])  # type: ignore[no-untyped-call]
    return any(article_id not in article_ids for article_id in links["article_id"].to_pylist())


def _unresolved_document_links(links_path: Path, document_ids: set[object]) -> bool:
    links = pq.read_table(links_path, columns=["document_id", "project"])  # type: ignore[no-untyped-call]
    return any(
        project == "wikipedia" and document_id not in document_ids
        for document_id, project in zip(
            links["document_id"].to_pylist(), links["project"].to_pylist(), strict=True
        )
    )


def _assert_section_references_resolve(
    sections_path: Path,
    document_ids: set[object],
    stem: str,
) -> None:
    if not sections_path.exists():
        return
    sections = pq.read_table(sections_path, columns=["document_id"])  # type: ignore[no-untyped-call]
    if any(document_id not in document_ids for document_id in sections["document_id"].to_pylist()):
        raise MigrationError(f"Stem {stem!r} has sections unresolved by canonical documents")


def _validate_retirement_inputs(data_root: DataRoot, stem: str) -> tuple[Path, Path]:
    canonical = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    legacy = data_root.processed_articles / f"{stem}.parquet"
    if not canonical.exists() or not pq.read_schema(canonical).equals(  # type: ignore[no-untyped-call]
        wikipedia_document_schema(), check_metadata=True
    ):
        raise MigrationError(f"Stem {stem!r} has no valid canonical Wikipedia document")
    if legacy.exists():
        _assert_legacy_rows_preserved(canonical, legacy, stem)
    _assert_references_resolve(data_root, stem)
    return canonical, legacy


def _update_processed_manifest(
    data_root: DataRoot, stem: str, canonical: Path, legacy: Path
) -> None:
    processed_manifest = data_root.processed_manifests / "processed_pbfs.json"
    if not processed_manifest.exists():
        return
    payload = json.loads(processed_manifest.read_text(encoding="utf-8"))
    entry = payload.get(f"{stem}.osm.pbf")
    if isinstance(entry, dict):
        entry.pop("articles_path", None)
        entry["wikipedia_documents_path"] = f"wikipedia/documents/{stem}.parquet"
        atomic_write_text(processed_manifest, dumps(payload) + "\n")


def _update_augmentation_manifest(
    data_root: DataRoot, stem: str, canonical: Path, legacy: Path
) -> None:
    augmentation_manifest = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    if not augmentation_manifest.exists():
        return
    payload = json.loads(augmentation_manifest.read_text(encoding="utf-8"))
    entry = payload.get(stem)
    if isinstance(entry, dict) and isinstance(entry.get("core_hashes"), dict):
        hashes = entry["core_hashes"]
        hashes.pop(str(legacy), None)
        hashes[str(canonical)] = sha256_file(canonical)
        atomic_write_text(augmentation_manifest, dumps(payload) + "\n")


def prepare_local_retirement(data_root: DataRoot, stem: str) -> None:
    """Verify losslessness and atomically repoint manifests to canonical data."""
    canonical, legacy = _validate_retirement_inputs(data_root, stem)
    _update_processed_manifest(data_root, stem, canonical, legacy)
    _update_augmentation_manifest(data_root, stem, canonical, legacy)


def finalize_local_retirement(data_root: DataRoot, stem: str) -> None:
    """Delete a legacy local article only after all safety checks succeed."""
    legacy = data_root.processed_articles / f"{stem}.parquet"
    if not legacy.exists():
        return
    prepare_local_retirement(data_root, stem)
    legacy.unlink()


__all__ = ["finalize_local_retirement", "prepare_local_retirement"]
