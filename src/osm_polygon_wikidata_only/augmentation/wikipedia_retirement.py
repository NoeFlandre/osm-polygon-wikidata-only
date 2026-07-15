"""Fail-closed retirement of locally staged legacy Wikipedia articles."""

from __future__ import annotations

import json

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps

from .steps import sha256_file
from .wikipedia_document_migration import MigrationError, plan_migration


def _assert_references_resolve(data_root: DataRoot, stem: str) -> None:
    documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    links_path = data_root.processed_links / f"{stem}.parquet"
    sections_path = data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet"
    documents = pq.read_table(documents_path, columns=["article_id", "document_id"])  # type: ignore[no-untyped-call]
    article_ids = set(documents["article_id"].to_pylist())
    document_ids = set(documents["document_id"].to_pylist())
    if links_path.exists():
        links = pq.read_table(links_path, columns=["article_id"])  # type: ignore[no-untyped-call]
        if any(article_id not in article_ids for article_id in links["article_id"].to_pylist()):
            raise MigrationError(
                f"Stem {stem!r} has polygon links unresolved by canonical documents"
            )
    if sections_path.exists():
        sections = pq.read_table(sections_path, columns=["document_id"])  # type: ignore[no-untyped-call]
        if any(
            document_id not in document_ids for document_id in sections["document_id"].to_pylist()
        ):
            raise MigrationError(f"Stem {stem!r} has sections unresolved by canonical documents")


def prepare_local_retirement(data_root: DataRoot, stem: str) -> None:
    """Verify losslessness and atomically repoint manifests to canonical data."""
    canonical = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    legacy = data_root.processed_articles / f"{stem}.parquet"
    if legacy.exists():
        plan = plan_migration(data_root.processed, stems={stem})
        if not plan.is_safe_to_apply or len(plan.stems) != 1:
            raise MigrationError(f"Stem {stem!r} is not safe to retire")
    if not canonical.exists() or not pq.read_schema(canonical).equals(  # type: ignore[no-untyped-call]
        wikipedia_document_schema(), check_metadata=True
    ):
        raise MigrationError(f"Stem {stem!r} has no valid canonical Wikipedia document")
    _assert_references_resolve(data_root, stem)
    processed_manifest = data_root.processed_manifests / "processed_pbfs.json"
    if processed_manifest.exists():
        payload = json.loads(processed_manifest.read_text(encoding="utf-8"))
        key = f"{stem}.osm.pbf"
        entry = payload.get(key)
        if isinstance(entry, dict):
            entry.pop("articles_path", None)
            entry["wikipedia_documents_path"] = f"wikipedia/documents/{stem}.parquet"
            atomic_write_text(processed_manifest, dumps(payload) + "\n")

    augmentation_manifest = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    if augmentation_manifest.exists():
        payload = json.loads(augmentation_manifest.read_text(encoding="utf-8"))
        entry = payload.get(stem)
        if isinstance(entry, dict) and isinstance(entry.get("core_hashes"), dict):
            hashes = entry["core_hashes"]
            hashes.pop(str(legacy), None)
            hashes[str(canonical)] = sha256_file(canonical)
            atomic_write_text(augmentation_manifest, dumps(payload) + "\n")


def finalize_local_retirement(data_root: DataRoot, stem: str) -> None:
    """Delete a legacy local article only after all safety checks succeed."""
    legacy = data_root.processed_articles / f"{stem}.parquet"
    if not legacy.exists():
        return
    prepare_local_retirement(data_root, stem)
    legacy.unlink()


__all__ = ["finalize_local_retirement", "prepare_local_retirement"]
