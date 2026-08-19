"""Explicit dependencies shared by publication assemblers.

The facade builds this object at call time. That keeps existing monkeypatch and
injection points working while the ordered assemblers live in small modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp


@dataclass(frozen=True, slots=True)
class PublicationHooks:
    """Runtime dependencies used by the pure publication assemblers."""

    dataset_hero_op: Callable[[], PublicationOp]
    augmentation_migration_ops: Callable[[Path], list[PublicationOp]]
    legacy_article_retirement_ops: Callable[..., list[PublicationOp]]
    snapshot_upload_manifests: Callable[..., tuple[Path, Path]]
    snapshot_canonical_document: Callable[..., Path]
    metadata_only_upload: Callable[..., list[PublicationOp]]
    write_readme_snapshot: Callable[[DataRoot, str, Path], None]
    refresh_coverage_assets: Callable[..., tuple[Path, Path, Path]]
    ensure_world_land: Callable[[Path], Path]
    generate_coverage_map: Callable[..., Any]
    load_centroids_from_parquet: Callable[..., Any]
    generate_geographic_text_presence: Callable[..., Any]
    load_text_presence: Callable[..., Any]
    generate_geographic_text_density_snapshot: Callable[..., Path]


__all__ = ["PublicationHooks"]
