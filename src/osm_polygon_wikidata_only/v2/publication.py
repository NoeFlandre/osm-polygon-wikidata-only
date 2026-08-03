"""Pure Hugging Face publication plans for the V2 artifact tree."""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op
from osm_polygon_wikidata_only.v2.reuse import SIDECAR_SUBDIRS


def region_publication_ops(processed_v2: Path, stem: str) -> list[PublicationOp]:
    """Return one deterministic atomic add plan for a V2 region."""
    required_paths = [
        Path("polygons") / f"{stem}.parquet",
        Path("wikipedia/documents") / f"{stem}.parquet",
        Path("wikipedia/sections") / f"{stem}.parquet",
        Path("polygon_document_links") / f"{stem}.parquet",
    ]
    optional_paths = [
        Path(subdir) / f"{stem}.parquet"
        for subdir in SIDECAR_SUBDIRS
        if subdir != "wikipedia/sections"
    ]
    return _add_existing(
        processed_v2,
        required_paths + [path for path in optional_paths if (processed_v2 / path).is_file()],
    )


def metadata_publication_ops(processed_v2: Path) -> list[PublicationOp]:
    """Return the V2 card and manifest publication plan."""
    return _add_existing(
        processed_v2,
        [Path("README.md"), Path("manifests/processed_pbfs.json")],
    )


def _add_existing(processed_v2: Path, paths: list[Path]) -> list[PublicationOp]:
    missing = [processed_v2 / path for path in paths if not (processed_v2 / path).is_file()]
    if missing:
        raise FileNotFoundError(f"V2 publication artifact(s) missing: {missing}")
    return [add_op(processed_v2 / path, path_in_repo=path.as_posix()) for path in paths]


__all__ = ["metadata_publication_ops", "region_publication_ops"]
