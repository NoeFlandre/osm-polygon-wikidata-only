"""Pure Hugging Face publication plans for the V2 artifact tree."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.v2.reuse import SIDECAR_SUBDIRS

Upload = Callable[[list[PublicationOp], str], None]
_REGION_UPLOAD_BATCH_SIZE = 16


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


def upload_region_batches(
    processed_v2: Path,
    stems: Sequence[str],
    *,
    upload: Upload,
    repair: bool,
) -> None:
    """Upload regions in bounded, deterministically ordered Hub commits."""
    for offset in range(0, len(stems), _REGION_UPLOAD_BATCH_SIZE):
        batch = tuple(stems[offset : offset + _REGION_UPLOAD_BATCH_SIZE])
        operations = [
            operation for stem in batch for operation in region_publication_ops(processed_v2, stem)
        ]
        upload(operations, _region_upload_message(batch, repair=repair))


def remote_region_complete(
    remote_inventory: RemoteInventory | None,
    processed_v2: Path,
    stem: str,
) -> bool:
    """Return whether every locally planned region artifact exists remotely."""
    if remote_inventory is None:
        return False
    return all(
        remote_inventory.contains(operation.path_in_repo)
        for operation in region_publication_ops(processed_v2, stem)
    )


def _region_upload_message(stems: Sequence[str], *, repair: bool) -> str:
    prefix = "Repair" if repair else "Add"
    if len(stems) == 1:
        return f"{prefix} V2 region {stems[0]}"
    return f"{prefix} V2 regions {stems[0]} through {stems[-1]} ({len(stems)} regions)"


def _add_existing(processed_v2: Path, paths: list[Path]) -> list[PublicationOp]:
    missing = [processed_v2 / path for path in paths if not (processed_v2 / path).is_file()]
    if missing:
        raise FileNotFoundError(f"V2 publication artifact(s) missing: {missing}")
    return [add_op(processed_v2 / path, path_in_repo=path.as_posix()) for path in paths]


__all__ = [
    "metadata_publication_ops",
    "region_publication_ops",
    "remote_region_complete",
    "upload_region_batches",
]
