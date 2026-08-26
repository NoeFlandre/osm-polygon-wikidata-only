"""Pure Hugging Face publication plans for the V2 artifact tree."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.hf.repo_layout import (
    LOCAL_V2_DATASET_HERO_FILE,
    REMOTE_DATASET_HERO_FILE,
)
from osm_polygon_wikidata_only.v2.config import V2_ASSET_PATHS
from osm_polygon_wikidata_only.v2.reuse import SIDECAR_SUBDIRS
from osm_polygon_wikidata_only.v2.sentence_runner import SENTENCE_MANIFEST_RELATIVE_PATH

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


def metadata_publication_ops(
    processed_v2: Path,
    *,
    hero_path: Path | None = LOCAL_V2_DATASET_HERO_FILE,
) -> list[PublicationOp]:
    """Return the V2 maps, card, and manifest publication plan."""
    operations = _add_existing(
        processed_v2,
        [
            *(Path(path) for path in V2_ASSET_PATHS),
            Path("manifests/processed_pbfs.json"),
            Path("README.md"),
        ],
    )
    if hero_path is not None:
        if not hero_path.is_file():
            raise FileNotFoundError(f"Dataset hero asset is missing: {hero_path}")
        operations.insert(0, add_op(hero_path, path_in_repo=REMOTE_DATASET_HERO_FILE))
    return operations


def sentence_publication_ops(processed_v2: Path, stems: Sequence[str]) -> list[PublicationOp]:
    """Return the sentence sidecars plus their routing manifest and card."""
    paths: list[Path] = []
    for stem in sorted(set(stems)):
        paths.append(Path("wikipedia/sentences") / f"{stem}.parquet")
        wikivoyage = Path("wikivoyage/sentences") / f"{stem}.parquet"
        if (processed_v2 / wikivoyage).is_file():
            paths.append(wikivoyage)
    paths.extend([SENTENCE_MANIFEST_RELATIVE_PATH, Path("README.md")])
    return _add_existing(processed_v2, paths)


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
    "sentence_publication_ops",
    "upload_region_batches",
]
