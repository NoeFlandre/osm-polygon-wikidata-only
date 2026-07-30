"""Fail-closed pairing rules for post-publication local retirement."""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp
from osm_polygon_wikidata_only.hf.repo_layout import (
    LEGACY_REMOTE_ARTICLES_DIR,
    REMOTE_WIKIPEDIA_DOCUMENTS_DIR,
)


def paired_retirement_stems(data_root: DataRoot, ops: list[PublicationOp]) -> set[str]:
    """Return stems with one valid canonical add and matching legacy delete."""
    add_counts: dict[str, int] = {}
    valid_adds: dict[str, Path] = {}
    deletes: set[str] = set()

    for operation in ops:
        remote = operation.path_in_repo
        if not isinstance(remote, str) or not remote:
            continue
        stem = Path(remote).stem
        if not _is_valid_stem(stem):
            continue
        if operation.action == "add":
            if remote != f"{REMOTE_WIKIPEDIA_DOCUMENTS_DIR}/{stem}.parquet":
                continue
            add_counts[stem] = add_counts.get(stem, 0) + 1
            local = operation.local_path
            if local is None:
                continue
            try:
                resolved = Path(local).resolve(strict=False)
                expected = (
                    data_root.processed / "wikipedia/documents" / f"{stem}.parquet"
                ).resolve()
            except (OSError, RuntimeError):
                continue
            if resolved != expected or not expected.is_file():
                continue
            prior = valid_adds.get(stem)
            valid_adds[stem] = (
                Path("__conflict__") if prior is not None and prior != resolved else resolved
            )
        elif (
            operation.action == "delete"
            and remote == f"{LEGACY_REMOTE_ARTICLES_DIR}/{stem}.parquet"
        ):
            deletes.add(stem)

    valid_single_adds = {
        stem
        for stem, local in valid_adds.items()
        if add_counts.get(stem) == 1 and local != Path("__conflict__")
    }
    return valid_single_adds & deletes


def _is_valid_stem(stem: str) -> bool:
    return bool(stem and stem not in {".", ".."} and "/" not in stem and "\\" not in stem)


__all__ = ["paired_retirement_stems"]
