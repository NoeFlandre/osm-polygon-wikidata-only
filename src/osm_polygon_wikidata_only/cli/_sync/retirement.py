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
    add_counts, valid_adds, deletes = _scan_operations(data_root, ops)
    return _valid_single_adds(add_counts, valid_adds) & deletes


def _scan_operations(
    data_root: DataRoot,
    ops: list[PublicationOp],
) -> tuple[dict[str, int], dict[str, Path], set[str]]:
    add_counts: dict[str, int] = {}
    valid_adds: dict[str, Path] = {}
    deletes: set[str] = set()

    for operation in ops:
        _scan_operation(data_root, operation, add_counts, valid_adds, deletes)
    return add_counts, valid_adds, deletes


def _scan_operation(
    data_root: DataRoot,
    operation: PublicationOp,
    add_counts: dict[str, int],
    valid_adds: dict[str, Path],
    deletes: set[str],
) -> None:
    stem = _operation_stem(operation)
    if stem is None:
        return
    if operation.action == "add":
        if _is_canonical_add(operation.path_in_repo, operation.action, stem):
            _record_add(data_root, operation, stem, add_counts, valid_adds)
    elif _is_legacy_delete(operation.path_in_repo, stem):
        deletes.add(stem)


def _record_add(
    data_root: DataRoot,
    operation: PublicationOp,
    stem: str,
    add_counts: dict[str, int],
    valid_adds: dict[str, Path],
) -> None:
    add_counts[stem] = add_counts.get(stem, 0) + 1
    resolved = _canonical_add_path(data_root, operation, stem)
    if resolved is not None:
        valid_adds[stem] = _merge_add_path(valid_adds.get(stem), resolved)


def _valid_single_adds(
    add_counts: dict[str, int],
    valid_adds: dict[str, Path],
) -> set[str]:
    return {
        stem
        for stem, local in valid_adds.items()
        if add_counts.get(stem) == 1 and local != Path("__conflict__")
    }


def _operation_stem(operation: PublicationOp) -> str | None:
    remote = operation.path_in_repo
    if not isinstance(remote, str) or not remote:
        return None
    stem = Path(remote).stem
    return stem if _is_valid_stem(stem) else None


def _canonical_add_path(
    data_root: DataRoot,
    operation: PublicationOp,
    stem: str,
) -> Path | None:
    if not _is_canonical_add(operation.path_in_repo, operation.action, stem):
        return None
    if operation.local_path is None:
        return None
    return _resolve_canonical_path(data_root, operation.local_path, stem)


def _is_canonical_add(remote: object, action: str, stem: str) -> bool:
    return action == "add" and remote == f"{REMOTE_WIKIPEDIA_DOCUMENTS_DIR}/{stem}.parquet"


def _resolve_canonical_path(
    data_root: DataRoot,
    local_path: object,
    stem: str,
) -> Path | None:
    if not isinstance(local_path, (str, Path)):
        return None
    try:
        resolved = Path(local_path).resolve(strict=False)
        expected = (data_root.processed / "wikipedia/documents" / f"{stem}.parquet").resolve()
    except (OSError, RuntimeError):
        return None
    return resolved if resolved == expected and expected.is_file() else None


def _merge_add_path(prior: Path | None, resolved: Path) -> Path:
    if prior is not None and prior != resolved:
        return Path("__conflict__")
    return resolved


def _is_legacy_delete(remote: object, stem: str) -> bool:
    return remote == f"{LEGACY_REMOTE_ARTICLES_DIR}/{stem}.parquet"


def _is_valid_stem(stem: str) -> bool:
    return bool(stem and stem not in {".", ".."} and "/" not in stem and "\\" not in stem)


__all__ = ["paired_retirement_stems"]
