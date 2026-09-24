"""Data-root resolution shared by the language release and publication."""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.config.paths import DataRoot

__all__ = ["resolve_data_root"]


def resolve_data_root(
    data_root: DataRoot | Path,
    error_cls: type[Exception],
    *,
    label: str = "data root",
) -> Path:
    """Return the resolved data-root directory or raise *error_cls*."""
    root = data_root.path if isinstance(data_root, DataRoot) else Path(data_root)
    root = root.resolve()
    if not root.is_dir():
        raise error_cls(f"{label} is not a directory: {root}")
    return root
