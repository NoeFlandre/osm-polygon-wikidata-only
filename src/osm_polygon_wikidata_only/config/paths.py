"""Resolve the external data root used by the pipeline.

The repository is intentionally code-only: all PBF inputs, intermediate
artifacts, saved datasets, and caches live outside the working tree on
the configured external drive.

Resolution precedence (highest first):

1. Explicit value passed to :class:`DataRoot` (typically from ``--data-root``).
2. ``OSM_POLYGON_DATA_ROOT`` environment variable.
3. The conventional local path ``/Volumes/Seagate M3/projects/osm-polygon-wikidata-only``
   when it exists on disk (recommended default on the local setup).

If none of the above yields a usable path, a clear error is raised.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


ENV_VAR = "OSM_POLYGON_DATA_ROOT"

# Recommended local path. Documented; never the only valid path.
DEFAULT_LOCAL_DATA_ROOT = Path("/Volumes/Seagate M3/projects/osm-polygon-wikidata-only")

# Conventional top-level sub-directories under the data root.
SUBDIR_RAW = "raw"
SUBDIR_PROCESSED = "processed"
SUBDIR_LOGS = "logs"
SUBDIR_HF_CACHE = "hf_cache"
SUBDIR_CACHE = "cache"

# Sub-sub-directories under ``processed/``.
PROCESSED_POLYGONS = "polygons"
PROCESSED_ARTICLES = "articles"
PROCESSED_LINKS = "polygon_articles"
PROCESSED_MANIFESTS = "manifests"

# Sub-sub-directories under ``cache/``.
CACHE_WIKIDATA = "wikidata"
CACHE_WIKIPEDIA = "wikipedia"


class DataRootError(RuntimeError):
    """Raised when the data root cannot be located or is unsafe to use."""


@dataclass(frozen=True)
class DataRoot:
    """Resolved external data root for the pipeline."""

    path: Path

    def sub(self, name: str) -> Path:
        """Return ``<path>/<name>`` without creating it."""
        return self.path / name

    @property
    def raw(self) -> Path:
        return self.sub(SUBDIR_RAW)

    @property
    def processed(self) -> Path:
        return self.sub(SUBDIR_PROCESSED)

    @property
    def logs(self) -> Path:
        return self.sub(SUBDIR_LOGS)

    @property
    def hf_cache(self) -> Path:
        return self.sub(SUBDIR_HF_CACHE)

    @property
    def cache(self) -> Path:
        return self.sub(SUBDIR_CACHE)

    @property
    def processed_polygons(self) -> Path:
        return self.processed / PROCESSED_POLYGONS

    @property
    def processed_articles(self) -> Path:
        return self.processed / PROCESSED_ARTICLES

    @property
    def processed_links(self) -> Path:
        return self.processed / PROCESSED_LINKS

    @property
    def processed_manifests(self) -> Path:
        return self.processed / PROCESSED_MANIFESTS

    @property
    def cache_wikidata(self) -> Path:
        return self.cache / CACHE_WIKIDATA

    @property
    def cache_wikipedia(self) -> Path:
        return self.cache / CACHE_WIKIPEDIA

    def ensure(self) -> None:
        """Create the data root and standard sub-directories if needed."""
        self.path.mkdir(parents=True, exist_ok=True)
        subdirs = (
            self.raw,
            self.processed,
            self.logs,
            self.hf_cache,
            self.cache,
            self.processed_polygons,
            self.processed_articles,
            self.processed_links,
            self.processed_manifests,
            self.cache_wikidata,
            self.cache_wikipedia,
        )
        for sub in subdirs:
            sub.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Data root ready: %s", self.path)


def _is_inside(child: Path, parent: Path) -> bool:
    """True if ``child`` resolves to a path inside ``parent``."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_data_root(
    explicit: str | os.PathLike[str] | None = None,
    *,
    repo_root: Path,
) -> DataRoot:
    """Resolve the data root.

    Parameters
    ----------
    explicit:
        CLI-provided override (``--data-root``).
    repo_root:
        Path to this repository's root. Used to detect unsafe configurations
        where the data root accidentally points inside the source tree.

    Raises
    ------
    DataRootError
        If no path could be resolved, the resolved path is missing,
        or it resolves to inside the repository.
    """
    # Explicit or env-var candidates MUST point to an existing directory.
    # This avoids silently falling back to the recommended local path when
    # a user typo'd an explicit value.
    explicit_candidates: list[tuple[str, Path]] = []
    if explicit is not None:
        explicit_candidates.append(("explicit --data-root", Path(explicit).expanduser()))
    if (env := os.environ.get(ENV_VAR)) is not None:
        explicit_candidates.append((f"${ENV_VAR}", Path(env).expanduser()))

    if explicit_candidates:
        for source, candidate in explicit_candidates:
            if not candidate.exists():
                raise DataRootError(f"Data root {candidate} ({source}) does not exist.")
        # All explicit candidates exist; validate and pick the first.
        for source, candidate in explicit_candidates:
            if not candidate.is_dir():
                raise DataRootError(
                    f"Data root candidate {candidate} ({source}) is not a directory."
                )
            if _is_inside(candidate, repo_root):
                raise DataRootError(
                    f"Data root {candidate} ({source}) is inside the repository "
                    f"({repo_root}). Refusing to write artifacts into the repo."
                )
            return DataRoot(candidate)

    # Fallback: the recommended local path, but only if it actually exists.
    if DEFAULT_LOCAL_DATA_ROOT.exists():
        if not DEFAULT_LOCAL_DATA_ROOT.is_dir():
            raise DataRootError(
                f"Recommended local data root {DEFAULT_LOCAL_DATA_ROOT} is not a directory."
            )
        if _is_inside(DEFAULT_LOCAL_DATA_ROOT, repo_root):
            raise DataRootError(
                f"Recommended local data root {DEFAULT_LOCAL_DATA_ROOT} is inside "
                f"the repository ({repo_root}). Refusing to write artifacts into the repo."
            )
        return DataRoot(DEFAULT_LOCAL_DATA_ROOT)

    raise DataRootError(
        "Could not resolve a data root. Provide one via --data-root, set "
        f"the {ENV_VAR} environment variable, or mount "
        f"{DEFAULT_LOCAL_DATA_ROOT}."
    )
