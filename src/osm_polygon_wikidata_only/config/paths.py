"""Resolve the configurable local data root used by the pipeline.

Generated dataset artifacts are published on Hugging Face. Local PBF inputs,
intermediate artifacts, and caches live outside the source working tree under
an operator-selected data root.

Resolution precedence (highest first):

1. Explicit value passed to :class:`DataRoot` (typically from ``--data-root``).
2. ``OSM_POLYGON_DATA_ROOT`` environment variable.
If neither source yields a usable path, a clear error is raised.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


ENV_VAR = "OSM_POLYGON_DATA_ROOT"

# Conventional top-level sub-directories under the data root.
SUBDIR_RAW = "raw"
SUBDIR_PROCESSED = "processed"
SUBDIR_PROCESSED_V2 = "processed_v2"
SUBDIR_LOGS = "logs"
SUBDIR_HF_CACHE = "hf_cache"
SUBDIR_CACHE = "cache"
SUBDIR_CACHE_V2 = "v2"

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
    """Resolved local data root for the pipeline."""

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
    def processed_v2(self) -> Path:
        """Return the isolated V2 artifact directory."""
        return self.sub(SUBDIR_PROCESSED_V2)

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
    def v2_cache(self) -> Path:
        """Return the isolated V2 cache directory."""
        return self.cache / SUBDIR_CACHE_V2

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
            self.processed_v2,
            self.logs,
            self.hf_cache,
            self.cache,
            self.v2_cache,
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
    explicit_candidates = _explicit_candidates(explicit)

    if explicit_candidates:
        _require_candidates_exist(explicit_candidates)
        return _validate_candidates(explicit_candidates, repo_root)

    raise DataRootError(
        "Could not resolve a data root. Provide one via --data-root, set "
        f"the {ENV_VAR} environment variable, and keep it outside the source repository."
    )


def _explicit_candidates(
    explicit: str | os.PathLike[str] | None,
) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    if explicit is not None:
        candidates.append(("explicit --data-root", Path(explicit).expanduser()))
    env = os.environ.get(ENV_VAR)
    if env is not None:
        candidates.append((f"${ENV_VAR}", Path(env).expanduser()))
    return candidates


def _require_candidates_exist(candidates: list[tuple[str, Path]]) -> None:
    for source, candidate in candidates:
        if not candidate.exists():
            raise DataRootError(f"Data root {candidate} ({source}) does not exist.")


def _validate_candidates(
    candidates: list[tuple[str, Path]],
    repo_root: Path,
) -> DataRoot:
    for source, candidate in candidates:
        _validate_candidate(candidate, source, repo_root)
        return DataRoot(candidate)
    raise AssertionError("at least one data-root candidate is required")


def _validate_candidate(candidate: Path, source: str, repo_root: Path) -> None:
    if not candidate.is_dir():
        raise DataRootError(f"Data root candidate {candidate} ({source}) is not a directory.")
    if _is_inside(candidate, repo_root):
        raise DataRootError(
            f"Data root {candidate} ({source}) is inside the repository "
            f"({repo_root}). Refusing to write artifacts into the repo."
        )
