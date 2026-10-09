"""Data contracts for planning, publishing, and reporting language partitions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from osm_polygon_wikidata_only.hf.language_split_release import LanguageSplitVersion
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps


@dataclass(frozen=True, slots=True)
class LanguagePublishedFile:
    """One local file and its exact remote publication identity."""

    local_path: Path
    path_in_repo: str
    size_bytes: int | None
    sha256: str | None
    _git_sha1: str | None = field(default=None, init=False, repr=False, compare=False)

    def git_sha1(self) -> str:
        """Return the git blob SHA-1 of the local file, computed once."""
        if self._git_sha1 is None:
            digest = _git_blob_sha1(self.local_path)
            object.__setattr__(self, "_git_sha1", digest)
            return digest
        return self._git_sha1

    def to_dict(self) -> dict[str, object]:
        """Return deterministic evidence for this file."""
        return {
            "local_path": str(self.local_path),
            "path_in_repo": self.path_in_repo,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class LanguagePublicationPlan:
    """Complete local publication plan for one dataset contract."""

    version: LanguageSplitVersion
    repo_id: str
    processed_root: Path
    output_root: Path
    manifest_path: Path
    manifest_remote_path: str
    languages: tuple[str, ...]
    configurations: tuple[str, ...]
    files: tuple[LanguagePublishedFile, ...]
    configuration_languages: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible plan evidence."""
        return {
            "dataset_version": self.version.value,
            "repo_id": self.repo_id,
            "processed_root": str(self.processed_root),
            "output_root": str(self.output_root),
            "manifest_path": str(self.manifest_path),
            "manifest_remote_path": self.manifest_remote_path,
            "languages": list(self.languages),
            "configurations": list(self.configurations),
            "configuration_languages": [
                {"configuration": configuration, "languages": list(languages)}
                for configuration, languages in self.configuration_languages
            ],
            "files": [item.to_dict() for item in self.files],
        }


@dataclass(frozen=True, slots=True)
class LanguagePublicationReport:
    """Evidence for one dry-run or one remote publication."""

    version: LanguageSplitVersion
    repo_id: str
    dry_run: bool
    published: bool
    committed: bool
    no_op: bool
    revision: str | None
    files: tuple[LanguagePublishedFile, ...]
    changed_files: tuple[str, ...] = ()
    stale_files: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Return deterministic release evidence."""
        return {
            "changed_files": list(self.changed_files),
            "committed": self.committed,
            "dataset_version": self.version.value,
            "dry_run": self.dry_run,
            "files": [item.to_dict() for item in self.files],
            "no_op": self.no_op,
            "published": self.published,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "stale_files": list(self.stale_files),
        }


@dataclass(frozen=True, slots=True)
class LanguagePublicationResult:
    """Combined evidence for the selected V1/V2 publication set."""

    reports: tuple[LanguagePublicationReport, ...]

    def to_payload(self) -> dict[str, object]:
        """Return deterministic JSON-compatible evidence."""
        return {
            "command": "publish-language-splits",
            "reports": [report.to_payload() for report in self.reports],
        }

    def to_json(self) -> str:
        """Serialize the publication evidence."""
        return json_dumps(self.to_payload())


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
