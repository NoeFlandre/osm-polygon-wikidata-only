"""Deterministic polygon-statistics release for one published dataset.

The release path recomputes ``stats.json`` from every manifest-listed polygon
row, records the complete local inventory, and updates only release-owned
statistics sections in the dataset card. A real apply uses one Hub client for
the remote comparison, upload, and revision-bound verification.

Both published contracts use it:

* V1 -- ``NoeFlandre/osm-polygon-wikidata-only`` from ``<data-root>/processed``;
* V2 -- ``NoeFlandre/osm-polygon-wikidata-and-wikipedia`` from
  ``<data-root>/processed_v2``.

No clock or network is consulted for a dry run. A date is emitted only when
the caller supplies pinned ``generated_on`` metadata.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import DEFAULT_REPO_ID
from osm_polygon_wikidata_only.hf._polygon_geometry.validation import PolygonStatsInputError
from osm_polygon_wikidata_only.hf._uploader.operations import _build_hf_api
from osm_polygon_wikidata_only.hf._uploader.plan import PublicationOp, add_op
from osm_polygon_wikidata_only.hf._uploader.protocol import HfHub
from osm_polygon_wikidata_only.hf._uploader.token import resolve_hf_token
from osm_polygon_wikidata_only.hf.language_split_publication import LANGUAGE_CARD_HEADING
from osm_polygon_wikidata_only.hf.polygon_geometry_stats import (
    load_polygon_geometry_stats,
    stats_payload,
)
from osm_polygon_wikidata_only.hf.repo_layout import (
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE,
    REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE,
    REMOTE_POLYGON_STATS_FILE,
)
from osm_polygon_wikidata_only.hf.uploader import upload_files
from osm_polygon_wikidata_only.io.atomic import atomic_write_json, atomic_write_text
from osm_polygon_wikidata_only.v2.config import V2_REPO_ID

REMOTE_CARD_FILE = "README.md"
RELEASE_COMMIT_MESSAGE = "Publish dataset card and polygon statistics report"
_MANIFEST_RELATIVE_PATH = Path("manifests/processed_pbfs.json")
_STAGING_DIRNAME = "stats_release_snapshots"
_REMOTE_CACHE_DIRNAME = "remote_verification"
_HF_DATASET_COMMIT_URL = re.compile(
    r"https://huggingface\.co/datasets/[^/]+/[^/]+/commit/(?P<revision>[0-9a-f]{40})"
)
RELEASE_ASSET_FILES = (
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE,
    REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE,
)
# Sections preserved verbatim from the remote card: author-owned prose, plus
# sections owned by another publication path. "## Language partitions" is
# written by the language-split release and describes artifacts this release
# knows nothing about, so regenerating the card must not drop it. Every other
# section is data-derived and is always regenerated, so a released card can
# never carry a stale statistic next to a fresh one.
_PRESERVED_SECTION_HEADINGS = frozenset(
    {
        "## Citation",
        LANGUAGE_CARD_HEADING,
        "## License",
        "## Licensing",
        "## Reproducibility",
        "## Data sources & licenses",
        "## How to load",
    }
)


CardWriter = Callable[[Path], None]
# Renders the public coverage assets into a staging directory and returns the
# mapping of remote ``assets/*.png`` path to the freshly rendered local file.
AssetWriter = Callable[[Path], Mapping[str, Path]]


class StatsReleaseError(RuntimeError):
    """Raised when a statistics release cannot be planned or verified."""


@dataclass(frozen=True)
class ReleasedFile:
    """One published metadata file and its exact identity."""

    path_in_repo: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class StatsReleaseReport:
    """Evidence for one statistics release run."""

    repo_id: str
    processed_dir: str
    files: tuple[ReleasedFile, ...]
    polygon_files: int
    polygon_rows: int
    published: bool
    revision: str | None
    provenance: dict[str, Any] = field(default_factory=dict)
    committed: bool = False
    no_op: bool = False
    changed_files: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "changed_files": list(self.changed_files),
            "committed": self.committed,
            "files": [
                {
                    "path_in_repo": item.path_in_repo,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.files
            ],
            "no_op": self.no_op,
            "polygon_files": self.polygon_files,
            "polygon_rows": self.polygon_rows,
            "processed_dir": self.processed_dir,
            "provenance": self.provenance,
            "published": self.published,
            "repo_id": self.repo_id,
            "revision": self.revision,
        }


def _hub_revision(revision: str) -> str:
    match = _HF_DATASET_COMMIT_URL.fullmatch(revision)
    return match.group("revision") if match else revision


class RemoteVerifier(Protocol):
    """Confirm the released files exist at the uploaded revision."""

    def __call__(
        self,
        repo_id: str,
        files: tuple[ReleasedFile, ...],
        *,
        revision: str,
    ) -> str: ...


@dataclass(frozen=True)
class _RemoteState:
    revision: str | None
    contents: dict[str, bytes]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_if_changed(path: Path, text: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    atomic_write_text(path, text)


def _stage_report(
    stats: Any,
    destination: Path,
    provenance: dict[str, Any],
) -> None:
    payload = stats_payload(stats)
    payload["provenance"] = provenance
    atomic_write_json(destination, payload)


def _released_file(path: Path, path_in_repo: str) -> ReleasedFile:
    return ReleasedFile(
        path_in_repo=path_in_repo,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _require_exact_repo(confirm_repo: str, repo_id: str) -> None:
    if confirm_repo != repo_id:
        raise StatsReleaseError(
            f"repository confirmation must equal {repo_id!r} (got {confirm_repo!r})"
        )


def _require_canonical_repo(repo_id: str, expected: str) -> None:
    if repo_id != expected:
        raise StatsReleaseError(
            f"release target must use canonical repository {expected!r} (got {repo_id!r})"
        )


def _require_published_polygons(processed_dir: Path) -> None:
    polygons = processed_dir / "polygons"
    if not polygons.is_dir() or not any(polygons.glob("*.parquet")):
        raise StatsReleaseError(f"no published polygon files under {polygons}")


def _manifest_stem(key: str) -> str:
    return key.removesuffix(".pbf").removesuffix(".osm")


def _manifest_entries(raw: object, manifest_path: Path) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(raw, dict):
        raise StatsReleaseError(f"processed manifest is not an object: {manifest_path}")
    entries = raw.get("regions", raw)
    if not isinstance(entries, dict) or not entries:
        raise StatsReleaseError(f"processed manifest has no region entries: {manifest_path}")
    result: list[tuple[str, dict[str, Any]]] = []
    for key, value in entries.items():
        result.append(_manifest_entry(key, value, manifest_path))
    return sorted(result)


def _manifest_entry(
    key: object,
    value: object,
    manifest_path: Path,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise StatsReleaseError(f"processed manifest entry is not an object: {manifest_path}")
    return str(key), cast(dict[str, Any], value)


def _manifest_row_count(entry: Mapping[str, Any], manifest_path: Path) -> int:
    row_counts = entry.get("row_counts")
    value = (
        row_counts.get("polygons") if isinstance(row_counts, dict) else entry.get("polygon_count")
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise StatsReleaseError(f"processed manifest lacks polygon row count: {manifest_path}")
    return value


def _manifest_polygon_metadata(
    key: str,
    entry: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[str, str, int]:
    polygons_path = entry.get("polygons_path")
    expected_path = f"polygons/{_manifest_stem(key)}.parquet"
    if polygons_path != expected_path:
        raise StatsReleaseError(
            f"processed manifest entry {key!r} must point to {expected_path!r}: {manifest_path}"
        )
    source_pbf = entry.get("source_pbf") or key
    if not isinstance(source_pbf, str) or not source_pbf:
        raise StatsReleaseError(f"processed manifest entry lacks source_pbf: {manifest_path}")
    return expected_path, source_pbf, _manifest_row_count(entry, manifest_path)


def _read_manifest(processed_dir: Path) -> tuple[Path, bytes, object]:
    manifest_path = processed_dir / _MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise StatsReleaseError(f"complete processed manifest is required: {manifest_path}")
    try:
        raw_bytes = manifest_path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StatsReleaseError(
            f"cannot read processed manifest {manifest_path}: {error}"
        ) from error
    return manifest_path, raw_bytes, raw


def _manifest_revision(raw: object, *names: str) -> str | None:
    if not isinstance(raw, dict):
        return None
    raw_mapping = cast(dict[str, Any], raw)
    for name in names:
        revision = _manifest_revision_for_name(raw_mapping, name)
        if revision:
            return revision
    return None


def _manifest_revision_for_name(raw: Mapping[str, Any], name: str) -> str | None:
    direct = _non_empty_string(raw.get(name))
    return direct or _nested_manifest_revision(raw, name)


def _nested_manifest_revision(raw: Mapping[str, Any], name: str) -> str | None:
    nested = raw.get(name.removesuffix("_revision"))
    if not isinstance(nested, dict):
        return None
    return _non_empty_string(nested.get("revision"))


def _non_empty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _file_row_count(path: Path) -> int:
    try:
        return int(pq.read_metadata(path).num_rows)
    except Exception as error:
        raise StatsReleaseError(f"cannot read Parquet metadata for {path}: {error}") from error


def _inventory_rows(
    processed_dir: Path,
    manifest_path: Path,
    entries: Sequence[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    manifest_paths: set[str] = set()
    rows: list[dict[str, Any]] = []
    for key, entry in entries:
        relative_path, row = _inventory_row(processed_dir, manifest_path, key, entry)
        manifest_paths.add(relative_path)
        rows.append(row)
    actual_paths = {
        str(path.relative_to(processed_dir))
        for path in sorted((processed_dir / "polygons").glob("*.parquet"))
    }
    unlisted = sorted(actual_paths - manifest_paths)
    missing = sorted(manifest_paths - actual_paths)
    if unlisted:
        raise StatsReleaseError("polygon files absent from the manifest: " + ", ".join(unlisted))
    if missing:
        raise StatsReleaseError("manifest-listed polygon files are missing: " + ", ".join(missing))
    return rows


def _inventory_row(
    processed_dir: Path,
    manifest_path: Path,
    key: str,
    entry: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    relative_path, source_pbf, declared_rows = _manifest_polygon_metadata(key, entry, manifest_path)
    path = processed_dir / relative_path
    if not path.is_file():
        raise StatsReleaseError(f"manifest-listed polygon file is missing: {relative_path}")
    actual_rows = _file_row_count(path)
    if actual_rows != declared_rows:
        raise StatsReleaseError(
            f"polygon row count mismatch for {relative_path}: "
            f"manifest={declared_rows}, file={actual_rows}"
        )
    return relative_path, {
        "path": relative_path,
        "row_count": actual_rows,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "source_pbf": source_pbf,
    }


def _build_provenance(
    processed_dir: Path,
    *,
    source_revision: str | None,
    data_revision: str | None,
) -> dict[str, Any]:
    manifest_path, raw_bytes, raw = _read_manifest(processed_dir)
    entries = _manifest_entries(raw, manifest_path)
    polygons = _inventory_rows(processed_dir, manifest_path, entries)
    source_pbf_files = _source_pbf_files(polygons)
    resolved_source_revision, resolved_data_revision = _provenance_revisions(
        raw,
        source_revision=source_revision,
        data_revision=data_revision,
    )
    return {
        "contract_version": _contract_version(entries),
        "data": {"revision": resolved_data_revision},
        "data_revision": resolved_data_revision,
        "manifest": _manifest_provenance(raw_bytes, len(entries)),
        "polygons": polygons,
        "source": {"pbf_files": source_pbf_files, "revision": resolved_source_revision},
        "source_pbf_files": source_pbf_files,
        "source_revision": resolved_source_revision,
    }


def _source_pbf_files(polygons: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(item["source_pbf"]) for item in polygons})


def _provenance_revisions(
    raw: object,
    *,
    source_revision: str | None,
    data_revision: str | None,
) -> tuple[str | None, str | None]:
    return (
        source_revision or _manifest_revision(raw, "source_revision", "source"),
        data_revision or _manifest_revision(raw, "data_revision", "data"),
    )


def _contract_version(entries: Sequence[tuple[str, Mapping[str, Any]]]) -> str | list[str]:
    versions = sorted(
        {str(entry["contract_version"]) for _, entry in entries if entry.get("contract_version")}
    )
    return versions[0] if len(versions) == 1 else versions


def _manifest_provenance(raw_bytes: bytes, entry_count: int) -> dict[str, Any]:
    return {
        "entry_count": entry_count,
        "path": str(_MANIFEST_RELATIVE_PATH),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "size_bytes": len(raw_bytes),
    }


def _split_h2_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    matches = list(re.finditer(r"(?m)^## [^\n]*\n?", markdown))
    if not matches:
        return markdown, []
    prefix = markdown[: matches[0].start()]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        heading = match.group(0).rstrip("\r\n")
        sections.append((heading, markdown[match.start() : end]))
    return prefix, sections


def _merge_release_card(existing: str, generated: str) -> str:
    """Return the released card: regenerated data, preserved prose and header.

    The generated body is authoritative. Every section it renders is
    data-derived and replaces whatever the remote card held, and any
    section the generated card no longer renders is dropped rather than
    carried forward -- that carry-forward is what previously let a stale
    statistic survive beside a freshly computed one.

    Two things are preserved from the remote card. Author-owned prose
    sections listed in :data:`_PRESERVED_SECTION_HEADINGS` are appended
    when the generated card omits them. The YAML front matter is kept
    verbatim, because the Dataset Viewer ``configs:`` block there is
    owned by the publication and language-split paths -- a statistics
    release must never drop the published language partitions from the
    Viewer.
    """
    if not existing:
        return generated
    existing_front_matter, existing_body = _split_front_matter(existing)
    generated_front_matter, generated_body = _split_front_matter(generated)
    prefix, generated_sections = _split_h2_sections(generated_body)
    if not generated_sections:
        return generated
    sections = _released_sections(generated_sections, existing_body)
    front_matter = existing_front_matter or generated_front_matter
    return front_matter + _join_card_sections(prefix, sections)


def _released_sections(
    generated_sections: Sequence[tuple[str, str]],
    existing_body: str,
) -> list[str]:
    """Return the generated sections followed by any preserved prose."""
    _prefix, existing_sections = _split_h2_sections(existing_body)
    rendered = {heading for heading, _section in generated_sections}
    preserved = [
        section
        for heading, section in existing_sections
        if heading in _PRESERVED_SECTION_HEADINGS and heading not in rendered
    ]
    return [section for _heading, section in generated_sections] + preserved


def _split_front_matter(card: str) -> tuple[str, str]:
    """Split a card into its YAML front matter and the markdown body."""
    if not card.startswith("---\n"):
        return "", card
    end = card.find("\n---\n", 4)
    if end < 0:
        return "", card
    boundary = end + len("\n---\n")
    return card[:boundary], card[boundary:]


def _join_card_sections(prefix: str, sections: Sequence[str]) -> str:
    parts = [prefix.strip("\n")] if prefix.strip() else []
    parts.extend(section.rstrip("\n") for section in sections)
    return "\n\n".join(parts) + "\n"


def _remote_revision(client: Any, repo_id: str) -> str | None:
    repo_info = getattr(client, "repo_info", None)
    if not callable(repo_info):
        return None
    info = repo_info(repo_id, repo_type="dataset")
    revision = getattr(info, "sha", None)
    return str(revision) if revision else None


def _remote_entries(client: Any, repo_id: str, path: str, revision: str) -> list[Any]:
    get_paths_info = getattr(client, "get_paths_info", None)
    if not callable(get_paths_info):
        return []
    try:
        entries = get_paths_info(
            repo_id,
            paths=[path],
            revision=_hub_revision(revision),
            repo_type="dataset",
        )
    except TypeError:
        entries = get_paths_info(repo_id, paths=[path], repo_type="dataset")
    return list(entries)


def _download_remote_file(
    client: Any,
    repo_id: str,
    path: str,
    revision: str,
    *,
    cache_dir: Path | None,
) -> Path:
    download = getattr(client, "hf_hub_download", None)
    if not callable(download):
        raise StatsReleaseError("Hub client cannot download files for release verification")
    kwargs: dict[str, Any] = {
        "repo_type": "dataset",
        "revision": _hub_revision(revision),
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        return Path(download(repo_id, path, **kwargs))
    except TypeError:
        kwargs.pop("cache_dir", None)
        return Path(download(repo_id, path, **kwargs))


def _load_remote_state(
    client: Any,
    repo_id: str,
    paths: Sequence[str],
    *,
    cache_dir: Path,
) -> _RemoteState:
    try:
        revision = _remote_revision(client, repo_id)
    except Exception:
        return _RemoteState(None, {})
    if not revision:
        return _RemoteState(None, {})
    contents: dict[str, bytes] = {}
    for path in paths:
        content = _download_remote_content(
            client,
            repo_id,
            path,
            revision,
            cache_dir=cache_dir,
        )
        if content is not None:
            contents[path] = content
    return _RemoteState(revision, contents)


def _download_remote_content(
    client: Any,
    repo_id: str,
    path: str,
    revision: str,
    *,
    cache_dir: Path,
) -> bytes | None:
    try:
        if not _remote_entries(client, repo_id, path, revision):
            return None
        local_path = _download_remote_file(
            client,
            repo_id,
            path,
            revision,
            cache_dir=cache_dir,
        )
        return local_path.read_bytes()
    except (OSError, StatsReleaseError):
        return None


def _changed_paths(
    remote: _RemoteState,
    files: Sequence[ReleasedFile],
    local_paths: Mapping[str, Path],
) -> tuple[str, ...]:
    if not remote.revision:
        return tuple(item.path_in_repo for item in files)
    return tuple(
        item.path_in_repo
        for item in files
        if remote.contents.get(item.path_in_repo) != local_paths[item.path_in_repo].read_bytes()
    )


def _client_for_release(hub: HfHub | None, token: str | None) -> Any:
    return hub or _build_hf_api(resolve_hf_token(token))


def _invoke_custom_verifier(
    verifier: RemoteVerifier,
    repo_id: str,
    files: tuple[ReleasedFile, ...],
    revision: str,
) -> str:
    legacy_verifier = cast(Callable[[str, tuple[ReleasedFile, ...]], str], verifier)
    try:
        parameters = inspect.signature(verifier).parameters.values()
    except (TypeError, ValueError):
        return legacy_verifier(repo_id, files)
    accepts_keyword = any(
        parameter.name == "revision" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_keyword:
        return verifier(repo_id, files, revision=revision)
    return legacy_verifier(repo_id, files)


def _verify_uploaded_files(
    repo_id: str,
    files: tuple[ReleasedFile, ...],
    *,
    upload_revision: str,
    verifier: RemoteVerifier | None,
    client: Any,
    token: str | None,
    cache_dir: Path,
) -> str:
    if verifier is None:
        verified_revision = default_remote_verifier(
            repo_id,
            files,
            revision=upload_revision,
            hub=client,
            token=token,
            cache_dir=cache_dir,
        )
    else:
        verified_revision = _invoke_custom_verifier(
            verifier,
            repo_id,
            files,
            upload_revision,
        )
    if not verified_revision:
        raise StatsReleaseError("remote verification returned an empty revision")
    if verified_revision != upload_revision:
        raise StatsReleaseError(
            "remote verification was not bound to the uploaded commit: "
            f"uploaded={upload_revision}, verified={verified_revision}"
        )
    return verified_revision


def _remote_revision_for_verifier(
    client: Any,
    repo_id: str,
    revision: str | None,
) -> str:
    selected = revision or _remote_revision(client, repo_id)
    if not selected:
        raise StatsReleaseError(f"hub repository {repo_id} returned an empty revision")
    return selected


def _verify_remote_file(
    client: Any,
    repo_id: str,
    item: ReleasedFile,
    revision: str,
    *,
    cache_dir: Path | None,
) -> None:
    entries = _remote_entries(client, repo_id, item.path_in_repo, revision)
    entry = _find_remote_entry(entries, item.path_in_repo)
    if entry is None:
        raise StatsReleaseError(f"remote file missing in revision {revision}: {item.path_in_repo}")
    _verify_remote_size(entry, item)
    _verify_remote_hash(
        client,
        repo_id,
        item,
        revision,
        cache_dir=cache_dir,
    )


def _find_remote_entry(entries: Sequence[Any], path: str) -> Any | None:
    return next(
        (candidate for candidate in entries if getattr(candidate, "path", path) == path),
        None,
    )


def _verify_remote_size(entry: Any, item: ReleasedFile) -> None:
    size = getattr(entry, "size", None)
    if size is not None and int(size) != item.size_bytes:
        raise StatsReleaseError(
            f"remote size mismatch for {item.path_in_repo}: local={item.size_bytes}, remote={size}"
        )


def _verify_remote_hash(
    client: Any,
    repo_id: str,
    item: ReleasedFile,
    revision: str,
    *,
    cache_dir: Path | None,
) -> None:
    local_path = _download_remote_file(
        client,
        repo_id,
        item.path_in_repo,
        revision,
        cache_dir=cache_dir,
    )
    if _sha256(local_path) != item.sha256:
        raise StatsReleaseError(f"remote SHA-256 mismatch for {item.path_in_repo}")


def default_remote_verifier(
    repo_id: str,
    files: tuple[ReleasedFile, ...],
    *,
    revision: str | None = None,
    hub: HfHub | None = None,
    token: str | None = None,
    cache_dir: Path | None = None,
) -> str:
    """Confirm each released file's identity at one exact Hub revision."""
    client = _client_for_release(hub, token)
    selected_revision = _remote_revision_for_verifier(client, repo_id, revision)
    for item in files:
        _verify_remote_file(
            client,
            repo_id,
            item,
            selected_revision,
            cache_dir=cache_dir,
        )
    return selected_revision


def release_polygon_stats(
    *,
    processed_dir: Path,
    staging_dir: Path,
    repo_id: str,
    confirm_repo: str,
    card_writer: CardWriter,
    asset_writer: AssetWriter | None = None,
    apply: bool = False,
    hub: HfHub | None = None,
    verifier: RemoteVerifier | None = None,
    token: str | None = None,
    source_revision: str | None = None,
    data_revision: str | None = None,
) -> StatsReleaseReport:
    """Recompute, publish, and verify the card, report, and coverage assets."""
    _require_exact_repo(confirm_repo, repo_id)
    _require_published_polygons(processed_dir)
    provenance = _build_provenance(
        processed_dir,
        source_revision=source_revision,
        data_revision=data_revision,
    )
    stats = _load_release_stats(processed_dir)
    client, remote = _release_remote_state(
        apply,
        hub=hub,
        token=token,
        repo_id=repo_id,
        staging_dir=staging_dir,
        asset_writer=asset_writer,
    )
    files, local_paths = _prepare_release_files(
        stats,
        provenance,
        staging_dir=staging_dir,
        card_writer=card_writer,
        asset_writer=asset_writer,
        remote=remote,
    )
    if not apply:
        return _release_report(
            repo_id=repo_id,
            processed_dir=processed_dir,
            stats=stats,
            files=files,
            provenance=provenance,
            published=False,
        )
    revision, committed, no_op, changed_files = _apply_release(
        repo_id,
        files,
        remote=remote,
        client=client,
        token=token,
        verifier=verifier,
        staging_dir=staging_dir,
        local_paths=local_paths,
    )
    return _release_report(
        repo_id=repo_id,
        processed_dir=processed_dir,
        stats=stats,
        files=files,
        provenance=provenance,
        published=True,
        revision=revision,
        committed=committed,
        no_op=no_op,
        changed_files=changed_files,
    )


def _load_release_stats(processed_dir: Path) -> Any:
    try:
        return load_polygon_geometry_stats(processed_dir)
    except (OSError, ValueError, PolygonStatsInputError) as error:
        raise StatsReleaseError(str(error)) from error


def _release_remote_state(
    apply: bool,
    *,
    hub: HfHub | None,
    token: str | None,
    repo_id: str,
    staging_dir: Path,
    asset_writer: AssetWriter | None = None,
) -> tuple[Any, _RemoteState]:
    if not apply:
        return None, _RemoteState(None, {})
    client = _client_for_release(hub, token)
    tracked = (REMOTE_CARD_FILE, REMOTE_POLYGON_STATS_FILE)
    if asset_writer is not None:
        tracked = (*tracked, *RELEASE_ASSET_FILES)
    remote = _load_remote_state(
        client,
        repo_id,
        tracked,
        cache_dir=staging_dir / _REMOTE_CACHE_DIRNAME,
    )
    return client, remote


def _prepare_release_files(
    stats: Any,
    provenance: dict[str, Any],
    *,
    staging_dir: Path,
    card_writer: CardWriter,
    asset_writer: AssetWriter | None,
    remote: _RemoteState,
) -> tuple[tuple[ReleasedFile, ...], dict[str, Path]]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    report_path = staging_dir / "stats.json"
    card_path = staging_dir / "README.md"
    _stage_report(stats, report_path, provenance)
    card_writer(card_path)
    _merge_remote_card(card_path, remote)
    local_paths = {
        REMOTE_CARD_FILE: card_path,
        REMOTE_POLYGON_STATS_FILE: report_path,
    }
    local_paths.update(_stage_release_assets(staging_dir, asset_writer))
    _require_consistent_text_coverage(card_path)
    return (
        tuple(_released_file(local_paths[path], path) for path in local_paths),
        local_paths,
    )


def _stage_release_assets(
    staging_dir: Path,
    asset_writer: AssetWriter | None,
) -> dict[str, Path]:
    """Render the public coverage assets that accompany this card.

    The card numbers and the map captions are produced from the same local
    Parquet tables in the same run, so a release can never ship a refreshed
    statistic beside a map rendered from an older snapshot.
    """
    if asset_writer is None:
        return {}
    rendered = dict(asset_writer(staging_dir / "assets"))
    _require_complete_assets(rendered)
    return rendered


def _require_complete_assets(rendered: Mapping[str, Path]) -> None:
    _require_every_asset_planned(rendered)
    _require_every_asset_written(rendered)


def _require_every_asset_planned(rendered: Mapping[str, Path]) -> None:
    missing = sorted(path for path in RELEASE_ASSET_FILES if path not in rendered)
    if missing:
        raise StatsReleaseError("coverage asset writer did not produce: " + ", ".join(missing))


def _require_every_asset_written(rendered: Mapping[str, Path]) -> None:
    unrendered = sorted(path for path, local in rendered.items() if not local.is_file())
    if unrendered:
        raise StatsReleaseError(f"coverage asset {unrendered[0]} was not rendered")


def _merge_remote_card(card_path: Path, remote: _RemoteState) -> None:
    if REMOTE_CARD_FILE not in remote.contents:
        return
    try:
        existing = remote.contents[REMOTE_CARD_FILE].decode("utf-8")
        generated = card_path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise StatsReleaseError(f"dataset card is not valid UTF-8: {error}") from error
    _write_text_if_changed(card_path, _merge_release_card(existing, generated))


def _apply_release(
    repo_id: str,
    files: tuple[ReleasedFile, ...],
    *,
    remote: _RemoteState,
    client: Any,
    token: str | None,
    verifier: RemoteVerifier | None,
    staging_dir: Path,
    local_paths: Mapping[str, Path],
) -> tuple[str | None, bool, bool, tuple[str, ...]]:
    changed_files = _changed_paths(remote, files, local_paths)
    if remote.revision and not changed_files:
        return remote.revision, False, True, changed_files
    ops: list[PublicationOp] = [
        add_op(local_paths[path], path_in_repo=path) for path in changed_files
    ]
    upload_revision = upload_files(
        repo_id,
        ops=ops,
        hub=client,
        token=token,
        commit_message=RELEASE_COMMIT_MESSAGE,
    )
    if not upload_revision:
        raise StatsReleaseError("Hub upload returned an empty revision")
    revision = _verify_uploaded_files(
        repo_id,
        files,
        upload_revision=upload_revision,
        verifier=verifier,
        client=client,
        token=token,
        cache_dir=staging_dir / _REMOTE_CACHE_DIRNAME,
    )
    return revision, True, False, changed_files


_HEADLINE_TEXT_COVERAGE = re.compile(
    r"(?:\|\s*Polygons with successful non-empty text \(unique OSM identities\)\s*\|"
    r"|-\s*\*\*Polygons with non-empty Wikipedia or Wikivoyage text:\*\*)"
    r"\s*([\d,]+)"
)
_CONTINENT_TABLE_ROW = re.compile(r"^\|([^|\n]+)\|([^\n]*)\|\s*$", re.MULTILINE)


def _require_consistent_text_coverage(card_path: Path) -> None:
    """Fail closed when the card states two different text-coverage totals.

    The headline figure and the per-continent breakdown are computed by
    separate renderers. They must agree, otherwise the published card
    contradicts itself -- and so does the map caption rendered from the
    same snapshot as the headline.
    """
    card = card_path.read_text(encoding="utf-8")
    headline = _headline_text_coverage(card)
    continent_total = _continent_text_coverage_total(card)
    if headline is None or continent_total is None:
        return
    if headline != continent_total:
        raise StatsReleaseError(
            "dataset card states inconsistent text-coverage totals: headline "
            f"{headline:,} but the continent table sums to {continent_total:,}; "
            "no release was published"
        )


def _headline_text_coverage(card: str) -> int | None:
    match = _HEADLINE_TEXT_COVERAGE.search(card)
    return int(match.group(1).replace(",", "")) if match else None


def _continent_text_coverage_total(card: str) -> int | None:
    section = _continent_section(card)
    if section is None:
        return None
    total = 0
    counted = False
    for line in section.splitlines():
        value = _continent_row_combined(line)
        if value is not None:
            total += value
            counted = True
    return total if counted else None


def _continent_section(card: str) -> str | None:
    heading = "## Geographic distribution by continent"
    start = card.find(heading)
    if start < 0:
        return None
    end = card.find("\n## ", start + len(heading))
    return card[start:] if end < 0 else card[start:end]


def _continent_row_combined(line: str) -> int | None:
    """Return the combined text-covered count from one continent data row."""
    cells = _continent_data_cells(line)
    return _parse_grouped_int(cells[5]) if cells else None


def _continent_data_cells(line: str) -> list[str] | None:
    """Return the cells of a continent data row, or None for any other line."""
    if not line.startswith("|"):
        return None
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) != 7 or not cells[-1].endswith("%"):
        return None
    return cells


def _parse_grouped_int(value: str) -> int | None:
    try:
        return int(value.replace(",", ""))
    except ValueError:
        return None


def _release_report(
    *,
    repo_id: str,
    processed_dir: Path,
    stats: Any,
    files: tuple[ReleasedFile, ...],
    provenance: dict[str, Any],
    published: bool,
    revision: str | None = None,
    committed: bool = False,
    no_op: bool = False,
    changed_files: tuple[str, ...] = (),
) -> StatsReleaseReport:
    return StatsReleaseReport(
        repo_id=repo_id,
        processed_dir=str(processed_dir),
        files=files,
        polygon_files=stats.file_count,
        polygon_rows=stats.polygon_count,
        published=published,
        revision=revision,
        provenance=provenance,
        committed=committed,
        no_op=no_op,
        changed_files=changed_files,
    )


def release_v1_polygon_stats(
    data_root: DataRoot,
    *,
    confirm_repo: str,
    repo_id: str = DEFAULT_REPO_ID,
    apply: bool = False,
    hub: HfHub | None = None,
    verifier: RemoteVerifier | None = None,
    token: str | None = None,
    source_revision: str | None = None,
    data_revision: str | None = None,
    generated_on: str | None = None,
) -> StatsReleaseReport:
    """Release the V1 Wikidata-only card and statistics report."""
    _require_canonical_repo(repo_id, DEFAULT_REPO_ID)
    from osm_polygon_wikidata_only.hf.publication import (
        _write_readme_snapshot,
        refresh_coverage_assets,
    )

    def write_card(destination: Path) -> None:
        _write_readme_snapshot(
            data_root,
            repo_id,
            destination,
            generated_on=generated_on,
        )

    def write_assets(destination: Path) -> Mapping[str, Path]:
        coverage, presence, density = refresh_coverage_assets(
            data_root=data_root,
            snapshot_stem="release",
            snapshots_dir=destination,
            world_land_warning=None,
        )
        return {
            REMOTE_COVERAGE_MAP_FILE: coverage,
            REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE: presence,
            REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE: density,
        }

    return release_polygon_stats(
        processed_dir=data_root.processed,
        staging_dir=data_root.cache / _STAGING_DIRNAME / "v1",
        repo_id=repo_id,
        confirm_repo=confirm_repo,
        card_writer=write_card,
        asset_writer=write_assets,
        apply=apply,
        hub=hub,
        verifier=verifier,
        token=token,
        source_revision=source_revision,
        data_revision=data_revision,
    )


def release_v2_polygon_stats(
    data_root: DataRoot,
    *,
    confirm_repo: str,
    repo_id: str = V2_REPO_ID,
    apply: bool = False,
    hub: HfHub | None = None,
    verifier: RemoteVerifier | None = None,
    token: str | None = None,
    source_revision: str | None = None,
    data_revision: str | None = None,
    generated_on: str | None = None,
) -> StatsReleaseReport:
    """Release the V2 Wikidata + Wikipedia card and statistics report."""
    _require_canonical_repo(repo_id, V2_REPO_ID)
    from osm_polygon_wikidata_only.v2.card import render_v2_card
    from osm_polygon_wikidata_only.v2.maps import generate_v2_map_assets

    processed_v2 = data_root.processed_v2

    def write_card(destination: Path) -> None:
        _write_text_if_changed(
            destination,
            render_v2_card(
                processed_v2,
                v1_processed=data_root.processed,
                generated_on=generated_on,
            ),
        )

    def write_assets(destination: Path) -> Mapping[str, Path]:
        coverage, presence, density = generate_v2_map_assets(
            processed_v2,
            destination,
            v1_processed=data_root.processed,
            land_cache_dir=data_root.cache,
        )
        return {
            REMOTE_COVERAGE_MAP_FILE: coverage,
            REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE: presence,
            REMOTE_GEOGRAPHIC_TEXT_DENSITY_FILE: density,
        }

    return release_polygon_stats(
        processed_dir=processed_v2,
        staging_dir=data_root.cache / _STAGING_DIRNAME / "v2",
        repo_id=repo_id,
        confirm_repo=confirm_repo,
        card_writer=write_card,
        asset_writer=write_assets,
        apply=apply,
        hub=hub,
        verifier=verifier,
        token=token,
        source_revision=source_revision,
        data_revision=data_revision,
    )


__all__ = [
    "RELEASE_COMMIT_MESSAGE",
    "REMOTE_CARD_FILE",
    "ReleasedFile",
    "StatsReleaseError",
    "StatsReleaseReport",
    "default_remote_verifier",
    "release_polygon_stats",
    "release_v1_polygon_stats",
    "release_v2_polygon_stats",
]
