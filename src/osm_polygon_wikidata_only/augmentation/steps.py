"""Decomposed augmentation pipeline mechanics.

The orchestrator (:mod:`augmentation.orchestrator`) owns the policy:
phase ordering, progress transitions, sidecar paths, the
core-hash drift check, the manifest write, and the thread-pool
lifecycle. This module owns the focused, narrow-responsibility
mechanics that the orchestrator calls.

Responsibility per helper (kept stable across the decomposition):

* :func:`load_core_inputs` -- raise if core artifacts are missing;
  otherwise return a :class:`CoreInputs` record carrying the
  documents, sorted QIDs and core paths/hashes needed for the
  post-write drift check. ``core_hashes`` is captured *before*
  any processing.
* :func:`resolve_entities` -- single ``entities(...)`` call with
  ``props='sitelinks|claims'`` and progress bookkeeping.
* :func:`fetch_wikivoyage_documents` -- concurrent ``wikivoyage_document``
  fetch using a caller-supplied executor; sort by ``document_id``;
  drop ``None`` results; advance progress once per *attempted* link.
* :func:`fetch_document_sections` -- concurrent ``parse_html`` for
  every document using the caller-supplied executor; partition
  sections into ``wikipedia`` / ``wikivoyage``; sort each bucket by
  ``(document_id, section_index)``.
* :func:`build_wikidata_facts` -- collect ``FACT_PROPERTIES`` plus all
  fact value entity ids, issue one ``entities(..., props='labels')``
  fetch, produce one :class:`WikidataFact` per claim in
  ``FACT_PROPERTIES``; sort by ``fact_id``.
* :func:`write_sidecars` -- write the five sidecar parquets in the
  canonical order using the local :func:`_write` (atomic temp +
  ``os.replace``); advance progress once per file.
* :func:`update_augmentation_manifest` -- atomic merge of a single
  stem entry into the existing ``augmentation_manifest.json``
  preserving other regions; create parent directory on first write.

Concurrency, batching, sorting and progress counters are preserved
exactly. Executor selection and lifecycle stay with the
orchestrator; helpers do *not* choose worker counts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import Executor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.models import (
    Document,
    Section,
    WikidataFact,
    document_from_article_row,
)
from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.augmentation.sections import parse_sections
from osm_polygon_wikidata_only.augmentation.wikimedia import (
    FACT_PROPERTIES,
    discover_wikivoyage_sitelinks,
    normalize_facts,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    build_wikipedia_document_table,
    wikipedia_document_schema,
)
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import ARTICLE_COLUMNS, article_schema
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps


class AugmentationClient(Protocol):
    def entities(self, qids: list[str] | set[str], *, props: str) -> dict[str, dict[str, Any]]: ...
    def parse_html(self, project: str, language: str, revision_id: int) -> str: ...
    def wikivoyage_document(
        self, qid: str, language: str, site: str, title: str
    ) -> Document | None: ...


CONTRACT_VERSION = "text-sidecars-v1"


# ---------------------------------------------------------------------------
# Wikipedia source-path selection
# ---------------------------------------------------------------------------


def _validate_source_stem(stem: str) -> None:
    """Reject empty stems, path separators, and traversal-like names."""
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError(f"Invalid Wikipedia source stem: {stem!r}")


@dataclass(frozen=True, slots=True)
class WikipediaSourcePaths:
    """Canonical and legacy local paths for one stem's Wikipedia data.

    During migration both files may coexist. The *legacy* article is the
    read source of truth until :func:`~osm_polygon_wikidata_only.augmentation.wikipedia_retirement.finalize_local_retirement`
    deletes it; afterwards the *canonical* document is the only source.

    Consumers must choose the appropriate policy explicitly:
    :func:`read_source_path` for augmentation input loading, or
    :func:`~osm_polygon_wikidata_only.augmentation.orchestrator.augmentation_is_current`
    for manifest-aware validation.
    """

    canonical: Path
    legacy: Path

    @property
    def either_exists(self) -> bool:
        """True when either source file is present."""
        return self.legacy.exists() or self.canonical.exists()


def wikipedia_source_paths(data_root: DataRoot, stem: str) -> WikipediaSourcePaths:
    """Return the canonical-document and legacy-article paths for *stem*.

    Side-effect-free: no directories are created and no files are read.
    Raises :class:`ValueError` for malformed or traversal-like stems.
    """
    _validate_source_stem(stem)
    return WikipediaSourcePaths(
        canonical=data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet",
        legacy=data_root.processed_articles / f"{stem}.parquet",
    )


def read_source_path(data_root: DataRoot, stem: str) -> Path:
    """Return the legacy article path if present, else the canonical document path.

    Used by augmentation input loading. During migration both files may
    coexist; the legacy article is read as the source of truth.
    After retirement the canonical document is the only remaining source.
    The returned path may not yet exist for a never-augmented stem.
    """
    sources = wikipedia_source_paths(data_root, stem)
    return sources.legacy if sources.legacy.exists() else sources.canonical


def sha256_file(path: Path) -> str:
    """SHA-256 of *path* streamed in 1 MiB chunks.

    Single shared implementation used for the initial core hash
    capture, the post-write drift check inside the orchestrator, and
    the resumability check in :func:`augmentation_is_current`.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CoreInputs:
    """Snapshot of the core artifacts captured before processing.

    The orchestrator only needs the documents, the sorted QIDs, and
    enough information to rehash the core paths after writing. The
    polygon rows are intentionally not exposed.
    """

    wikipedia_documents: tuple[Document, ...]
    qids: tuple[str, ...]
    core_paths: tuple[Path, Path]
    core_hashes: dict[str, str]


def _write_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(raw_tmp)
    os.close(fd)
    try:
        pq.write_table(table, tmp_path, compression="snappy")  # type: ignore[no-untyped-call]
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_atomic_from_rows(
    path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...], schema: pa.Schema
) -> None:
    normalized = [{column: row.get(column) for column in columns} for row in rows]
    table = pa.Table.from_pylist(normalized, schema=schema)
    _write_atomic(path, table)


def _normalized_article_table(path: Path) -> pa.Table:
    """Read articles and fill only historically absent nullable columns."""
    source = pq.read_table(path)  # type: ignore[no-untyped-call]
    unknown = set(source.column_names) - set(ARTICLE_COLUMNS)
    if unknown:
        raise ValueError(f"Core article Parquet has unknown columns at {path}: {sorted(unknown)}")
    schema = article_schema()
    rows = []
    for source_row in source.to_pylist():
        row: dict[str, Any] = {}
        for column in ARTICLE_COLUMNS:
            if column in source_row:
                row[column] = source_row[column]
            else:
                row[column] = "" if pa.types.is_string(schema.field(column).type) else None
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=schema)


def _label_maps(entities: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for qid, entity in entities.items():
        labels = entity.get("labels") or {}
        out[qid] = {
            str(language): str(value.get("value", ""))
            for language, value in labels.items()
            if isinstance(value, dict) and value.get("value")
        }
    return out


# ---------------------------------------------------------------------------
# load_core_inputs
# ---------------------------------------------------------------------------


def load_core_inputs(data_root: DataRoot, stem: str) -> CoreInputs:
    """Load the core artifacts for *stem* and capture pre-processing
    hashes. Raises :class:`FileNotFoundError` if either core parquet
    is missing."""
    articles_path = read_source_path(data_root, stem)
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    if not articles_path.exists() or not polygons_path.exists():
        raise FileNotFoundError(f"Core region is incomplete: {stem}")
    core_paths = (articles_path, polygons_path)
    core_hashes = {str(path): sha256_file(path) for path in core_paths}

    source_table = pq.read_table(articles_path)  # type: ignore[no-untyped-call]
    if source_table.schema.equals(wikipedia_document_schema(), check_metadata=True):
        wikipedia_documents = [
            Document(**{column: row[column] for column in DOCUMENT_COLUMNS})
            for row in source_table.to_pylist()
        ]
    else:
        article_table = _normalized_article_table(articles_path)
        wikipedia_documents = [document_from_article_row(row) for row in article_table.to_pylist()]
    polygon_rows = pq.read_table(polygons_path, columns=["wikidata"]).to_pylist()  # type: ignore[no-untyped-call]
    wikipedia_documents.sort(key=lambda row: row.document_id)
    qids = sorted({str(row["wikidata"]) for row in polygon_rows if row.get("wikidata")})
    return CoreInputs(
        wikipedia_documents=tuple(wikipedia_documents),
        qids=tuple(qids),
        core_paths=core_paths,
        core_hashes=core_hashes,
    )


# ---------------------------------------------------------------------------
# resolve_entities
# ---------------------------------------------------------------------------


def resolve_entities(
    client: AugmentationClient,
    qids: list[str],
    *,
    progress: AugmentationProgress,
) -> dict[str, dict[str, Any]]:
    """Fetch the ``sitelinks|claims`` projection for *qids* and report
    progress under the ``Wikidata entities`` phase."""
    progress.start("Wikidata entities", total=len(qids))
    entities = client.entities(qids, props="sitelinks|claims")
    progress.complete()
    return entities


# ---------------------------------------------------------------------------
# fetch_wikivoyage_documents
# ---------------------------------------------------------------------------


def fetch_wikivoyage_documents(
    client: AugmentationClient,
    *,
    entities: dict[str, dict[str, Any]],
    progress: AugmentationProgress,
    executor: Executor,
) -> list[Document]:
    """Concurrently fetch Wikivoyage documents on *executor*, drop
    ``None`` results, and sort by ``document_id``. Progress advances
    once per *attempted* link (matching legacy semantics). The
    caller selects and owns the executor (the orchestrator uses
    ``max_workers=8``)."""
    voyage_links = [
        (qid, language, site, title)
        for qid, entity in sorted(entities.items())
        for language, site, title in discover_wikivoyage_sitelinks(entity)
    ]
    progress.start("Wikivoyage documents", total=len(voyage_links))

    def fetch_voyage(item: tuple[str, str, str, str]) -> Document | None:
        try:
            return client.wikivoyage_document(*item)
        finally:
            progress.advance()

    voyage_documents = [
        document for document in executor.map(fetch_voyage, voyage_links) if document is not None
    ]
    voyage_documents.sort(key=lambda row: row.document_id)
    return voyage_documents


# ---------------------------------------------------------------------------
# fetch_document_sections
# ---------------------------------------------------------------------------


def fetch_document_sections(
    client: AugmentationClient,
    *,
    documents: list[Document],
    progress: AugmentationProgress,
    executor: Executor,
) -> dict[str, list[Section]]:
    """Concurrently fetch article HTML on *executor*, partition the
    resulting sections by project, and sort each bucket by
    ``(document_id, section_index)``. The caller selects and owns
    the executor (the orchestrator uses ``max_workers=8``)."""
    progress.start("Article sections", total=len(documents))

    def fetch_html(document: Document) -> str:
        try:
            return client.parse_html(document.project, document.language, document.revision_id)
        finally:
            progress.advance()

    sections_by_project: dict[str, list[Section]] = {"wikipedia": [], "wikivoyage": []}
    pending = {executor.submit(fetch_html, document): document for document in documents}
    for future in as_completed(pending):
        document = pending[future]
        html = future.result()
        sections_by_project[document.project].extend(parse_sections(document, html))
    for rows in sections_by_project.values():
        rows.sort(key=lambda row: (row.document_id, row.section_index))
    return sections_by_project


# ---------------------------------------------------------------------------
# build_wikidata_facts
# ---------------------------------------------------------------------------


def build_wikidata_facts(
    client: AugmentationClient,
    *,
    entities: dict[str, dict[str, Any]],
    progress: AugmentationProgress,
) -> list[WikidataFact]:
    """Collect the union of ``FACT_PROPERTIES`` plus all fact value
    entity ids, fetch ``labels`` for them, and produce one
    :class:`WikidataFact` per claim in ``FACT_PROPERTIES``. Output is
    sorted by ``fact_id``."""
    label_ids = set(FACT_PROPERTIES)
    for entity in entities.values():
        for property_id, claims in (entity.get("claims") or {}).items():
            if property_id not in FACT_PROPERTIES:
                continue
            for claim in claims:
                value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
                if isinstance(value, dict) and value.get("id"):
                    label_ids.add(str(value["id"]))
    labels = _label_maps(client.entities(label_ids, props="labels"))
    progress.start("Wikidata facts", total=len(entities))
    facts: list[WikidataFact] = []
    for entity in entities.values():
        facts.extend(normalize_facts(entity, labels))
        progress.advance()
    facts.sort(key=lambda row: row.fact_id)
    return facts


# ---------------------------------------------------------------------------
# write_sidecars
# ---------------------------------------------------------------------------


def write_sidecars(
    paths: tuple[Path, Path, Path, Path, Path],
    *,
    wikipedia_documents: list[Document],
    wikivoyage_documents: list[Document],
    sections_by_project: dict[str, list[Section]],
    facts: list[WikidataFact],
    progress: AugmentationProgress,
    articles_path: Path | None = None,
) -> None:
    """Write the five sidecar parquets in the canonical order and
    advance the ``Writing sidecars`` phase once per file."""
    progress.start("Writing sidecars", total=len(paths))

    if articles_path is None:
        table = pa.Table.from_pylist(
            [row.to_dict() for row in wikipedia_documents],
            schema=document_schema(),
        )
    else:
        try:
            source_table = pq.read_table(articles_path)  # type: ignore[no-untyped-call]
        except Exception as exc:
            raise ValueError(
                f"Failed to read core article Parquet from {articles_path}: {exc}"
            ) from exc
        table = (
            source_table
            if source_table.schema.equals(wikipedia_document_schema(), check_metadata=True)
            else build_wikipedia_document_table(_normalized_article_table(articles_path))
        )

    _write_atomic(paths[0], table)
    progress.advance()

    _write_atomic_from_rows(
        paths[1],
        [row.to_dict() for row in sections_by_project["wikipedia"]],
        SECTION_COLUMNS,
        section_schema(),
    )
    progress.advance()

    _write_atomic_from_rows(
        paths[2],
        [row.to_dict() for row in wikivoyage_documents],
        DOCUMENT_COLUMNS,
        document_schema(),
    )
    progress.advance()

    _write_atomic_from_rows(
        paths[3],
        [row.to_dict() for row in sections_by_project["wikivoyage"]],
        SECTION_COLUMNS,
        section_schema(),
    )
    progress.advance()

    _write_atomic_from_rows(
        paths[4],
        [row.to_dict() for row in facts],
        FACT_COLUMNS,
        fact_schema(),
    )
    progress.advance()


# ---------------------------------------------------------------------------
# update_augmentation_manifest
# ---------------------------------------------------------------------------


def update_augmentation_manifest(
    data_root: DataRoot,
    *,
    stem: str,
    paths: tuple[Path, Path, Path, Path, Path],
    core_hashes: dict[str, str],
    counts: dict[str, int],
    completed_at: str,
    rejections: dict[str, Any] | None = None,
) -> Path:
    """Atomic merge of ``stem``'s entry into the augmentation manifest
    while keeping every other region intact. Returns the manifest
    path; creates the parent directory on first write."""
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entry: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "core_hashes": core_hashes,
        "paths": [str(path.relative_to(data_root.processed)) for path in paths],
        "counts": counts,
        "completed_at": completed_at,
    }
    if rejections is not None:
        entry["rejections"] = rejections
    manifest[stem] = entry
    atomic_write_text(manifest_path, dumps(manifest) + "\n")
    return manifest_path


__all__ = [
    "CONTRACT_VERSION",
    "AugmentationClient",
    "CoreInputs",
    "build_wikidata_facts",
    "fetch_document_sections",
    "fetch_wikivoyage_documents",
    "load_core_inputs",
    "resolve_entities",
    "sha256_file",
    "update_augmentation_manifest",
    "write_sidecars",
]
