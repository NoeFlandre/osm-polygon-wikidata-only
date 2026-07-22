"""Augmentation orchestrator split -- characterization tests.

These tests pin the exact behaviour the decomposition of
``augmentation.orchestrator.augment_region`` into focused helpers
must preserve. The seven extracted helpers in
:mod:`augmentation.steps` are:

* ``load_core_inputs``
* ``resolve_entities``
* ``fetch_wikivoyage_documents``
* ``fetch_document_sections``
* ``build_wikidata_facts``
* ``write_sidecars``
* ``update_augmentation_manifest``

The orchestrator retains the policy: phase ordering, progress
transitions, sidecar paths, the core-hash drift check, the manifest
write, and the thread-pool lifecycle. Each helper-level test freezes
one observable invariant from the pre-split behaviour.

A dedicated test asserts that ``augment_region`` is the only caller
that opens a pool: it constructs exactly two pools, each with eight
workers, and neither helper is responsible for selecting worker counts.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.models import (
    Document,
    WikidataFact,
    document_from_article_row,
    document_id,
)
from osm_polygon_wikidata_only.augmentation.orchestrator import (
    augment_region,
    sidecar_paths,
)
from osm_polygon_wikidata_only.augmentation.progress import AugmentationProgress
from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
)
from osm_polygon_wikidata_only.config.paths import DataRoot

# ---------------------------------------------------------------------------
# Synthetic article row + QID/entity fixture
# ---------------------------------------------------------------------------


def article_row() -> dict[str, object]:
    return {
        "article_id": "Q1:en:10:20",
        "wikidata": "Q1",
        "language": "en",
        "site": "enwiki",
        "title": "Andorra",
        "url": "https://en.wikipedia.org/wiki/Andorra",
        "page_id": 10,
        "revision_id": 20,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-02T00:00:00Z",
        "full_text": "Lead. History text.",
        "full_text_format": "plain_text",
        "article_length_chars": 19,
        "article_length_words": 3,
        "article_length_tokens_estimate": 4,
        "license": "CC BY-SA 4.0",
        "attribution": "Wikipedia",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": "abc",
    }


class FakeAugmentationClient:
    """Minimal in-memory client mirroring the fixture used by
    ``test_augmentation.py``. Records each call for ordering
    assertions."""

    def __init__(self) -> None:
        self.entities_calls: list[tuple[tuple[str, ...], str]] = []
        self.parse_html_calls: list[tuple[str, str, int]] = []
        self.wikivoyage_document_calls: list[tuple[str, str, str, str]] = []

    def entities(self, qids: list[str] | set[str], *, props: str) -> dict[str, dict[str, Any]]:
        qid_tuple = tuple(sorted(qids))
        self.entities_calls.append((qid_tuple, props))
        if props == "labels":
            return {
                qid: {"id": qid, "labels": {"en": {"value": f"English {qid}"}}} for qid in qid_tuple
            }
        return {
            "Q1": {
                "id": "Q1",
                "sitelinks": {"frwikivoyage": {"title": "Andorre"}},
                "claims": {
                    "P31": [
                        {
                            "rank": "normal",
                            "mainsnak": {
                                "snaktype": "value",
                                "datatype": "wikibase-item",
                                "datavalue": {"value": {"id": "Q6256"}},
                            },
                        }
                    ]
                },
            }
        }

    def parse_html(self, project: str, language: str, revision_id: int) -> str:
        self.parse_html_calls.append((project, language, revision_id))
        return '<div class="mw-parser-output"><p>Lead.</p><h2>History</h2><p>Past.</p></div>'

    def wikivoyage_document(self, qid: str, language: str, site: str, title: str) -> Document:
        self.wikivoyage_document_calls.append((qid, language, site, title))
        row = article_row()
        row.update(
            article_id="",
            language=language,
            site=site,
            title=title,
            page_id=30,
            revision_id=40,
            full_text="Travel text.",
        )
        source = document_from_article_row(row)
        values = source.to_dict()
        values.update(
            document_id=document_id(qid, "wikivoyage", language, 30, 40),
            project="wikivoyage",
        )
        return Document(**values)


# ---------------------------------------------------------------------------
# load_core_inputs
# ---------------------------------------------------------------------------


def _seed_core(tmp_path: Path) -> DataRoot:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    pq.write_table(
        pa.Table.from_pylist([article_row()]),
        data_root.processed_articles / "andorra-latest.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"wikidata": "Q1"}]),
        data_root.processed_polygons / "andorra-latest.parquet",
    )
    return data_root


def test_load_core_inputs_missing_articles_raises(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.augmentation.steps import load_core_inputs

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    pq.write_table(
        pa.Table.from_pylist([{"wikidata": "Q1"}]),
        data_root.processed_polygons / "andorra-latest.parquet",
    )

    with pytest.raises(FileNotFoundError):
        load_core_inputs(data_root, "andorra-latest")


def test_load_core_inputs_missing_polygons_raises(tmp_path: Path) -> None:
    from osm_polygon_wikidata_only.augmentation.steps import load_core_inputs

    data_root = DataRoot(tmp_path)
    data_root.ensure()
    pq.write_table(
        pa.Table.from_pylist([article_row()]),
        data_root.processed_articles / "andorra-latest.parquet",
    )

    with pytest.raises(FileNotFoundError):
        load_core_inputs(data_root, "andorra-latest")


def test_load_core_inputs_returns_core_inputs_record(tmp_path: Path) -> None:
    """``load_core_inputs`` returns a ``CoreInputs`` record carrying
    the wikipedia documents, the sorted QIDs, the core paths, and
    the SHA-256 hashes of both core parquets captured before any
    processing. The polygon rows are intentionally not exposed."""

    from osm_polygon_wikidata_only.augmentation.steps import (
        CoreInputs,
        load_core_inputs,
        sha256_file,
    )

    data_root = _seed_core(tmp_path)
    articles_path = data_root.processed_articles / "andorra-latest.parquet"
    polygons_path = data_root.processed_polygons / "andorra-latest.parquet"

    result = load_core_inputs(data_root, "andorra-latest")

    assert isinstance(result, CoreInputs)
    assert [doc.document_id for doc in result.wikipedia_documents] == [
        document_id("Q1", "wikipedia", "en", 10, 20),
    ]
    assert list(result.qids) == ["Q1"]
    assert result.core_paths == (articles_path, polygons_path)
    assert result.core_hashes == {
        str(articles_path): sha256_file(articles_path),
        str(polygons_path): sha256_file(polygons_path),
    }


# ---------------------------------------------------------------------------
# resolve_entities
# ---------------------------------------------------------------------------


def test_resolve_entities_calls_client_with_sorted_qids() -> None:
    """``resolve_entities`` calls ``client.entities`` with a sorted
    list of QIDs and ``props='sitelinks|claims'``, returns the
    entities dict, and reports progress through
    ``Wikidata entities (completed=total)``."""

    from osm_polygon_wikidata_only.augmentation.steps import resolve_entities

    client = FakeAugmentationClient()
    progress = AugmentationProgress()
    qids = ["Q1", "Q2", "Q3"]

    entities = resolve_entities(client, qids, progress=progress)

    assert client.entities_calls == [(("Q1", "Q2", "Q3"), "sitelinks|claims")]
    assert entities == {
        "Q1": client.entities(["Q1"], props="sitelinks|claims")["Q1"],
    }
    snapshot = progress.snapshot()
    assert snapshot.phase == "Wikidata entities"
    assert snapshot.completed == snapshot.total == 3


def test_resolve_entities_empty_qids_no_progress_yet() -> None:
    from osm_polygon_wikidata_only.augmentation.steps import resolve_entities

    class _EmptyClient:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], str]] = []

        def entities(self, qids, *, props):
            self.calls.append((tuple(sorted(qids)), props))
            return {}

    client = _EmptyClient()
    progress = AugmentationProgress()

    entities = resolve_entities(client, [], progress=progress)

    assert entities == {}
    assert client.calls == [((), "sitelinks|claims")]
    snapshot = progress.snapshot()
    assert snapshot.phase == "Wikidata entities"
    assert snapshot.completed == snapshot.total == 0


# ---------------------------------------------------------------------------
# fetch_wikivoyage_documents
# ---------------------------------------------------------------------------


def test_fetch_wikivoyage_documents_returns_sorted_docs_and_progress(tmp_path: Path) -> None:
    """Documents are sorted by ``document_id``; ``None`` results are
    filtered out; progress ends on
    ``Wikivoyage documents (completed=total)``."""

    from osm_polygon_wikidata_only.augmentation.steps import (
        fetch_wikivoyage_documents,
    )

    _seed_core(tmp_path)

    client = FakeAugmentationClient()
    progress = AugmentationProgress()
    entities = client.entities(["Q1"], props="sitelinks|claims")

    with ThreadPoolExecutor(max_workers=8) as executor:
        docs = fetch_wikivoyage_documents(
            client, entities=entities, progress=progress, executor=executor
        )

    assert [doc.document_id for doc in docs] == [
        document_id("Q1", "wikivoyage", "fr", 30, 40),
    ]
    snapshot = progress.snapshot()
    assert snapshot.phase == "Wikivoyage documents"
    assert snapshot.completed == snapshot.total == 1


def test_fetch_wikivoyage_documents_filters_none(tmp_path: Path) -> None:
    """If ``client.wikivoyage_document`` returns ``None`` for some
    items, those entries are dropped before the sort. The progress
    counter still advances once per *attempted* link."""

    from osm_polygon_wikidata_only.augmentation.steps import (
        fetch_wikivoyage_documents,
    )

    _seed_core(tmp_path)

    class _PartialClient(FakeAugmentationClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def wikivoyage_document(self, qid, language, site, title):
            self.calls += 1
            if self.calls == 1:
                return None
            return super().wikivoyage_document(qid, language, site, title)

    client = _PartialClient()
    progress = AugmentationProgress()
    entities = {
        "Q1": {
            "id": "Q1",
            "sitelinks": {
                "enwikivoyage": {"title": "Eng"},
                "frwikivoyage": {"title": "Fr"},
            },
            "claims": {},
        }
    }

    with ThreadPoolExecutor(max_workers=8) as executor:
        docs = fetch_wikivoyage_documents(
            client, entities=entities, progress=progress, executor=executor
        )
    assert len(docs) == 1
    snapshot = progress.snapshot()
    # Counter advances once per *attempted* link, not per surviving doc.
    assert snapshot.completed == snapshot.total == 2


# ---------------------------------------------------------------------------
# fetch_document_sections
# ---------------------------------------------------------------------------


def test_fetch_document_sections_partitions_by_project_and_sorts(tmp_path: Path) -> None:
    """Sections are partitioned into wikipedia/wikivoyage buckets
    and each bucket is sorted by ``(document_id, section_index)``."""

    from osm_polygon_wikidata_only.augmentation.steps import (
        fetch_document_sections,
    )

    _seed_core(tmp_path)

    client = FakeAugmentationClient()
    progress = AugmentationProgress()

    docs = [document_from_article_row(article_row())]
    voyage = client.wikivoyage_document("Q1", "fr", "frwikivoyage", "Andorre")

    with ThreadPoolExecutor(max_workers=8) as executor:
        sections = fetch_document_sections(
            client,
            documents=[*docs, voyage],
            progress=progress,
            executor=executor,
        )

    assert sorted(sections.keys()) == ["wikipedia", "wikivoyage"]
    for rows in sections.values():
        keys = [(row.document_id, row.section_index) for row in rows]
        assert keys == sorted(keys)
    assert len(sections["wikipedia"]) == 2
    assert len(sections["wikivoyage"]) == 2


def test_fetch_document_sections_parses_completed_html_without_waiting_for_tail() -> None:
    """A slow final fetch must not retain and block already-fetched HTML."""
    from osm_polygon_wikidata_only.augmentation.steps import fetch_document_sections

    client = FakeAugmentationClient()
    progress = AugmentationProgress()
    first = document_from_article_row(article_row())
    second_row = article_row()
    second_row.update(article_id="Q2:en:11:21", wikidata="Q2", page_id=11, revision_id=21)
    second = document_from_article_row(second_row)
    release_tail = threading.Event()
    first_parsed = threading.Event()
    original_parse_html = client.parse_html

    def fetch_html(project: str, language: str, revision_id: int) -> str:
        if revision_id == second.revision_id:
            assert release_tail.wait(timeout=2), "test did not release tail fetch"
        return original_parse_html(project, language, revision_id)

    def record_parse(document: Document, _html: str) -> list[object]:
        if document.document_id == first.document_id:
            first_parsed.set()
        return []

    client.parse_html = fetch_html  # type: ignore[method-assign]
    result: list[dict[str, list[object]]] = []

    def execute() -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            with mock.patch(
                "osm_polygon_wikidata_only.augmentation.steps.parse_sections",
                side_effect=record_parse,
            ):
                result.append(
                    fetch_document_sections(
                        client,
                        documents=[first, second],
                        progress=progress,
                        executor=executor,
                    )
                )

    thread = threading.Thread(target=execute)
    thread.start()
    try:
        assert first_parsed.wait(timeout=1), "completed HTML was retained behind tail fetch"
    finally:
        release_tail.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(result) == 1


# ---------------------------------------------------------------------------
# build_wikidata_facts
# ---------------------------------------------------------------------------


def test_build_wikidata_facts_calls_labels_with_property_plus_value_ids() -> None:
    """``build_wikidata_facts`` issues exactly one ``labels`` fetch
    for the union of ``FACT_PROPERTIES`` and all fact value entity
    ids, then produces one ``WikidataFact`` per claim in
    ``FACT_PROPERTIES``, sorted by ``fact_id``."""

    from osm_polygon_wikidata_only.augmentation.steps import build_wikidata_facts

    client = FakeAugmentationClient()
    progress = AugmentationProgress()
    entities = client.entities(["Q1"], props="sitelinks|claims")

    facts = build_wikidata_facts(client, entities=entities, progress=progress)

    label_calls = [call for call in client.entities_calls if call[1] == "labels"]
    assert len(label_calls) == 1
    label_ids = set(label_calls[0][0])
    assert {"P31", "Q6256"} <= label_ids
    assert len(facts) == 1
    assert facts[0].fact_id
    fact_ids = [f.fact_id for f in facts]
    assert fact_ids == sorted(fact_ids)
    snapshot = progress.snapshot()
    assert snapshot.phase == "Wikidata facts"
    assert snapshot.completed == snapshot.total == 1


# ---------------------------------------------------------------------------
# write_sidecars
# ---------------------------------------------------------------------------


def test_write_sidecars_writes_five_files_in_approved_order(tmp_path: Path) -> None:
    """All five sidecar paths are written in the canonical order
    (wikipedia_documents, wikipedia_sections, voyage_documents,
    voyage_sections, facts) and the progress counter advances once
    per file."""

    from osm_polygon_wikidata_only.augmentation.steps import write_sidecars

    data_root = _seed_core(tmp_path)
    paths = sidecar_paths(data_root, "andorra-latest")
    doc = document_from_article_row(article_row())
    voyage = doc.__class__(
        **{
            **doc.to_dict(),
            "document_id": document_id("Q1", "wikivoyage", "fr", 30, 40),
            "project": "wikivoyage",
        }
    )
    sections = {"wikipedia": [], "wikivoyage": []}

    fact = WikidataFact(
        fact_id="hash",
        wikidata="Q1",
        property_id="P31",
        property_label_en="instance of",
        property_labels="{}",
        value_type="wikibase-item",
        value_entity_id="Q6256",
        value_label_en="country",
        value_labels="{}",
        value_text="Q6256",
        numeric_value=None,
        unit_entity_id="",
        rank="normal",
        qualifiers="{}",
        references="[]",
        retrieved_at="now",
        source_api="wikidata_action_api",
    )

    progress = AugmentationProgress()
    write_sidecars(
        paths,
        wikipedia_documents=[doc],
        wikivoyage_documents=[voyage],
        sections_by_project=sections,
        facts=[fact],
        progress=progress,
    )

    expected_first4 = (
        DOCUMENT_COLUMNS[:4],
        SECTION_COLUMNS[:4],
        DOCUMENT_COLUMNS[:4],
        SECTION_COLUMNS[:4],
        FACT_COLUMNS[:4],
    )
    for path, expected in zip(paths, expected_first4):
        assert path.exists()
        table = pq.read_table(path)
        assert tuple(table.schema.names[:4]) == expected
    snapshot = progress.snapshot()
    assert snapshot.phase == "Writing sidecars"
    assert snapshot.completed == snapshot.total == 5


# ---------------------------------------------------------------------------
# update_augmentation_manifest
# ---------------------------------------------------------------------------


def test_update_augmentation_manifest_merges_existing_entries(tmp_path: Path) -> None:
    """A pre-existing manifest keeps its other regions intact; only
    the targeted stem is overwritten, with the canonical key set
    and the relative paths under ``data_root.processed``."""

    from osm_polygon_wikidata_only.augmentation.steps import (
        update_augmentation_manifest,
    )

    data_root = _seed_core(tmp_path)
    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "monaco-latest": {
                    "contract_version": "text-sidecars-v1",
                    "core_hashes": {"x": "y"},
                    "paths": ["a"],
                    "counts": {"wikipedia_documents": 5},
                    "completed_at": "2026-01-01T00:00:00Z",
                }
            }
        )
    )

    paths = sidecar_paths(data_root, "andorra-latest")
    core_hashes = {"h1": "abc"}
    counts = {"wikipedia_documents": 1}

    update_augmentation_manifest(
        data_root,
        stem="andorra-latest",
        paths=paths,
        core_hashes=core_hashes,
        counts=counts,
        completed_at="2026-02-02T00:00:00Z",
    )

    manifest = json.loads(manifest_path.read_text())
    assert "monaco-latest" in manifest
    assert manifest["monaco-latest"]["core_hashes"] == {"x": "y"}
    assert manifest["andorra-latest"] == {
        "contract_version": "text-sidecars-v1",
        "core_hashes": core_hashes,
        "paths": [
            "wikipedia/documents/andorra-latest.parquet",
            "wikipedia/sections/andorra-latest.parquet",
            "wikivoyage/documents/andorra-latest.parquet",
            "wikivoyage/sections/andorra-latest.parquet",
            "wikidata/facts/andorra-latest.parquet",
        ],
        "counts": counts,
        "completed_at": "2026-02-02T00:00:00Z",
    }


def test_update_augmentation_manifest_atomic_write_creates_parent(tmp_path: Path) -> None:
    """The manifest write creates the parent directory on first
    write."""

    from osm_polygon_wikidata_only.augmentation.steps import (
        update_augmentation_manifest,
    )

    data_root = _seed_core(tmp_path)
    paths = sidecar_paths(data_root, "andorra-latest")

    update_augmentation_manifest(
        data_root,
        stem="andorra-latest",
        paths=paths,
        core_hashes={"h1": "abc"},
        counts={"wikipedia_documents": 1},
        completed_at="2026-02-02T00:00:00Z",
    )

    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    assert manifest_path.exists()


# ---------------------------------------------------------------------------
# Pool ownership: orchestrator opens exactly two 8-worker pools
# ---------------------------------------------------------------------------


def test_augment_region_opens_two_pools_of_eight_workers(tmp_path: Path) -> None:
    """Pool selection lives at the facade, not in the helpers. The
    orchestrator must open exactly two pools for the standard run,
    each sized ``max_workers=8``, and the helpers must consume the
    executor they receive without consulting worker counts."""

    from unittest import mock

    import osm_polygon_wikidata_only.augmentation.orchestrator as orchestrator

    pool_sizes: list[int] = []
    helpers_received: list[tuple[str, int]] = []

    class _RecordingPool:
        def __init__(self, max_workers: int) -> None:
            self._max_workers = max_workers

        def __enter__(self) -> _RecordingPool:
            pool_sizes.append(self._max_workers)
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def map(self, fn, iterable):  # type: ignore[no-untyped-def]
            return [fn(item) for item in iterable]

        def submit(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
            from concurrent.futures import Future

            future: Future = Future()
            future.set_result(fn(*args, **kwargs))
            return future

    original_voyage = orchestrator.fetch_wikivoyage_documents
    original_sections = orchestrator.fetch_document_sections
    real_pool = orchestrator.ThreadPoolExecutor

    def voyage_recorder(client: Any, **kwargs: Any) -> Any:
        helpers_received.append(("voyage", kwargs["executor"]._max_workers))
        return original_voyage(client, **kwargs)

    def sections_recorder(client: Any, **kwargs: Any) -> Any:
        helpers_received.append(("sections", kwargs["executor"]._max_workers))
        return original_sections(client, **kwargs)

    pool_patch = mock.patch.object(orchestrator, "ThreadPoolExecutor", _RecordingPool)
    voyage_patch = mock.patch.object(orchestrator, "fetch_wikivoyage_documents", voyage_recorder)
    sections_patch = mock.patch.object(orchestrator, "fetch_document_sections", sections_recorder)
    pool_patch.start()
    voyage_patch.start()
    sections_patch.start()
    try:
        data_root = _seed_core(tmp_path)
        augment_region(data_root, "andorra-latest", FakeAugmentationClient())
    finally:
        pool_patch.stop()
        voyage_patch.stop()
        sections_patch.stop()
        # mock.patch.stop() already restored the originals.

    # Sanity: pool is the real class again.
    assert orchestrator.ThreadPoolExecutor is real_pool
    # Orchestrator: two pools, both with eight workers.
    assert pool_sizes == [8, 8]
    # Helpers: receive the executor that the orchestrator chose.
    assert helpers_received == [("voyage", 8), ("sections", 8)]


# ---------------------------------------------------------------------------
# Shared hashing: one implementation backs every content-addressed hash
# ---------------------------------------------------------------------------


def test_hashing_is_a_single_shared_implementation() -> None:
    """The initial hash capture (in ``load_core_inputs``), the drift
    check (in ``augment_region``), and the resumability check (in
    ``augmentation_is_current``) all call the same ``sha256_file``
    object from :mod:`augmentation.steps`."""

    from osm_polygon_wikidata_only.augmentation import orchestrator, steps

    assert orchestrator.sha256_file is steps.sha256_file


# ---------------------------------------------------------------------------
# Exception propagation: no manifest on failure
# ---------------------------------------------------------------------------


def test_augment_region_does_not_persist_manifest_on_core_drift(tmp_path: Path) -> None:
    """If the core hash drifts during processing, ``augment_region``
    raises *without* writing the manifest entry."""

    data_root = _seed_core(tmp_path)

    class _MutatingClient(FakeAugmentationClient):
        def entities(self, qids, *, props):
            if props == "sitelinks|claims":
                articles_path = data_root.processed_articles / "andorra-latest.parquet"
                pq.write_table(
                    pa.Table.from_pylist([article_row(), article_row()]),
                    articles_path,
                )
            return super().entities(qids, props=props)

    manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )

    with pytest.raises(RuntimeError, match="Core artifacts changed during augmentation"):
        augment_region(data_root, "andorra-latest", _MutatingClient())
    assert not manifest_path.exists()


def test_augment_region_propagates_value_error_from_non_object_json(tmp_path: Path) -> None:
    """A ``ValueError`` raised deep inside a client call surfaces
    from ``augment_region`` unchanged."""

    data_root = _seed_core(tmp_path)

    class _BoomClient(FakeAugmentationClient):
        def parse_html(self, project, language, revision_id):
            raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        augment_region(data_root, "andorra-latest", _BoomClient())


# ---------------------------------------------------------------------------
# Facade end-to-end equivalence
# ---------------------------------------------------------------------------


def test_facade_end_to_end_equivalence(tmp_path: Path) -> None:
    """Run the full ``augment_region`` facade and assert sidecar
    ordering, manifest counts and the ``AugmentationResult`` mapping
    are preserved."""

    data_root = _seed_core(tmp_path)
    result = augment_region(data_root, "andorra-latest", FakeAugmentationClient())

    paths = list(sidecar_paths(data_root, "andorra-latest"))
    expected_order = (
        "wikipedia/documents/andorra-latest.parquet",
        "wikipedia/sections/andorra-latest.parquet",
        "wikivoyage/documents/andorra-latest.parquet",
        "wikivoyage/sections/andorra-latest.parquet",
        "wikidata/facts/andorra-latest.parquet",
    )
    assert tuple(str(p.relative_to(data_root.processed)) for p in paths) == expected_order
    assert result.wikipedia_documents_path == paths[0]
    assert result.wikipedia_sections_path == paths[1]
    assert result.wikivoyage_documents_path == paths[2]
    assert result.wikivoyage_sections_path == paths[3]
    assert result.wikidata_facts_path == paths[4]
    assert result.counts == {
        "wikipedia_documents": 1,
        "wikipedia_sections": 2,
        "wikivoyage_documents": 1,
        "wikivoyage_sections": 2,
        "wikidata_facts": 1,
    }
