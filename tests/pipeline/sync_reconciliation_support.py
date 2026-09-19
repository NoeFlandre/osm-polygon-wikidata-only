from __future__ import annotations

# ruff: noqa: F401
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.cli import commands, run_sync
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.config.settings import Settings
from osm_polygon_wikidata_only.domain.schema import (
    article_schema,
    polygon_article_schema,
    polygon_schema,
)
from osm_polygon_wikidata_only.hf._uploader.stub import StubHfHub
from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory
from osm_polygon_wikidata_only.pipeline.sync_planner import SyncAction, plan_sync_states


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _setup_mock_region(
    data_root: DataRoot, stem: str, augmented: bool = True, invalid_core: bool = False
) -> None:
    data_root.processed_polygons.mkdir(parents=True, exist_ok=True)
    data_root.processed_links.mkdir(parents=True, exist_ok=True)
    data_root.processed_manifests.mkdir(parents=True, exist_ok=True)

    poly_table = pa.Table.from_pylist(
        [
            {
                "polygon_id": "1",
                "wikidata": "Q1",
                "lat": 1.0,
                "lon": 2.0,
                "source_pbf": f"{stem}.osm.pbf",
                "region": stem,
                "osm_type": "relation",
                "osm_id": 1,
            }
        ],
        schema=polygon_schema(),
    )
    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    pq.write_table(poly_table, polygons_path)  # type: ignore[no-untyped-call]

    if not invalid_core:
        links_table = pa.Table.from_pylist(
            [
                {
                    "polygon_id": "1",
                    "article_id": "Q1:es:1234:5678",
                    "wikidata": "Q1",
                    "language": "es",
                    "source_pbf": f"{stem}.osm.pbf",
                    "region": stem,
                    "osm_type": "relation",
                    "osm_id": 1,
                    "page_id": 1234,
                    "revision_id": 5678,
                    "is_best_language": True,
                }
            ],
            schema=polygon_article_schema(),
        )
        pq.write_table(links_table, data_root.processed_links / f"{stem}.parquet")  # type: ignore[no-untyped-call]

    manifest_data = {}
    processed_pbfs_path = data_root.processed_manifests / "processed_pbfs.json"
    if processed_pbfs_path.exists():
        manifest_data = json.loads(processed_pbfs_path.read_text(encoding="utf-8"))
    if not invalid_core:
        manifest_data[f"{stem}.osm.pbf"] = {
            "source_pbf": f"{stem}.osm.pbf",
            "region": stem,
            "polygons_path": f"polygons/{stem}.parquet",
            "polygon_articles_path": f"polygon_articles/{stem}.parquet",
            "wikipedia_documents_path": f"wikipedia/documents/{stem}.parquet",
            "polygon_count": 1,
            "article_count": 1,
            "link_count": 1,
        }
        processed_pbfs_path.write_text(json.dumps(manifest_data, indent=2))

    # wikipedia documents
    wikipedia_documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    wikipedia_documents_path.parent.mkdir(parents=True, exist_ok=True)
    doc_table = pa.Table.from_pylist(
        [
            {
                "document_id": "Q1:wikipedia:es:1234:5678",
                "article_id": "Q1:es:1234:5678",
                "wikidata": "Q1",
                "language": "es",
                "source_pbf": f"{stem}.osm.pbf",
                "region": stem,
                "osm_type": "relation",
                "osm_id": 1,
                "page_id": 1234,
                "revision_id": 5678,
                "project": "wikipedia",
            }
        ],
        schema=wikipedia_document_schema(),
    )
    pq.write_table(doc_table, wikipedia_documents_path)  # type: ignore[no-untyped-call]

    if augmented:
        wikipedia_sections_path = data_root.processed / "wikipedia" / "sections" / f"{stem}.parquet"
        wikivoyage_documents_path = (
            data_root.processed / "wikivoyage" / "documents" / f"{stem}.parquet"
        )
        wikivoyage_sections_path = (
            data_root.processed / "wikivoyage" / "sections" / f"{stem}.parquet"
        )
        wikidata_facts_path = data_root.processed / "wikidata" / "facts" / f"{stem}.parquet"

        from osm_polygon_wikidata_only.augmentation.schema import (
            document_schema,
            fact_schema,
            section_schema,
        )

        for p, schema in [
            (wikipedia_sections_path, section_schema()),
            (wikivoyage_documents_path, document_schema()),
            (wikivoyage_sections_path, section_schema()),
            (wikidata_facts_path, fact_schema()),
        ]:
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_batches([], schema=schema), p)  # type: ignore[no-untyped-call]

        aug_manifest = {}
        aug_manifest[stem] = {
            "contract_version": "text-sidecars-v1",
            "core_hashes": {
                str(polygons_path): compute_sha256(polygons_path),
                str(wikipedia_documents_path): compute_sha256(wikipedia_documents_path),
            },
            "counts": {
                "wikipedia_documents": 1,
                "wikipedia_sections": 0,
                "wikivoyage_documents": 0,
                "wikivoyage_sections": 0,
                "wikidata_facts": 0,
            },
        }
        aug_manifest_path = (
            data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
        )
        aug_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Merge into any existing augmentation manifest so that setting
        # up multiple regions (e.g. mexico-latest + hungary-latest) does
        # not clobber earlier entries. A clobbered entry would drop the
        # stem from the finalized set and wrongly classify it as AUGMENT,
        # which then triggers a real Wikidata/Wikipedia fetch.
        if aug_manifest_path.exists():
            existing = json.loads(aug_manifest_path.read_text(encoding="utf-8"))
            existing.update(aug_manifest)
            aug_manifest = existing
        aug_manifest_path.write_text(json.dumps(aug_manifest, indent=2))


@pytest.fixture
def mock_hf_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(commands, "resolve_hf_token", lambda value: "fake-token")
    monkeypatch.setattr(commands, "verify_hf_token", lambda value: "noeflandre")
    monkeypatch.setattr(commands, "verify_repo_authorization", lambda token, repo_id: "noeflandre")


class _LoggerSpy:
    """In-process recorder for ``LOGGER`` info/error calls.

    Each emission is stored with its interpolated message string.
    Tests can assert on the recorded list without depending on
    pytest's caplog state, which can be reset by earlier tests
    that call ``configure_logging``. ``error`` is intercepted in
    addition to ``info`` so failure-path messages (e.g.
    ``"Unified sync completed with failures"``) are visible to
    tests, while ``warning``, ``debug``, and ``critical`` are
    stubbed defensively to keep the spy stable across any future
    changes to the cli module.
    """

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.messages.append(str(message))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.messages.append(str(message))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _install_logger_spy(monkeypatch: pytest.MonkeyPatch) -> _LoggerSpy:
    """Replace ``cli.LOGGER`` methods with a deterministic spy.

    ``info`` and ``error`` are swapped for the spy's recording
    implementations; ``warning``/``debug``/``critical`` are
    stubbed so any silent fallback paths don't surface as
    real logger calls during a focused test. Restoration is
    automatic when the ``monkeypatch`` fixture is torn down.
    """
    spy = _LoggerSpy()
    monkeypatch.setattr(run_sync.LOGGER, "info", spy.info)
    monkeypatch.setattr(run_sync.LOGGER, "error", spy.error)
    monkeypatch.setattr(run_sync.LOGGER, "warning", spy.warning)
    monkeypatch.setattr(run_sync.LOGGER, "debug", spy.debug)
    monkeypatch.setattr(run_sync.LOGGER, "critical", spy.critical)
    return spy


class _NetworkBoundaryRecorder:
    """Tracks invocations of the stubbed ``ensure_world_land``.

    Tests assert on ``calls`` to confirm that the production
    code reached the stubbed boundary (and therefore would have
    hit a real download in production) and on ``urlretrieve_calls``
    to confirm that ``urllib.request.urlretrieve`` -- the actual
    network primitive -- was never invoked.
    """

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.urlretrieve_calls: list[tuple[Any, ...]] = []


def _block_network(monkeypatch: pytest.MonkeyPatch) -> _NetworkBoundaryRecorder:
    """Replace the real network boundary with a deterministic stub.

    The publication module imports ``ensure_world_land`` directly
    from :mod:`hf.coverage_map` (``from .coverage_map import
    ensure_world_land``), so the binding actually used at runtime
    lives on :mod:`hf.publication`. Patching only
    ``hf.coverage_map.ensure_world_land`` leaves the publication
    module holding the original function and the boundary is
    silently bypassed. This helper therefore patches both the
    canonical definition AND the symbol already imported by
    :mod:`hf.publication`, ensuring any call from the publication
    code path is intercepted.

    ``urllib.request.urlretrieve`` is patched to raise as a
    secondary guard: any future caller that re-introduces a direct
    download becomes a deterministic test failure rather than a
    silent HTTP request.
    """
    recorder = _NetworkBoundaryRecorder()

    def _fake_ensure_world_land(cache_dir: Path) -> Path:
        recorder.calls.append(Path(cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / "world_land.geojson"
        target.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        return target

    # Canonical definition: any future caller that imports
    # ``ensure_world_land`` from :mod:`hf.coverage_map` is covered.
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.coverage_map.ensure_world_land",
        _fake_ensure_world_land,
    )
    # Publication binding: this is the symbol actually called at
    # runtime because :mod:`hf.publication` does
    # ``from .coverage_map import ensure_world_land`` at import
    # time, binding the function into its own namespace.
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        _fake_ensure_world_land,
    )

    def _no_urlretrieve(*args: Any, **kwargs: Any) -> Any:
        recorder.urlretrieve_calls.append((args, kwargs))
        raise AssertionError(
            "Real network call attempted: urllib.request.urlretrieve; "
            "tests must mock network boundaries."
        )

    monkeypatch.setattr("urllib.request.urlretrieve", _no_urlretrieve, raising=False)
    return recorder


class _ReconciliationNetworkRecorder:
    """Records any guarded network-boundary invocation.

    Tests assert ``wikimedia_calls == []`` and ``urlretrieve_calls == []``
    to prove the production code reached neither the augmentation
    transport nor the coverage-map download.
    """

    def __init__(self) -> None:
        self.wikimedia_calls: list[tuple[Any, ...]] = []
        self.urlretrieve_calls: list[tuple[Any, ...]] = []


def _block_reconciliation_network(
    monkeypatch: pytest.MonkeyPatch,
) -> _ReconciliationNetworkRecorder:
    """Fail loudly on the real network boundaries used by the sync flow.

    The augmentation client issues HTTPS fetches through
    :func:`augmentation.mediawiki.read_wikimedia_json` (re-exported
    from :mod:`enrichment.wikimedia`). A test that wrongly classifies
    a finalized region as AUGMENT would call it. The publication
    coverage-map rendering calls ``urllib.request.urlretrieve`` to
    download the Natural Earth land GeoJSON.

    Patching the actual augmentation boundary (not the low-level
    ``urllib.request.urlopen`` it happens to sit on) makes the guard
    authoritative: the symbol the production code imports is the one
    intercepted, so a future refactor that swaps the transport cannot
    silently bypass the guard. Both primitives are replaced with
    fail-loud stubs and their invocations are recorded.

    This guard is defense-in-depth only: callers MUST still correct
    their action classification so the network code is never reached
    in the first place.
    """
    recorder = _ReconciliationNetworkRecorder()

    import osm_polygon_wikidata_only.augmentation.mediawiki as mediawiki_mod

    def _no_read_wikimedia_json(*args: Any, **kwargs: Any) -> Any:
        recorder.wikimedia_calls.append((args, kwargs))
        raise AssertionError(
            "Real network call attempted: augmentation.mediawiki.read_wikimedia_json; "
            "reconciliation tests must classify finalized regions as "
            "COMPLETE/PUBLISH, not AUGMENT."
        )

    monkeypatch.setattr(mediawiki_mod, "read_wikimedia_json", _no_read_wikimedia_json)

    # Stub the coverage-map download so the publication path never
    # reaches ``urllib.request.urlretrieve``. The publication module
    # imports ``ensure_world_land`` directly from ``hf.coverage_map``,
    # so both the canonical definition and the symbol already bound
    # into ``hf.publication`` are replaced. This is the deterministic
    # substitute that satisfies the "urlretrieve calls are zero"
    # assertion; the ``urlretrieve`` guard below remains as a
    # fail-loud backstop if a future caller re-introduces a direct
    # download.
    def _fake_ensure_world_land(cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / "world_land.geojson"
        target.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        return target

    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.coverage_map.ensure_world_land",
        _fake_ensure_world_land,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        _fake_ensure_world_land,
    )

    def _no_urlretrieve(*args: Any, **kwargs: Any) -> Any:
        recorder.urlretrieve_calls.append((args, kwargs))
        raise AssertionError(
            "Real network call attempted: urllib.request.urlretrieve; "
            "tests must stub the coverage-map network boundary."
        )

    monkeypatch.setattr("urllib.request.urlretrieve", _no_urlretrieve, raising=False)
    return recorder


def _refresh_augmentation_manifest(data_root: DataRoot, stem: str) -> None:
    """Recompute and persist the canonical core-hash entry for *stem*.

    Used after a test overwrites the canonical Wikipedia-document (or
    legacy articles) file so the stored manifest hashes match the
    on-disk bytes. Without this refresh ``augmentation_is_current``
    returns False and the region is mis-classified as AUGMENT, which
    would then issue a real Wikidata/Wikipedia fetch.
    """
    from osm_polygon_wikidata_only.augmentation.steps import sha256_file

    polygons_path = data_root.processed_polygons / f"{stem}.parquet"
    wikipedia_documents_path = data_root.processed / "wikipedia" / "documents" / f"{stem}.parquet"
    core_hashes = {
        str(polygons_path): sha256_file(polygons_path),
        str(wikipedia_documents_path): sha256_file(wikipedia_documents_path),
    }
    aug_manifest_path = (
        data_root.processed / "augmentation" / "manifests" / "augmentation_manifest.json"
    )
    aug_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = (
        json.loads(aug_manifest_path.read_text(encoding="utf-8"))
        if aug_manifest_path.exists()
        else {}
    )
    entry = manifest.get(stem, {})
    entry["contract_version"] = "text-sidecars-v1"
    entry["core_hashes"] = core_hashes
    manifest[stem] = entry
    aug_manifest_path.write_text(json.dumps(manifest, indent=2))


def setup_test_hub(monkeypatch: pytest.MonkeyPatch, stub: StubHfHub) -> None:
    # 1. Mock RemoteInventory.fetch to return files from the stub
    from osm_polygon_wikidata_only.hf.remote_inventory import RemoteInventory

    def mock_fetch(repo_id: str, token: str | None = None, hub: Any = None) -> RemoteInventory:
        files = stub.list_repo_files(repo_id=repo_id, repo_type="dataset")
        return RemoteInventory(set(files))

    monkeypatch.setattr(RemoteInventory, "fetch", mock_fetch)

    # 2. Inject the stub HfHub into the background upload queue construction
    original_build_queue = run_sync._build_upload_queue

    def mock_build_queue(*args: Any, **kwargs: Any) -> Any:
        kwargs["_hub"] = stub
        return original_build_queue(*args, **kwargs)

    monkeypatch.setattr(run_sync, "_build_upload_queue", mock_build_queue)


__all__ = [name for name in globals() if not name.startswith("__")]
