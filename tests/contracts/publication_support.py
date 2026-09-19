"""Freeze the publication assembly contracts and submission counts.

The :mod:`osm_polygon_wikidata_only.hf.publication` module owns
pure assembly of :class:`PublicationOp` lists for three documented
publication contracts. This module captures those op lists as
golden expectations, asserts the assemblers are PURE (no
``submit`` parameter, no upload side effect), and asserts that
CLI callsites perform exactly one submission per publication.
Tests use stub data only -- no Parquet I/O, no network, no HF
client.

Contracts exercised:

* Legacy core publication (no augmentation): polygons, articles,
  polygon_articles, processed manifest, geographic text coverage,
  geographic polygon count, README, legacy coverage map.
* Unified sync with changed core: core block first, then the
  seven augmentation artifacts (wikipedia + wikivoyage +
  wikidata + per-region augmentation manifest snapshot +
  migration ``delete`` op + README).
* Augmentation-only publication (legacy augmentation command):
  five sidecars + ``augmentation_result.manifest_path`` (NOT a
  new stem snapshot) + migration ``delete`` op + README. No
  coverage assets are regenerated.

The canonical augmentation manifest
``manifests/augmentation_manifest.json`` unifies the remote
layout: every augmentation publication emits an ``add`` for the
canonical path AND a ``delete`` for the legacy
``augmentation/manifests/augmentation_manifest.json`` path in the
SAME atomic commit. The delete is idempotent once the legacy
remote file is gone.

The tests also verify exact submission counts:
* Legacy core CLI: one assembly, one queue submission.
* Augmentation command: one assembly, one direct upload call.
* Unified sync: one assembly, one queue submission (no double
  submit when the runner's ``_maybe_submit`` calls the upload
  queue with the assembled list).
* Assembly failure: zero submissions.
"""

from __future__ import annotations

# ruff: noqa: F401
import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.orchestrator import AugmentationResult
from osm_polygon_wikidata_only.config.paths import DataRoot
from osm_polygon_wikidata_only.domain.schema import article_schema
from osm_polygon_wikidata_only.hf.publication import (
    assemble_augmentation_upload,
    assemble_core_upload,
    assemble_metadata_only_upload,
    assemble_region_upload,
    coverage_refresh_required,
    refresh_coverage_assets,
    snapshot_upload_manifests,
)
from osm_polygon_wikidata_only.pipeline.processor import ProcessResult

STEM = "monaco-latest"
REPO_ID = "NoeFlandre/osm-polygon-wikidata-only"


def _stub_process_result(tmp_path: Path) -> tuple[ProcessResult, DataRoot]:
    data_root = DataRoot(tmp_path)
    data_root.ensure()
    polygons = data_root.processed_polygons / f"{STEM}.parquet"
    articles = data_root.processed_articles / f"{STEM}.parquet"
    links = data_root.processed_links / f"{STEM}.parquet"
    manifest = data_root.processed_manifests / "processed_pbfs.json"
    for p in (polygons, links, manifest):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    pq.write_table(pa.Table.from_pylist([], schema=article_schema()), articles)
    manifest.write_text("{}", encoding="utf-8")
    return (
        ProcessResult(
            polygons_path=polygons,
            articles_path=articles,
            polygon_articles_path=links,
            manifest_path=manifest,
            polygon_count=10,
            article_count=5,
            link_count=15,
            manifest_entry={"source_pbf": f"{STEM}.osm.pbf"},
            stage_timings_s={},
        ),
        data_root,
    )


def _stub_augmentation_result(processed_root: Path) -> AugmentationResult:
    """Build an :class:`AugmentationResult` whose paths live under ``processed_root``.

    Mirrors the real layout: ``data_root.processed/{wikipedia,wikivoyage,wikidata,...}``.
    """
    paths = {
        "wikipedia_documents_path": processed_root / "wikipedia" / "documents" / f"{STEM}.parquet",
        "wikipedia_sections_path": processed_root / "wikipedia" / "sections" / f"{STEM}.parquet",
        "wikivoyage_documents_path": processed_root
        / "wikivoyage"
        / "documents"
        / f"{STEM}.parquet",
        "wikivoyage_sections_path": processed_root / "wikivoyage" / "sections" / f"{STEM}.parquet",
        "wikidata_facts_path": processed_root / "wikidata" / "facts" / f"{STEM}.parquet",
        "manifest_path": processed_root
        / "augmentation"
        / "manifests"
        / "augmentation_manifest.json",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    return AugmentationResult(**paths, counts={"wikipedia_documents": 1})


def _stub_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the geographic/coverage generators so tests don't depend on
    matplotlib, real dataset stats, or world-land download.

    The stubs create empty files at the requested destinations so the
    assembly helpers return paths that exist on disk.
    """
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._load_text_presence",
        lambda _root: object(),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_density_snapshot",
        lambda *a, **kw: a[1].touch() or a[1],
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication._generate_geographic_text_presence",
        lambda _root, dest, **_kw: dest.touch() or dest,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.load_centroids_from_parquet",
        lambda _dir: ([], []),
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.ensure_world_land",
        lambda _dir: None,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.generate_coverage_map",
        lambda _lons, _lats, dest, **_kw: dest.touch() or dest,
    )
    monkeypatch.setattr(
        "osm_polygon_wikidata_only.hf.publication.write_readme_snapshot",
        lambda *a, **kw: None,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
