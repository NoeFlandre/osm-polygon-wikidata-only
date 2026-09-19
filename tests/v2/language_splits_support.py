"""Acceptance tests for the V2-only row-level language split generator."""

from __future__ import annotations

# ruff: noqa: F401
import errno
import json
import subprocess
import sys
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.hf.language_splits import (
    DatasetContract,
    LanguageBucket,
    LanguageInventory,
    LanguageTable,
    LanguageTableInventory,
    build_language_inventory,
    language_table_specs,
)
from osm_polygon_wikidata_only.v2 import language_splits
from osm_polygon_wikidata_only.v2.language_splits import (
    DEFAULT_BATCH_SIZE,
    LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH,
    V2_LANGUAGE_SPLIT_CONTRACT_VERSION,
    V2LanguageSplitError,
    V2LanguageSplitFile,
    V2LanguageSplitResult,
    _file_sort_key,
    _language_sort_key,
    _manifest_partition_paths,
    _validate_conservation,
    build_v2_language_splits,
    main,
)
from osm_polygon_wikidata_only.v2.schema import (
    polygon_document_link_v2_schema,
    polygon_v2_schema,
    wikipedia_document_v2_schema,
)


def _row_for_schema(schema: pa.Schema, **values: object) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in schema:
        if pa.types.is_boolean(field.type):
            row[field.name] = False
        elif pa.types.is_integer(field.type):
            row[field.name] = 0
        elif pa.types.is_floating(field.type):
            row[field.name] = 0.0
        else:
            row[field.name] = ""
    row.update(values)
    return row

def _write_table(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="snappy",
        row_group_size=1,
    )

def _polygon(polygon_id: str, *, best_language: str) -> dict[str, object]:
    return _row_for_schema(
        polygon_v2_schema(),
        polygon_id=polygon_id,
        region="fixture",
        source_pbf=f"{polygon_id}.osm.pbf",
        osm_type="way",
        osm_id=int(polygon_id.removeprefix("polygon-")),
        wikidata="Q1",
        best_language=best_language,
    )

def _document(document_id: str, language: object, title: str) -> dict[str, object]:
    return _row_for_schema(
        wikipedia_document_v2_schema(),
        document_id=document_id,
        article_id=f"article-{document_id}",
        wikidata="Q1",
        project="wikipedia",
        language=language,
        site=f"{language or 'unknown'}wiki",
        title=title,
        page_id=1,
        revision_id=1,
        full_text=title,
        fetch_status="ok",
    )

def _section(section_id: str, language: object) -> dict[str, object]:
    return _row_for_schema(
        section_schema(),
        section_id=section_id,
        document_id=f"document-{section_id}",
        article_id=f"article-{section_id}",
        project="wikipedia",
        language=language,
        page_id=1,
        revision_id=1,
        section_index=0,
        text=f"text-{section_id}",
    )

def _link(
    document_id: str, language: object, *, polygon_id: str = "polygon-1"
) -> dict[str, object]:
    return _row_for_schema(
        polygon_document_link_v2_schema(),
        polygon_id=polygon_id,
        document_id=document_id,
        project="wikipedia",
        wikidata="Q1",
        language=language,
        source_pbf="a-latest.osm.pbf",
        region="fixture",
        osm_type="way",
        osm_id=1,
        page_id=1,
        revision_id=1,
        link_sources='["wikidata"]',
    )

def _write_v2_manifest(root: Path, stems: tuple[str, ...]) -> None:
    regions = {
        stem: {
            "source_pbf": f"{stem}.osm.pbf",
            "region": "fixture",
            "polygons_path": f"polygons/{stem}.parquet",
            "documents_path": f"wikipedia/documents/{stem}.parquet",
            "sections_path": f"wikipedia/sections/{stem}.parquet",
            "links_path": f"polygon_document_links/{stem}.parquet",
        }
        for stem in stems
    }
    manifest = {"contract_version": "wikipedia-tags-v2", "regions": regions}
    path = root / "manifests" / "processed_pbfs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

def _write_v2_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "processed_v2"
    stems = ("z-latest", "a-latest")

    _write_table(
        root / "polygons/a-latest.parquet",
        [_polygon("polygon-1", best_language="en")],
        polygon_v2_schema(),
    )
    _write_table(
        root / "polygons/z-latest.parquet",
        [_polygon("polygon-2", best_language="de")],
        polygon_v2_schema(),
    )
    _write_table(
        root / "wikipedia/documents/a-latest.parquet",
        [
            _document("doc-fr-a", "fr", "French A"),
            _document("doc-de-a", "de", "German A"),
            _document("doc-unknown-a", "en/fr", "Unknown A"),
        ],
        wikipedia_document_v2_schema(),
    )
    _write_table(
        root / "wikipedia/documents/z-latest.parquet",
        [_document("doc-fr-z", " FR ", "French Z")],
        wikipedia_document_v2_schema(),
    )
    _write_table(
        root / "wikipedia/sections/a-latest.parquet",
        [
            _section("section-fr-a", "fr"),
            _section("section-de-a", "de"),
            _section("section-unknown-a", None),
        ],
        section_schema(),
    )
    _write_table(
        root / "wikipedia/sections/z-latest.parquet",
        [_section("section-fr-z", "fr")],
        section_schema(),
    )
    _write_table(
        root / "polygon_document_links/a-latest.parquet",
        [
            _link("doc-fr-a", "fr"),
            _link("doc-de-a", "de"),
            _link("doc-missing-a", None),
            _link("doc-blank-a", "  "),
            _link("doc-malformed-a", "en/fr"),
            _link("doc-simple-a", "simple"),
            _link("doc-alias-a", "be_x_old"),
        ],
        polygon_document_link_v2_schema(),
    )
    _write_table(
        root / "polygon_document_links/z-latest.parquet",
        [_link("doc-fr-z", "fr", polygon_id="polygon-2")],
        polygon_document_link_v2_schema(),
    )
    _write_v2_manifest(root, stems)
    return root

def _manifest(root: Path) -> dict[str, object]:
    return json.loads((root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))

def _shard(root: Path, table: str, language: str, stem: str) -> Path:
    configuration = f"{table}_by_language"
    return root / "language_splits" / configuration / f"lang-{language}" / f"{stem}.parquet"

def _rows(root: Path, table: str, language: str) -> list[dict[str, object]]:
    configuration = f"{table}_by_language"
    paths = sorted(
        (root / "language_splits" / configuration / f"lang-{language}").glob("*.parquet")
    )
    return [row for path in paths for row in pq.read_table(path).to_pylist()]

def _release_snapshot(root: Path) -> dict[str, bytes]:
    paths = [
        path
        for directory in (root / "language_splits",)
        for path in directory.rglob("*")
        if path.is_file()
    ]
    manifest = root / LANGUAGE_SPLITS_MANIFEST_RELATIVE_PATH
    if manifest.is_file():
        paths.append(manifest)
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(paths)}

__all__ = [name for name in globals() if not name.startswith("__")]

