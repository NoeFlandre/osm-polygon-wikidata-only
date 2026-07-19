"""Build the checked-in synthetic fixtures used by ``tests/contracts/``.

Run from the repo root:

    uv run python tests/fixtures/_builders.py

This produces the parquet files, the per-region manifest JSON, and
the dataset-card Markdown golden files. The output is committed to
the repository so the contract tests do not need to regenerate
fixtures at runtime.

The fixtures are deliberately tiny — a single fictitious region
("monaco-latest") with one polygon, one linked article, and the
matching minimal augmentation sidecars. They exercise every column
and every type in each public schema without requiring network
access, the external data root, or any caches.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from osm_polygon_wikidata_only.augmentation.models import Document, Section, WikidataFact
from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
    document_schema,
    fact_schema,
    section_schema,
)
from osm_polygon_wikidata_only.domain.schema import (
    ARTICLE_COLUMNS,
    ARTICLE_DESCRIPTIONS,
    POLYGON_ARTICLE_COLUMNS,
    POLYGON_ARTICLE_DESCRIPTIONS,
    POLYGON_COLUMNS,
    POLYGON_DESCRIPTIONS,
    article_schema,
    polygon_article_schema,
    polygon_schema,
)
from osm_polygon_wikidata_only.hf.dataset_card import render_dataset_card

REGION = "monaco-latest"


def _polygon_row() -> dict:
    return {
        "polygon_id": f"{REGION}:way:1",
        "region": "monaco",
        "source_pbf": f"{REGION}.osm.pbf",
        "osm_type": "way",
        "osm_id": 1,
        "wikidata": "Q235",
        "name": "Monaco",
        "tags": json.dumps({"name": "Monaco"}, sort_keys=True),
        "tag_keys": json.dumps(["name"]),
        "tag_count": 1,
        "osm_primary_tag": "",
        "centroid": json.dumps({"type": "Point", "coordinates": [7.42, 43.73]}),
        "lat": 43.73,
        "lon": 7.42,
        "bbox": json.dumps([7.42, 43.73, 7.43, 43.74]),
        "geometry": json.dumps(
            {
                "type": "Polygon",
                "coordinates": [
                    [[7.42, 43.73], [7.43, 43.73], [7.43, 43.74], [7.42, 43.74], [7.42, 43.73]]
                ],
            }
        ),
        "area_m2": 1_000.0,
        "area_km2": 0.001,
        "area_bucket": "100m2-1k_m2",
        "has_name": True,
        "has_wikidata": True,
        "has_wikipedia": True,
        "wikipedia_language_count": 1,
        "wikipedia_languages": json.dumps(["en"]),
        "wikipedia_article_count": 1,
        "has_english_wikipedia": True,
        "has_french_wikipedia": False,
        "text_available": True,
        "best_language": "en",
        "extraction_version": "0.1.0",
        "extracted_at": "2026-01-01T00:00:00Z",
    }


def _article_row() -> dict:
    return {
        "article_id": "Q235:en:1:1",
        "wikidata": "Q235",
        "language": "en",
        "site": "enwiki",
        "title": "Monaco",
        "url": "https://en.wikipedia.org/wiki/Monaco",
        "page_id": 1,
        "revision_id": 1,
        "revision_timestamp": "2026-01-01T00:00:00Z",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "wikidata_label": "Monaco",
        "wikidata_description": "Country in Europe",
        "wikidata_aliases": json.dumps([]),
        "lead_text": "Monaco is a sovereign city-state.",
        "extract": "Monaco is a sovereign city-state.",
        "full_text": "Monaco is a sovereign city-state bordered by France.",
        "full_text_format": "plain_text",
        "article_length_chars": 56,
        "article_length_words": 9,
        "article_length_tokens_estimate": 14,
        "thumbnail_url": "",
        "thumbnail_width": None,
        "thumbnail_height": None,
        "categories": json.dumps([]),
        "license": "CC BY-SA 4.0",
        "attribution": "Wikipedia",
        "source_api": "mediawiki_action_api",
        "fetch_status": "ok",
        "fetch_error": "",
        "content_hash": hashlib.sha256(
            b"Monaco is a sovereign city-state bordered by France."
        ).hexdigest(),
    }


def _link_row() -> dict:
    return {
        "polygon_id": f"{REGION}:way:1",
        "article_id": "Q235:en:1:1",
        "wikidata": "Q235",
        "language": "en",
        "source_pbf": f"{REGION}.osm.pbf",
        "region": "monaco",
        "osm_type": "way",
        "osm_id": 1,
        "page_id": 1,
        "revision_id": 1,
        "is_best_language": True,
    }


def _document_row() -> Document:
    return Document(
        document_id="Q235:wikipedia:en:1:1",
        article_id="Q235:en:1:1",
        wikidata="Q235",
        project="wikipedia",
        language="en",
        site="enwiki",
        title="Monaco",
        url="https://en.wikipedia.org/wiki/Monaco",
        page_id=1,
        revision_id=1,
        revision_timestamp="2026-01-01T00:00:00Z",
        retrieved_at="2026-01-01T00:00:00Z",
        full_text="Monaco is a sovereign city-state bordered by France.",
        full_text_format="plain_text",
        article_length_chars=56,
        article_length_words=9,
        article_length_tokens_estimate=14,
        license="CC BY-SA 4.0",
        attribution="Wikipedia",
        source_api="mediawiki_action_api",
        fetch_status="ok",
        fetch_error="",
        content_hash=hashlib.sha256(
            b"Monaco is a sovereign city-state bordered by France."
        ).hexdigest(),
    )


def _section_row() -> Section:
    return Section(
        section_id="Q235:wikipedia:en:1:1:0",
        document_id="Q235:wikipedia:en:1:1",
        article_id="Q235:en:1:1",
        wikidata="Q235",
        project="wikipedia",
        language="en",
        site="enwiki",
        page_id=1,
        revision_id=1,
        section_index=0,
        heading="Overview",
        anchor="Overview",
        level=1,
        parent_section_id="",
        section_path="Overview",
        text="Monaco is a sovereign city-state bordered by France.",
        text_length_chars=56,
        text_length_words=9,
        text_length_tokens_estimate=14,
        content_hash=hashlib.sha256(
            b"Monaco is a sovereign city-state bordered by France."
        ).hexdigest(),
        license="CC BY-SA 4.0",
        attribution="Wikipedia",
    )


def _fact_row() -> WikidataFact:
    return WikidataFact(
        fact_id="Q235:P31:Q6256:0",
        wikidata="Q235",
        property_id="P31",
        property_label_en="instance of",
        property_labels=json.dumps({"en": "instance of"}),
        value_type="entity",
        value_entity_id="Q6256",
        value_label_en="country",
        value_labels=json.dumps({"en": "country"}),
        value_text="",
        numeric_value=None,
        unit_entity_id="",
        rank="normal",
        qualifiers="{}",
        references="{}",
        retrieved_at="2026-01-01T00:00:00Z",
        source_api="wikidata",
    )


def build_parquets(out: Path) -> dict[str, Path]:
    """Write the six checked-in Parquet fixtures. Returns the relative paths."""
    processed = out / "processed"
    (processed / "polygons").mkdir(parents=True, exist_ok=True)
    (processed / "articles").mkdir(parents=True, exist_ok=True)
    (processed / "polygon_articles").mkdir(parents=True, exist_ok=True)
    (processed / "wikipedia" / "documents").mkdir(parents=True, exist_ok=True)
    (processed / "wikipedia" / "sections").mkdir(parents=True, exist_ok=True)
    (processed / "wikivoyage" / "documents").mkdir(parents=True, exist_ok=True)
    (processed / "wikivoyage" / "sections").mkdir(parents=True, exist_ok=True)
    (processed / "wikidata" / "facts").mkdir(parents=True, exist_ok=True)
    (processed / "manifests").mkdir(parents=True, exist_ok=True)

    paths = {
        "polygons": processed / "polygons" / f"{REGION}.parquet",
        "articles": processed / "articles" / f"{REGION}.parquet",
        "polygon_articles": processed / "polygon_articles" / f"{REGION}.parquet",
        "wikipedia_documents": processed / "wikipedia" / "documents" / f"{REGION}.parquet",
        "wikipedia_sections": processed / "wikipedia" / "sections" / f"{REGION}.parquet",
        "wikivoyage_documents": processed / "wikivoyage" / "documents" / f"{REGION}.parquet",
        "wikivoyage_sections": processed / "wikivoyage" / "sections" / f"{REGION}.parquet",
        "wikidata_facts": processed / "wikidata" / "facts" / f"{REGION}.parquet",
        "manifest": processed / "manifests" / "processed_pbfs.json",
    }

    pq.write_table(
        pa.Table.from_pylist([_polygon_row()], schema=polygon_schema()),
        paths["polygons"],
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([_article_row()], schema=article_schema()),
        paths["articles"],
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([_link_row()], schema=polygon_article_schema()),
        paths["polygon_articles"],
        compression="snappy",
    )

    doc = _document_row().to_dict()
    pq.write_table(
        pa.Table.from_pylist([doc], schema=document_schema()),
        paths["wikipedia_documents"],
        compression="snappy",
    )
    sec = _section_row().to_dict()
    pq.write_table(
        pa.Table.from_pylist([sec], schema=section_schema()),
        paths["wikipedia_sections"],
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([], schema=document_schema()),
        paths["wikivoyage_documents"],
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([], schema=section_schema()),
        paths["wikivoyage_sections"],
        compression="snappy",
    )
    fact = _fact_row().to_dict()
    pq.write_table(
        pa.Table.from_pylist([fact], schema=fact_schema()),
        paths["wikidata_facts"],
        compression="snappy",
    )

    paths["manifest"].write_text(
        json.dumps(
            {
                f"{REGION}.osm.pbf": {
                    "source_pbf": f"{REGION}.osm.pbf",
                    "region": "monaco",
                    "polygons_path": f"polygons/{REGION}.parquet",
                    "articles_path": f"articles/{REGION}.parquet",
                    "polygon_articles_path": f"polygon_articles/{REGION}.parquet",
                    "extraction_version": "0.1.0",
                    "processed_at": "2026-01-01T00:00:00Z",
                    "polygon_count": 1,
                    "unique_wikidata_count": 1,
                    "article_count": 1,
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {k: v.relative_to(out) for k, v in paths.items()}


def _field_metadata(metadata: dict | None) -> dict[str, str]:
    """Decode a pyarrow ``Field.metadata`` dict into a JSON-friendly form.

    PyArrow stores metadata keys/values as ``bytes``; downstream tests
    read them as plain ``str`` for stability.
    """
    if not metadata:
        return {}
    decoded: dict[str, str] = {}
    for key, value in metadata.items():
        decoded[key.decode() if isinstance(key, bytes) else str(key)] = (
            value.decode() if isinstance(value, bytes) else str(value)
        )
    return decoded


def build_schema_snapshots(out: Path) -> dict[str, Path]:
    """Write JSON snapshots of every public schema."""
    snap = {}
    for name, schema_fn, columns in (
        ("polygon", polygon_schema, POLYGON_COLUMNS),
        ("article", article_schema, ARTICLE_COLUMNS),
        ("polygon_article", polygon_article_schema, POLYGON_ARTICLE_COLUMNS),
        ("document", document_schema, DOCUMENT_COLUMNS),
        ("section", section_schema, SECTION_COLUMNS),
        ("fact", fact_schema, FACT_COLUMNS),
    ):
        schema = schema_fn()
        fields = [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": bool(field.nullable),
                "metadata": _field_metadata(field.metadata),
            }
            for field in schema
        ]
        path = out / "schemas" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"columns": list(columns), "fields": fields}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        snap[name] = path.relative_to(out)
    return snap


def build_golden_card(out: Path) -> Path:
    """Capture the dataset card Markdown produced by the current renderer."""
    card = render_dataset_card(
        repo_id="NoeFlandre/osm-polygon-wikidata-only",
        stats={"polygon_count": 1, "article_count": 1, "unique_wikidata_count": 1},
        polygon_columns=list(POLYGON_COLUMNS),
        polygon_descriptions=POLYGON_DESCRIPTIONS,
        article_columns=list(ARTICLE_COLUMNS),
        article_descriptions=ARTICLE_DESCRIPTIONS,
        link_columns=list(POLYGON_ARTICLE_COLUMNS),
        link_descriptions=POLYGON_ARTICLE_DESCRIPTIONS,
        maintainer="Noé Flandre",
    )
    path = out / "golden" / "dataset_card.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card, encoding="utf-8")
    return path.relative_to(out)


def build_golden_publication(out: Path) -> Path:
    """Capture the unified-sync upload file list shape."""
    pub = {
        "core": [
            "polygons/monaco-latest.parquet",
            "polygon_articles/monaco-latest.parquet",
            "manifests/processed_pbfs.json",
            "assets/geographic_text_presence.png",
            "assets/geographic_wikipedia_text_coverage.png",
            "assets/geographic_polygon_count.png",
            "assets/coverage_map.png",
            "coverage_map.png",
        ],
        "augmentation": [
            "wikipedia/documents/monaco-latest.parquet",
            "articles/monaco-latest.parquet",
            "wikipedia/sections/monaco-latest.parquet",
            "wikivoyage/documents/monaco-latest.parquet",
            "wikivoyage/sections/monaco-latest.parquet",
            "wikidata/facts/monaco-latest.parquet",
            "manifests/augmentation_manifest.json",
            "augmentation/manifests/augmentation_manifest.json",
            "README.md",
        ],
    }
    path = out / "golden" / "publication_file_list.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pub, indent=2, sort_keys=True), encoding="utf-8")
    return path.relative_to(out)


def build_golden_help(out: Path) -> dict[str, Path]:
    """Capture the full normalized CLI help text for every command.

    The output is normalized so the program name is ``PROG`` and
    long option lists wrap at 100 columns. The result is the
    checked-in golden file used by the CLI contract tests.
    """
    from osm_polygon_wikidata_only.cli.parser import build_parser

    class _FrozenFormatter(argparse.HelpFormatter):
        def __init__(self, prog: str) -> None:
            super().__init__(prog, width=100, max_help_position=30)

    parser = build_parser()
    parser.formatter_class = _FrozenFormatter
    sub_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for sub_parser in sub_action.choices.values():
        sub_parser.formatter_class = _FrozenFormatter

    def _capture(prog_args: list[str]) -> str:
        import pytest

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with pytest.raises(SystemExit) as exc:
                parser.parse_args([*prog_args, "--help"])
        assert exc.value.code == 0
        text = buffer.getvalue()
        text = re.sub(r"^usage:\s+\S+\s+", "usage: PROG ", text, flags=re.MULTILINE)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"

    out_paths: dict[str, Path] = {}
    targets = {
        "root": [],
        "process-pbf": ["process-pbf"],
        "process-dir": ["process-dir"],
        "sync-dir": ["sync-dir"],
        "augment-region": ["augment-region"],
        "augment-dir": ["augment-dir"],
    }
    for name, prog_args in targets.items():
        path = out / "golden" / f"cli_help_{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_capture(prog_args), encoding="utf-8")
        out_paths[name] = path.relative_to(out)
    return out_paths


def main() -> None:
    out = Path(__file__).resolve().parent
    parquets = build_parquets(out)
    schemas = build_schema_snapshots(out)
    card = build_golden_card(out)
    publication = build_golden_publication(out)
    help_files = build_golden_help(out)
    summary = {
        "region": REGION,
        "parquets": {k: str(v) for k, v in parquets.items()},
        "schemas": {k: str(v) for k, v in schemas.items()},
        "golden_card": str(card),
        "golden_publication": str(publication),
        "golden_help": {k: str(v) for k, v in help_files.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
