"""TDD contracts for the end-to-end automatic silver evaluator."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.evaluate_geographic_ner_pilot import (
    _add_group_metrics,
    _aliases,
    _available,
    _clean_names,
    _document_columns,
    _document_names,
    _document_values,
    _entities,
    _identity,
    _merge_link_reference,
    _polygon_names,
    _processed_root,
    _read_json,
    _read_link_rows,
    _read_rows,
    _reference_parts,
    _source_identity,
    _string_aliases,
    _successful_document,
    evaluate,
)


def _write_fixture(root: Path, pilot: Path, primary: Path, secondary: Path) -> None:
    processed = root / "processed_v2"
    for directory in (
        processed / "wikipedia/sentences",
        processed / "wikipedia/documents",
        processed / "polygon_document_links",
        processed / "polygons",
    ):
        directory.mkdir(parents=True)
    sentence = pa.table(
        {
            "sentence_id": ["s1"],
            "document_id": ["doc-1"],
            "project": ["wikipedia"],
            "language": ["en"],
            "text": ["Paris is a city"],
            "segmentation_status": ["split"],
        }
    )
    pq.write_table(sentence, processed / "wikipedia/sentences/region.parquet")
    pq.write_table(
        pa.table(
            {
                "document_id": ["doc-1"],
                "title": ["Paris"],
                "wikidata_label": ["Paris"],
                "wikidata_aliases": ['["City of Paris"]'],
                "fetch_status": ["ok"],
                "full_text": ["Paris is the capital."],
            }
        ),
        processed / "wikipedia/documents/region.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "document_id": ["doc-1"],
                "project": ["wikipedia"],
                "osm_type": ["relation"],
                "osm_id": [123],
            }
        ),
        processed / "polygon_document_links/region.parquet",
    )
    pq.write_table(
        pa.table({"osm_type": ["relation"], "osm_id": [123], "name": ["Paris"]}),
        processed / "polygons/region.parquet",
    )
    pilot.mkdir()
    pq.write_table(sentence, pilot / "input.parquet")
    (pilot / "selection.json").write_text(
        json.dumps({"source_files": ["wikipedia/sentences/region.parquet"]})
    )
    for output, entities in (
        (
            primary,
            [
                {
                    "text": "Paris",
                    "label": "named geographic location",
                    "start": 0,
                    "end": 5,
                    "score": 0.9,
                },
                {
                    "text": "Q123",
                    "label": "named geographic location",
                    "start": 6,
                    "end": 10,
                    "score": 0.7,
                },
            ],
        ),
        (
            secondary,
            [
                {
                    "text": "Paris",
                    "label": "named geographic location",
                    "start": 0,
                    "end": 5,
                    "score": 0.8,
                }
            ],
        ),
    ):
        output.mkdir()
        pq.write_table(
            pa.table(
                {
                    "sentence_id": ["s1"],
                    "document_id": ["doc-1"],
                    "project": ["wikipedia"],
                    "language": ["en"],
                    "status": ["ok"],
                    "validation_status": ["pilot_unvalidated"],
                    "contract_id": ["a" * 64],
                    "entities": [entities],
                }
            ),
            output / "batch-000000.parquet",
        )


def test_evaluate_reports_silver_signals_and_linked_polygon_counts(tmp_path) -> None:
    pilot = tmp_path / "pilot"
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _write_fixture(tmp_path, pilot, primary, secondary)

    result = evaluate(tmp_path, pilot, primary, secondary)

    assert result["status"] == "silver_unvalidated"
    assert result["sample_size"] == 1
    assert result["unique_documents"] == 1
    assert result["unique_polygons"] == 1
    assert result["primary_entities"] == 2
    assert result["primary_artifacts"] == 1
    assert result["primary_known_name_matches"] == 1
    assert result["primary_unmatched"] == 0
    assert result["secondary_entities"] == 1
    assert result["model_agreements"] == 1
    assert result["entity_bearing_polygons"] == 1
    assert result["by_language"]["en"]["rows"] == 1
    assert result["by_project"]["wikipedia"]["rows"] == 1


def test_evaluate_accepts_the_processed_v2_directory_directly(tmp_path) -> None:
    pilot = tmp_path / "pilot"
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _write_fixture(tmp_path, pilot, primary, secondary)

    assert _processed_root(tmp_path / "processed_v2") == tmp_path / "processed_v2"


def test_processed_root_rejects_a_missing_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="processed_v2 directory is missing"):
        _processed_root(tmp_path)


def test_identity_and_entities_use_the_complete_prediction_key() -> None:
    row = {
        "sentence_id": "sentence",
        "document_id": "document",
        "project": "project",
        "language": "language",
        "entities": [{"text": "Paris"}],
    }

    assert _identity(row) == ("sentence", "document", "project", "language")
    assert _entities(row) == [{"text": "Paris"}]
    assert _entities({}) == []
    with pytest.raises(ValueError, match="list of objects"):
        _entities({"entities": {"text": "Paris"}})
    with pytest.raises(ValueError, match="list of objects"):
        _entities({"entities": [{"text": "Paris"}, "invalid"]})


def test_source_identity_rejects_non_sentence_parquet_paths() -> None:
    assert _source_identity("wikipedia/sentences/region.parquet") == (
        "wikipedia",
        "region",
    )
    for invalid in (
        "region.parquet",
        "wikipedia/documents/region.parquet",
        "wikipedia/sentences/region.json",
        "wikipedia/sentences/region.parquet/extra",
    ):
        with pytest.raises(ValueError, match="Invalid selected sentence path"):
            _source_identity(invalid)


def test_read_helpers_preserve_requested_columns_and_validate_json(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"first": [1], "second": [2]}), path)

    assert _read_rows(path, ["first"]) == [{"first": 1}]
    with pytest.raises(ValueError, match="Missing Parquet input"):
        _read_rows(tmp_path / "missing.parquet")

    json_path = tmp_path / "value.json"
    json_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        _read_json(json_path)


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"fetch_status": "ok", "full_text": " text "}, True),
        ({"fetch_status": "ok", "full_text": "   "}, False),
        ({"fetch_status": "ok", "full_text": None}, False),
        ({"fetch_status": "ok"}, False),
        ({"fetch_status": "failed", "full_text": "text"}, False),
        ({"full_text": "text"}, False),
    ],
)
def test_successful_document_requires_ok_status_and_non_empty_text(row, expected) -> None:
    assert _successful_document(row) is expected


def test_document_reference_helpers_collect_titles_labels_and_aliases(tmp_path) -> None:
    root = tmp_path / "processed_v2"
    root.mkdir()
    document_path = root / "documents.parquet"
    pq.write_table(
        pa.table(
            {
                "document_id": ["doc-1", "doc-2", "doc-3"],
                "title": ["Paris", "Failed", "Empty"],
                "wikidata_label": ["Paris", "Failed", "Empty"],
                "wikidata_aliases": ['["City"]', "[]", "[]"],
                "fetch_status": ["ok", "failed", "ok"],
                "full_text": ["text", "text", "  "],
            }
        ),
        document_path,
    )

    assert _document_columns(document_path) == {
        "document_id",
        "title",
        "wikidata_label",
        "wikidata_aliases",
        "fetch_status",
        "full_text",
    }
    assert _document_names(document_path) == {"doc-1": ("City", "Paris")}
    assert _document_values(
        {"title": "Paris", "wikidata_label": "Île", "wikidata_aliases": '["City"]'},
        {"title", "wikidata_label", "wikidata_aliases"},
    ) == ("Paris", "Île", "City")
    assert _document_values({"title": "Paris"}, {"title"}) == ("Paris",)

    incomplete = tmp_path / "incomplete.parquet"
    pq.write_table(pa.table({"document_id": ["doc-1"], "full_text": ["text"]}), incomplete)
    with pytest.raises(ValueError, match="missing silver reference columns"):
        _document_columns(incomplete)


def test_alias_and_name_cleaning_reject_malformed_values() -> None:
    assert _aliases('["Paris", "Lutetia"]') == ("Paris", "Lutetia")
    assert _aliases(["Paris"]) == ("Paris",)
    assert _aliases("not-json") == ()
    assert _aliases(42) == ()
    assert _string_aliases(["Paris", "Lutetia"]) == ("Paris", "Lutetia")
    assert _string_aliases(["Paris", 42]) == ()
    assert _string_aliases(("Paris",)) == ()
    assert _clean_names(["Paris", " ", "", 42, None]) == ("Paris",)


def test_reference_and_link_helpers_preserve_polygon_identity(tmp_path) -> None:
    pilot = tmp_path / "pilot"
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _write_fixture(tmp_path, pilot, primary, secondary)
    processed = tmp_path / "processed_v2"

    parts = list(_reference_parts(processed, ["wikipedia/sentences/region.parquet"]))
    assert len(parts) == 1
    assert parts[0][0] == "wikipedia"
    assert parts[0][1] == {"doc-1": ("City of Paris", "Paris")}
    assert parts[0][2] == {("relation", "123"): ("Paris",)}
    assert parts[0][3] == [{"document_id": "doc-1", "osm_type": "relation", "osm_id": "123"}]

    names = defaultdict(set)
    polygons_by_document = defaultdict(set)
    polygons = set()
    _merge_link_reference(
        "wikipedia",
        {"document_id": "doc-1", "osm_type": "relation", "osm_id": "123"},
        {"doc-1": ("Paris",)},
        {("relation", "123"): ("Paris",)},
        names,
        polygons_by_document,
        polygons,
    )
    _merge_link_reference(
        "wikipedia",
        {"document_id": "doc-2", "osm_type": "way", "osm_id": "456"},
        {},
        {},
        names,
        polygons_by_document,
        polygons,
    )
    _merge_link_reference(
        "wikipedia",
        {"document_id": "doc-3", "osm_type": "way", "osm_id": "789"},
        {"doc-3": ("Document name",)},
        {("way", "789"): ("Polygon name",)},
        names,
        polygons_by_document,
        polygons,
    )
    assert names[("wikipedia", "doc-1")] == {"Paris"}
    assert names[("wikipedia", "doc-3")] == {"Document name", "Polygon name"}
    assert polygons == {
        ("relation", "123"),
        ("way", "456"),
        ("way", "789"),
    }
    assert polygons_by_document[("wikipedia", "doc-2")] == {("way", "456")}


def test_link_and_polygon_readers_validate_columns_and_clean_names(tmp_path) -> None:
    links = tmp_path / "links.parquet"
    pq.write_table(
        pa.table(
            {
                "document_id": ["doc-1"],
                "osm_type": ["relation"],
                "osm_id": [123],
            }
        ),
        links,
    )
    assert _read_link_rows(links) == [
        {"document_id": "doc-1", "osm_type": "relation", "osm_id": "123"}
    ]

    polygons = tmp_path / "polygons.parquet"
    pq.write_table(
        pa.table(
            {
                "osm_type": ["relation", "way", "node"],
                "osm_id": [1, 2, 3],
                "name": ["Paris", " ", None],
            }
        ),
        polygons,
    )
    assert _polygon_names(polygons) == {("relation", "1"): ("Paris",)}
    assert _available(polygons, ("osm_type", "missing")) == {"osm_type"}

    incomplete = tmp_path / "incomplete-links.parquet"
    pq.write_table(pa.table({"document_id": ["doc-1"]}), incomplete)
    with pytest.raises(ValueError, match="missing polygon identity columns"):
        _read_link_rows(incomplete)

    incomplete_polygon = tmp_path / "incomplete-polygons.parquet"
    pq.write_table(pa.table({"osm_type": ["relation"], "osm_id": [1]}), incomplete_polygon)
    with pytest.raises(ValueError, match="missing silver reference columns"):
        _polygon_names(incomplete_polygon)


def test_group_metrics_accumulate_rows_and_metric_values() -> None:
    target = Counter({"rows": 5})

    _add_group_metrics(target, {"primary_entities": 2})

    assert target == Counter({"rows": 6, "primary_entities": 2})


def test_reference_parts_rejects_invalid_selection_source_files(tmp_path) -> None:
    for invalid in (None, [1]):
        with pytest.raises(ValueError, match="source_files must be a list of strings"):
            list(_reference_parts(tmp_path, invalid))


def test_evaluate_counts_references_and_unreferenced_rows_precisely(tmp_path) -> None:
    pilot = tmp_path / "pilot"
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _write_fixture(tmp_path, pilot, primary, secondary)
    processed = tmp_path / "processed_v2"

    documents_path = processed / "wikipedia/documents/region.parquet"
    documents = pq.read_table(documents_path)
    extra_document = pa.Table.from_pylist(
        [
            {
                "document_id": "doc-2",
                "title": "Hidden",
                "wikidata_label": "Hidden",
                "wikidata_aliases": "[]",
                "fetch_status": "failed",
                "full_text": "ignored",
            }
        ],
        schema=documents.schema,
    )
    pq.write_table(pa.concat_tables([documents, extra_document]), documents_path)

    links_path = processed / "polygon_document_links/region.parquet"
    links = pq.read_table(links_path)
    extra_link = pa.Table.from_pylist(
        [
            {
                "document_id": "doc-2",
                "project": "wikipedia",
                "osm_type": "relation",
                "osm_id": 456,
            }
        ],
        schema=links.schema,
    )
    pq.write_table(pa.concat_tables([links, extra_link]), links_path)

    polygons_path = processed / "polygons/region.parquet"
    polygons = pq.read_table(polygons_path)
    extra_polygon = pa.Table.from_pylist(
        [{"osm_type": "relation", "osm_id": 456, "name": None}], schema=polygons.schema
    )
    pq.write_table(pa.concat_tables([polygons, extra_polygon]), polygons_path)

    input_table = pq.read_table(pilot / "input.parquet")
    extra_input = pa.Table.from_pylist(
        [
            {
                "sentence_id": "s2",
                "document_id": "doc-2",
                "project": "wikipedia",
                "language": "en",
                "text": "Hidden",
                "segmentation_status": "split",
            },
            {
                "sentence_id": "s3",
                "document_id": "doc-3",
                "project": "wikipedia",
                "language": "en",
                "text": "Berlin",
                "segmentation_status": "split",
            },
        ],
        schema=input_table.schema,
    )
    pq.write_table(pa.concat_tables([input_table, extra_input]), pilot / "input.parquet")

    primary_table = pq.read_table(primary / "batch-000000.parquet")
    prediction_schema = primary_table.schema
    extra_primary = pa.Table.from_pylist(
        [
            {
                "sentence_id": "s2",
                "document_id": "doc-2",
                "project": "wikipedia",
                "language": "en",
                "status": "ok",
                "validation_status": "pilot_unvalidated",
                "contract_id": "a" * 64,
                "entities": [
                    {
                        "text": "Q999",
                        "label": "named geographic location",
                        "start": 0,
                        "end": 4,
                        "score": 0.8,
                    }
                ],
            },
            {
                "sentence_id": "s3",
                "document_id": "doc-3",
                "project": "wikipedia",
                "language": "en",
                "status": "ok",
                "validation_status": "pilot_unvalidated",
                "contract_id": "a" * 64,
                "entities": [
                    {
                        "text": "Berlin",
                        "label": "named geographic location",
                        "start": 0,
                        "end": 6,
                        "score": 0.8,
                    }
                ],
            },
        ],
        schema=prediction_schema,
    )
    pq.write_table(
        pa.concat_tables([primary_table, extra_primary]), primary / "batch-000000.parquet"
    )

    secondary_table = pq.read_table(secondary / "batch-000000.parquet")
    extra_secondary = pa.Table.from_pylist(
        [
            {
                "sentence_id": "s2",
                "document_id": "doc-2",
                "project": "wikipedia",
                "language": "en",
                "status": "ok",
                "validation_status": "pilot_unvalidated",
                "contract_id": "a" * 64,
                "entities": [],
            },
            {
                "sentence_id": "s3",
                "document_id": "doc-3",
                "project": "wikipedia",
                "language": "en",
                "status": "ok",
                "validation_status": "pilot_unvalidated",
                "contract_id": "a" * 64,
                "entities": [],
            },
        ],
        schema=secondary_table.schema,
    )
    pq.write_table(
        pa.concat_tables([secondary_table, extra_secondary]),
        secondary / "batch-000000.parquet",
    )

    result = evaluate(tmp_path, pilot, primary, secondary)

    assert result["sample_size"] == 3
    assert result["unique_documents"] == 3
    assert result["reference_rows"] == 1
    assert result["unique_polygons"] == 2
    assert result["entity_bearing_polygons"] == 1
    assert result["primary_entities"] == 4
    assert result["primary_artifacts"] == 2
    assert result["by_language"]["en"]["rows"] == 3
    assert result["by_language"]["en"]["primary_entities"] == 4
    assert result["by_project"]["wikipedia"]["primary_entities"] == 4
    assert result["secondary_supported_languages"] == ["de", "en", "es", "fr", "pt", "ru"]
