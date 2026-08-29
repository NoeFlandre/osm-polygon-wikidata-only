from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import wikipedia_document_schema
from osm_polygon_wikidata_only.domain.schema import empty_row, polygon_schema
from osm_polygon_wikidata_only.v2 import card
from osm_polygon_wikidata_only.v2.card import (
    _word_column,
    compute_v2_card_stats,
    render_v2_card,
    write_v2_card,
)
from osm_polygon_wikidata_only.v2.schema import wikipedia_document_v2_schema
from osm_polygon_wikidata_only.v2.storage import write_v2_region


def test_card_is_factual_and_deterministic(tmp_path: Path) -> None:
    write_v2_region(
        tmp_path,
        "region-latest",
        polygons=[{"polygon_id": "p1", "has_wikidata": False}],
        documents=[],
        links=[],
    )
    first = render_v2_card(tmp_path)
    second = render_v2_card(tmp_path)
    assert first == second
    assert "Wikipedia-tag-only polygons:** 1" in first
    assert "exact V1 22-column section schema" in first
    assert "--dataset-version v2" in first
    assert "external drive" not in first
    assert first.rstrip().endswith(
        "Download the dataset citation metadata from [`CITATION.cff`](CITATION.cff)."
    )
    assert first.index("## Citation") > first.index("## Reproducibility")
    assert write_v2_card(tmp_path).read_text(encoding="utf-8") == first


def test_card_documents_exact_sentence_split_scope_when_sidecars_exist(tmp_path: Path) -> None:
    write_v2_region(tmp_path, "region-latest", polygons=[], documents=[], links=[])
    sentence_path = tmp_path / "wikipedia/sentences/region-latest.parquet"
    sentence_path.parent.mkdir(parents=True)
    sentence_path.touch()

    card_text = render_v2_card(tmp_path)

    assert "sat-3l-sm" in card_text
    assert "exact ISO codes listed in `docs/sentence-splitting.md`" in card_text
    assert "one unsplit row" in card_text
    assert "segmentation_status=unsupported_language" in card_text
    assert "never passed to SaT" in card_text
    assert "wikipedia/sentences/<stem>.parquet" in card_text
    assert "wikipedia_sentences" in card_text


def test_word_column_prefers_article_length_and_has_legacy_fallback() -> None:
    assert _word_column({"article_length_words", "text_length_words"}) == ("article_length_words")
    assert _word_column({"text_length_words"}) == "text_length_words"
    assert _word_column({"document_id"}) is None


def test_record_numeric_value_ignores_empty_ids_and_rejects_conflicts() -> None:
    values: dict[str, int] = {}
    card._record_numeric_value(values, None, 7, "words")
    card._record_numeric_value(values, "", 7, "words")
    card._record_numeric_value(values, "doc", None, "words")
    card._record_numeric_value(values, "doc", 0, "words")
    assert values == {"doc": 0}
    with pytest.raises(ValueError, match="Inconsistent words"):
        card._record_numeric_value(values, "doc", 1, "words")


def test_sum_first_available_file_uses_first_present_word_column(tmp_path: Path) -> None:
    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({"document_id": ["a", "b"], "article_length_words": [3, None]}), path)
    assert card._sum_first_available_file(path, ("article_length_words", "text_length_words")) == 3
    assert card._sum_first_available_file(path, ("missing",)) == 0


def test_card_metrics_scan_document_columns_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "documents.parquet"
    pq.write_table(
        pa.table(
            {
                "document_id": ["doc-1"],
                "language": ["en"],
                "article_length_words": [4],
            }
        ),
        path,
    )
    files = card._CardFiles(
        stems=(),
        polygon_files=[],
        wikipedia_document_files=[path],
        wikipedia_section_files=[],
        wikivoyage_document_files=[],
        wikivoyage_section_files=[],
        wikidata_fact_files=[],
        link_files=[],
        parquet_files=(path,),
    )

    def fail_if_document_scan_is_repeated(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("document columns must be scanned once")

    monkeypatch.setattr(card, "_text_document_languages", fail_if_document_scan_is_repeated)
    monkeypatch.setattr(card, "_wikipedia_language_counts", fail_if_document_scan_is_repeated)
    monkeypatch.setattr(card, "_sum_first_available", fail_if_document_scan_is_repeated)
    original_unique_values = card._unique_values

    def fail_if_document_values_are_rescanned(paths, column):
        if path in tuple(paths):
            raise AssertionError("document columns must be scanned once")
        return original_unique_values(paths, column)

    monkeypatch.setattr(card, "_unique_values", fail_if_document_values_are_rescanned)

    metrics = card._compute_card_metrics(files)

    assert metrics.document_ids == {"doc-1"}
    assert metrics.languages == {"en"}
    assert metrics.document_words == 4
    assert metrics.top_languages == (("en", 1),)


def test_card_metrics_scan_polygon_columns_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "polygons.parquet"
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["polygon-1"],
                "wikidata": ["Q42"],
                "has_wikidata": [False],
            }
        ),
        path,
    )
    files = card._CardFiles(
        stems=(),
        polygon_files=[path],
        wikipedia_document_files=[],
        wikipedia_section_files=[],
        wikivoyage_document_files=[],
        wikivoyage_section_files=[],
        wikidata_fact_files=[],
        link_files=[],
        parquet_files=(path,),
    )

    def fail_if_polygon_scan_is_repeated(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("polygon columns must be scanned once")

    monkeypatch.setattr(card, "_unique_values", fail_if_polygon_scan_is_repeated)
    monkeypatch.setattr(card, "_unique_qids", fail_if_polygon_scan_is_repeated)
    monkeypatch.setattr(card, "_count_boolean_false", fail_if_polygon_scan_is_repeated)

    metrics = card._compute_card_metrics(files)

    assert metrics.polygon_ids == {"polygon-1"}
    assert metrics.qids == {"Q42"}
    assert metrics.wikipedia_tag_only == 1


def test_card_metrics_preserve_rows_when_metric_columns_are_missing(tmp_path: Path) -> None:
    polygon_path = tmp_path / "polygons.parquet"
    document_path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({"unrelated": [1, 2]}), polygon_path)
    pq.write_table(pa.table({"unrelated": [1, 2, 3]}), document_path)
    files = card._CardFiles(
        stems=(),
        polygon_files=[polygon_path],
        wikipedia_document_files=[document_path],
        wikipedia_section_files=[],
        wikivoyage_document_files=[],
        wikivoyage_section_files=[],
        wikidata_fact_files=[],
        link_files=[],
        parquet_files=(polygon_path, document_path),
    )

    metrics = card._compute_card_metrics(files)

    assert metrics.polygon_row_count == 2
    assert metrics.wikipedia_document_row_count == 3


def test_sum_metadata_uses_bounded_parallel_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"table-{index}.parquet"
        pq.write_table(pa.table({"value": [index]}), path)
        paths.append(path)

    original_executor = card.ThreadPoolExecutor
    worker_counts: list[int] = []

    def tracking_executor(*args: object, **kwargs: object):
        worker_counts.append(int(kwargs["max_workers"]))
        return original_executor(*args, **kwargs)

    monkeypatch.setattr(card, "ThreadPoolExecutor", tracking_executor)

    assert card._sum_metadata(paths) == 3
    assert worker_counts == [3]


def test_build_card_stats_reuses_collected_row_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = card._CardFiles((), [], [], [], [], [], [], [], ())
    metrics = card._compute_card_metrics(files)

    def fail_if_counts_are_read_again(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("card row counts must not be read twice")

    monkeypatch.setattr(card, "_sum_metadata", fail_if_counts_are_read_again)

    snapshot = card._build_card_stats(files, metrics, card._V1Comparison())

    assert snapshot.polygons == 0
    assert snapshot.wikipedia_documents == 0
    assert snapshot.wikivoyage_documents == 0
    assert snapshot.wikipedia_sections == 0
    assert snapshot.wikivoyage_sections == 0
    assert snapshot.wikidata_facts == 0
    assert snapshot.polygon_document_links == 0


def test_unique_values_reuses_the_open_parquet_file_for_schema_and_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({"language": ["en", "fr"]}), path)

    def fail_if_opened_separately(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("schema inspection must reuse the row-reading handle")

    monkeypatch.setattr(card.pq, "read_schema", fail_if_opened_separately)

    assert card._unique_values_file(path, "language") == {"en", "fr"}


def test_validated_source_list_rejects_non_lists_and_non_strings() -> None:
    assert card._validated_source_list(["wikidata"], "p1", "sources") == ["wikidata"]
    with pytest.raises(ValueError, match="Invalid sources"):
        card._validated_source_list({"source": "wikidata"}, "p1", "sources")
    with pytest.raises(ValueError, match="Invalid sources"):
        card._validated_source_list(["wikidata", 1], "p1", "sources")


def test_write_v2_card_preserves_previous_card_when_atomic_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed card write must leave the previous README intact."""
    target = tmp_path / "README.md"
    target.write_text("previous card\n", encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(card, "atomic_write_text", fail, raising=False)

    with pytest.raises(OSError, match="disk full"):
        write_v2_card(tmp_path)

    assert target.read_text(encoding="utf-8") == "previous card\n"
    assert not list(tmp_path.glob(".README.md.*.tmp"))


def _empty_row(schema: pa.Schema) -> dict[str, object]:
    return {field.name: None for field in schema}


def test_v2_card_reports_word_and_section_deltas_from_v1(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"

    v1_document_schema = wikipedia_document_schema()
    v1_document = _empty_row(v1_document_schema)
    v1_document.update({"document_id": "v1-doc", "article_length_words": 4})
    v1_document_path = v1 / "wikipedia" / "documents" / "region-latest.parquet"
    v1_document_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([v1_document], schema=v1_document_schema),
        v1_document_path,
    )

    v1_section_schema = section_schema()
    v1_section = _empty_row(v1_section_schema)
    v1_section.update({"section_id": "v1-section", "document_id": "v1-doc"})
    v1_section_path = v1 / "wikipedia" / "sections" / "region-latest.parquet"
    v1_section_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([v1_section], schema=v1_section_schema), v1_section_path)

    v2_document_schema = wikipedia_document_v2_schema()
    v2_document = _empty_row(v2_document_schema)
    v2_document.update({"document_id": "v2-doc", "article_length_words": 10})
    v2_section = _empty_row(v1_section_schema)
    v2_section.update({"section_id": "v2-section", "document_id": "v2-doc"})
    write_v2_region(
        v2,
        "region-latest",
        polygons=[],
        documents=[v2_document],
        links=[],
        sections=[v2_section],
    )

    stats = compute_v2_card_stats(v2, v1_processed=v1)

    assert stats.additional_document_words_vs_v1 == 6
    assert stats.additional_sections_vs_v1 == 0
    card_text = render_v2_card(v2, v1_processed=v1, stats=stats)
    assert "**Additional document-row words in V2 (Wikipedia + Wikivoyage):** 6" in card_text
    assert "**Additional section rows in V2 (Wikipedia + Wikivoyage):** 0" in card_text


def test_v2_card_reports_source_split_and_unique_content_deltas(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"

    v1_polygon = empty_row(tuple(field.name for field in polygon_schema()))
    v1_polygon.update({"polygon_id": "region-latest:way:1"})
    v1_polygon_path = v1 / "polygons" / "region-latest.parquet"
    v1_polygon_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([v1_polygon], schema=polygon_schema()),
        v1_polygon_path,
    )

    v1_document_schema = wikipedia_document_schema()
    v1_document = _empty_row(v1_document_schema)
    v1_document.update(
        {
            "document_id": "old-doc",
            "article_length_words": 4,
            "content_hash": "shared-content",
        }
    )
    v1_document_path = v1 / "wikipedia" / "documents" / "region-latest.parquet"
    v1_document_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([v1_document], schema=v1_document_schema),
        v1_document_path,
    )

    v1_section_schema = section_schema()
    v1_section = _empty_row(v1_section_schema)
    v1_section.update({"section_id": "old-section", "document_id": "old-doc"})
    v1_section_path = v1 / "wikipedia" / "sections" / "region-latest.parquet"
    v1_section_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([v1_section], schema=v1_section_schema),
        v1_section_path,
    )

    def v2_polygon(polygon_id: str, sources: str) -> dict[str, object]:
        row = empty_row(tuple(field.name for field in polygon_schema()))
        row.update(
            {
                "polygon_id": polygon_id,
                "discovery_sources": sources,
                "has_wikidata": "wikidata" in sources,
            }
        )
        return row

    v2_document_schema = wikipedia_document_v2_schema()
    v2_document = _empty_row(v2_document_schema)
    v2_document.update(
        {
            "document_id": "new-doc",
            "article_length_words": 10,
            "content_hash": "shared-content",
        }
    )
    v2_section = _empty_row(v1_section_schema)
    v2_section.update({"section_id": "new-section", "document_id": "new-doc"})
    write_v2_region(
        v2,
        "region-latest",
        polygons=[
            v2_polygon("region-latest:way:1", '["wikidata"]'),
            v2_polygon("region-latest:way:2", '["wikipedia_tag"]'),
            v2_polygon("region-latest:way:3", '["wikidata","wikipedia_tag"]'),
            v2_polygon("region-latest:way:4", '["wikidata"]'),
        ],
        documents=[v2_document],
        links=[
            {
                "polygon_id": "region-latest:way:2",
                "document_id": "new-doc",
                "project": "wikipedia",
                "link_sources": '["osm_wikipedia_tag"]',
            },
            {
                "polygon_id": "region-latest:way:3",
                "document_id": "new-doc",
                "project": "wikipedia",
                "link_sources": '["osm_wikipedia_tag"]',
            },
        ],
        sections=[v2_section],
    )

    stats = compute_v2_card_stats(v2, v1_processed=v1)
    assert stats.new_polygons_vs_v1 == 3
    assert stats.new_polygons_wikipedia_tag_vs_v1 == 2
    assert stats.new_polygons_wikidata_only_vs_v1 == 1
    assert stats.new_wikipedia_tag_polygons_without_document == 0
    assert stats.new_wikipedia_tag_document_polygons_vs_v1 == 1
    assert stats.new_wikipedia_document_identity_words_vs_v1 == 10
    assert stats.new_wikipedia_documents_sharing_v1_content == 1
    assert stats.additional_unique_sections_vs_v1 == 1

    card_text = render_v2_card(v2, v1_processed=v1, stats=stats)
    assert "**Additional polygon identities:** 3" in card_text
    assert "**Of those, polygons with a Wikipedia tag:** 2" in card_text
    assert "**Of those, Wikidata-only polygons:** 1" in card_text
    assert "**New Wikipedia-tag polygons without a matching page at the snapshot:** 0" in card_text
    assert (
        "**V2-added polygons with a new Wikipedia-tag document and no Wikidata discovery:** 1"
        in card_text
    )
    assert "assets/v2_added_wikipedia_tag_documents.png" in card_text
    assert "**Words in newly added Wikipedia document identities:** 10" in card_text
    assert "**New Wikipedia document identities sharing content with V1:** 1" in card_text
    assert "**Additional unique section identities:** 1" in card_text
