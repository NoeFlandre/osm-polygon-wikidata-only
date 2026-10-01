"""Branch coverage for functions sitting near the CRAP gate (issue #116)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.enrichment.article_linker import (
    LinkSummary,
    _populate_one_summary,
)
from osm_polygon_wikidata_only.enrichment.wikipedia_client import FetchResult
from osm_polygon_wikidata_only.hf import v1_language_splits
from osm_polygon_wikidata_only.hf._polygon_geometry import validation
from osm_polygon_wikidata_only.pipeline import pending_publications
from osm_polygon_wikidata_only.pipeline._link_migration import artifacts as link_artifacts
from osm_polygon_wikidata_only.v2 import language_split_manifest, language_splits, reuse_load


@pytest.mark.parametrize("stem", ["", ".", "..", "a/b", "a\\b"])
def test_validate_stem_rejects_unsafe_values(stem: str) -> None:
    with pytest.raises(ValueError, match="Invalid pending publication stem"):
        pending_publications._validate_stem(stem)


def test_validate_stem_accepts_plain_value() -> None:
    assert pending_publications._validate_stem("europe-france") == "europe-france"


@pytest.mark.parametrize(
    ("stems", "error"),
    [("x", TypeError), (["a", "a"], ValueError), ([1], TypeError), (["a/b"], ValueError)],
)
def test_validate_metadata_marker_stems_rejects_bad_values(
    stems: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        pending_publications._validate_metadata_marker_stems(stems)


def test_validate_metadata_marker_stems_accepts_unique_strings() -> None:
    pending_publications._validate_metadata_marker_stems(["a", "b"])


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([("id", pa.int64())])), path)
    return path


def test_stage_voyage_replacements_branches(tmp_path: Path) -> None:
    documents = _write(tmp_path / "docs.parquet", [{"id": 1}, {"id": 2}])
    sections = _write(tmp_path / "sections.parquet", [{"id": 1}])
    staged = tmp_path / "staged"
    staged.mkdir()
    inputs = SimpleNamespace(voyage_documents_path=documents, voyage_sections_path=sections)

    def context(plan: object) -> Any:
        return cast(Any, SimpleNamespace(inputs=inputs, integrity_plan=plan))

    assert link_artifacts._stage_voyage_replacements(tmp_path, staged, context(None)) == []
    same = SimpleNamespace(retained_documents=[{"id": 1}, {"id": 2}], retained_sections=[{"id": 1}])
    assert link_artifacts._stage_voyage_replacements(tmp_path, staged, context(same)) == []
    changed = SimpleNamespace(retained_documents=[{"id": 1}], retained_sections=[])
    result = link_artifacts._stage_voyage_replacements(tmp_path, staged, context(changed))
    assert [target for target, _ in result] == [documents, sections]
    assert pq.read_table(result[0][1]).num_rows == 1
    assert pq.read_table(result[1][1]).num_rows == 0
    sections.unlink()
    result = link_artifacts._stage_voyage_replacements(tmp_path, staged, context(changed))
    assert [target for target, _ in result] == [documents]


def _shard(tmp_path: Path, row_count: int) -> Any:
    path = _write(tmp_path / "shard.parquet", [{"id": 1}])
    return SimpleNamespace(
        staged_path=path,
        final_path=tmp_path / "out" / "shard.parquet",
        row_count=row_count,
        language="en",
        source_files=["s.parquet"],
    )


def test_validated_output_file_branches(tmp_path: Path) -> None:
    schema = pa.schema([("id", pa.int64())])
    spec = cast(Any, SimpleNamespace(table="t", configuration="c"))
    result = language_split_manifest.validate_output_file(
        tmp_path, spec, _shard(tmp_path, 1), schema
    )
    assert result.row_count == 1
    with pytest.raises(language_splits.V2LanguageSplitError, match="row count mismatch"):
        language_split_manifest.validate_output_file(tmp_path, spec, _shard(tmp_path, 2), schema)
    other = pa.schema([("id", pa.string())])
    with pytest.raises(language_splits.V2LanguageSplitError, match="schema mismatch"):
        language_split_manifest.validate_output_file(tmp_path, spec, _shard(tmp_path, 1), other)
    broken = _shard(tmp_path, 1)
    broken.staged_path.write_text("not parquet", encoding="utf-8")
    with pytest.raises(language_splits.V2LanguageSplitError, match="Could not validate"):
        language_split_manifest.validate_output_file(tmp_path, spec, broken, schema)


def test_polygon_refs_branches() -> None:
    assert reuse_load.polygon_refs({"wikipedia_tag_refs": "{bad"}) == ()
    assert reuse_load.polygon_refs({"wikipedia_tag_refs": "{}"}) == ()
    refs = json.dumps([1, {"raw_key": "wikipedia", "raw_value": "en:Paris"}])
    assert len(reuse_load.polygon_refs({"wikipedia_tag_refs": refs})) == 1


def test_populate_one_summary_branches() -> None:
    empty = LinkSummary(qid="Q1", entity=None)
    _populate_one_summary(empty, {}, None, None)
    assert empty.statuses == {}
    entity = cast(Any, SimpleNamespace(sitelinks={"enwiki": "A", "frwiki": "B"}))
    article = cast(Any, object())
    fetched = cast(
        Any,
        {
            ("en", "enwiki"): {"A": FetchResult("ok", article)},
            ("fr", "frwiki"): {"B": FetchResult("ok", article)},
        },
    )
    summary = LinkSummary(qid="Q1", entity=entity)
    _populate_one_summary(summary, fetched, {"en", "fr"}, 1)
    assert len(summary.articles) == 1
    filtered = LinkSummary(qid="Q1", entity=entity)
    _populate_one_summary(filtered, fetched, {"en"}, None)
    assert list(filtered.statuses) == ["enwiki"]


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (None, "non-object entry"),
        ({"polygons_path": 1, "row_counts": {"polygons": 1}}, "lacks polygons_path"),
        ({"polygons_path": "polygons/b.parquet", "row_counts": {"polygons": 1}}, "points at"),
    ],
)
def test_v2_manifest_entry_rejects_bad_entries(entry: object, message: str) -> None:
    with pytest.raises(validation.PolygonStatsInputError, match=message):
        validation._v2_manifest_entry(Path("m.json"), "a", entry)


def test_v2_manifest_entry_accepts_canonical_entry() -> None:
    entry = {"polygons_path": "polygons/a.parquet", "row_counts": {"polygons": 3}}
    assert validation._v2_manifest_entry(Path("m.json"), "a", entry) == ("a", 3)


def test_read_previous_files_branches(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    assert v1_language_splits._read_previous_files(manifest) is None
    for text, expected in [
        ("{bad", None),
        ("[]", None),
        ('{"files": 1}', None),
        ('{"files": [1]}', [1]),
    ]:
        manifest.write_text(text, encoding="utf-8")
        assert v1_language_splits._read_previous_files(manifest) == expected
