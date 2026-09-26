"""Round-trip and partition properties for cached summaries and language splits."""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa
from hypothesis import given
from hypothesis import strategies as st

from osm_polygon_wikidata_only.hf._dataset_stats.models import PerFileSummary
from osm_polygon_wikidata_only.hf._dataset_stats.summary_codec import (
    summary_from_json,
    summary_to_json,
)
from osm_polygon_wikidata_only.hf.language_splits import (
    normalize_language,
    partition_row_indices,
)

_text = st.text(max_size=12)
_counts = st.integers(min_value=0, max_value=2**53)
_string_sets = st.frozensets(_text, max_size=5)
_int_maps = st.dictionaries(_text, _counts, max_size=5)

_summaries = st.builds(
    PerFileSummary,
    relative_path=_text,
    fingerprint=_text,
    file_size_bytes=_counts,
    kind=st.sampled_from(["documents", "sections", "facts", "other"]),
    scan_failed=st.booleans(),
    rows=_counts,
    non_empty=_counts,
    empty_or_null=_counts,
    total_chars=_counts,
    total_words=_counts,
    total_tokens_estimate=_counts,
    document_ids=_string_sets,
    section_ids=_string_sets,
    qids=_string_sets,
    languages=_int_maps,
    fact_rows=_counts,
    fact_ids=_string_sets,
    subject_qids=_string_sets,
    property_ids=_string_sets,
    property_labels=st.dictionaries(_text, _text, max_size=5),
    property_counts=_int_maps,
    with_property_en_label=_counts,
    with_value_en_label=_counts,
    with_qualifiers=_counts,
    with_references=_counts,
    unavailable_qualifiers=_counts,
    unavailable_references=_counts,
    value_type_counts=_int_maps,
)


@given(summary=_summaries)
def test_summary_codec_round_trips(summary: PerFileSummary) -> None:
    assert summary_from_json(summary_to_json(summary)) == summary


@given(summary=_summaries)
def test_summary_codec_round_trips_through_json_text(summary: PerFileSummary) -> None:
    blob = json.loads(json.dumps(summary_to_json(summary)))
    assert summary_from_json(blob) == summary
    assert summary_to_json(summary) == summary_to_json(summary)


_raw_languages = st.one_of(
    st.none(),
    st.sampled_from(["en", "EN", " fr ", "de", "zh_Hant", "zh-hant", "", "  ", "x", "en!"]),
    st.text(max_size=6),
)


def _partitions(values: list[Any]) -> dict[str, list[int]]:
    batch = pa.RecordBatch.from_arrays([pa.array(values, type=pa.string())], names=["language"])
    return {name: indices.to_pylist() for name, indices in partition_row_indices(batch, 0).items()}


@given(values=st.lists(_raw_languages, max_size=40))
def test_language_partitions_cover_rows_exactly_once(values: list[Any]) -> None:
    partitions = _partitions(values)
    flat = [index for indices in partitions.values() for index in indices]
    assert sorted(flat) == list(range(len(values)))
    for name, indices in partitions.items():
        assert indices == sorted(indices)
        assert all(normalize_language(values[i]).partition == name for i in indices)


@given(values=st.lists(_raw_languages, max_size=40), data=st.data())
def test_language_partitions_are_independent_of_row_order(values: list[Any], data: Any) -> None:
    order = data.draw(st.permutations(range(len(values))))
    shuffled = [values[i] for i in order]

    def by_value(rows: list[Any]) -> dict[str, list[Any]]:
        return {
            name: sorted((repr(rows[i]) for i in indices))
            for name, indices in _partitions(rows).items()
        }

    assert by_value(shuffled) == by_value(values)
