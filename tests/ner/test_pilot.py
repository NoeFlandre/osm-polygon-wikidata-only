"""Focused offline tests for the deterministic geographic NER pilot sampler."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_wikidata_only.ner.pipeline import INPUT_COLUMNS, LABEL, MODEL_ID, MODEL_REVISION
from scripts import prepare_geographic_ner_pilot as pilot
from scripts.prepare_geographic_ner_pilot import (
    TARGET_LANGUAGES,
    _add_link_row,
    _balance_projects,
    _batches,
    _candidate_is_linked,
    _candidate_row,
    _diverse_rows,
    _fill_language_shortfall,
    _language_quotas,
    _movable_language,
    _parser,
    _pilot_stem_selection_is_complete,
    _processed_root,
    _quotas_are_diverse,
    _reduce_single_project_targets,
    _retain_candidate,
    _row_key,
    _select_group_rows,
    _selection_manifest,
    build_contract,
    main,
    select_rows,
)
from tests.fixtures.case_sensitive_paths import case_sensitive_paths as case_sensitive_paths

WIKIVOYAGE_LANGUAGES = ("en", "fr", "de", "es", "zh", "ru", "ja", "pt")


def test_zero_quota_never_selects_rows_from_a_stratum():
    row = dict(sentence_id="s", document_id="d", project="wikipedia", language="en")
    assert pilot._diverse_rows([("score", row)], 0) == []


def test_region_selection_stops_once_coverage_and_minimum_diversity_are_satisfied():
    coverage = {f"region-{index:02}": {("wikipedia", "en")} for index in range(10)}
    assert pilot._greedy_pilot_stems(coverage) == tuple(
        f"region-{index:02}" for index in range(9, 1, -1)
    )


def test_quotas_accept_exact_total_capacity():
    candidates = {
        (project, language): [(str(index), {"document_id": str(index)}) for index in range(2)]
        for project in ("wikipedia", "wikivoyage")
        for language in TARGET_LANGUAGES
    }
    capacities = dict.fromkeys(candidates, 2)
    assert pilot._quotas(candidates, capacities, sample_size=40, seed="exact") == capacities


def test_language_targets_fill_shortfall_from_the_available_capacity():
    groups = {
        language: [(project, language) for project in ("wikipedia", "wikivoyage")]
        for language in TARGET_LANGUAGES
    }
    capacities = {
        group: 50 if language == "en" else 1
        for language, language_groups in groups.items()
        for group in language_groups
    }
    assert pilot._language_targets(groups, capacities, 60) == {
        language: 42 if language == "en" else 2 for language in TARGET_LANGUAGES
    }


def test_single_project_language_keeps_its_last_sentence_quota():
    targets = dict.fromkeys(TARGET_LANGUAGES, 1)
    groups = {language: [("wikipedia", language)] for language in TARGET_LANGUAGES}
    pilot._reduce_single_project_targets(targets, groups)
    assert targets == dict.fromkeys(TARGET_LANGUAGES, 1)


def test_candidate_scan_identifies_the_malformed_source_file(tmp_path):
    _write_fixture(tmp_path)
    path = tmp_path / "processed_v2/wikipedia/sentences/wikipedia-fixture.parquet"
    table = pq.read_table(path).drop(["text"])
    pq.write_table(table, path)
    with raises_exactly(f"Sentence file is missing required columns: {path}"):
        pilot.select_rows(tmp_path, sample_size=36)


def test_two_project_pilot_respects_zero_quotas(tmp_path):
    _write_fixture(tmp_path)
    rows, manifest = pilot.select_rows(tmp_path, sample_size=2)
    assert len(rows) == 2
    assert {row["project"] for row in rows} == {"wikipedia", "wikivoyage"}
    assert manifest["sample_size"] == 2


def test_odd_pilot_balances_projects_and_reproduces_seeded_selection(
    tmp_path, case_sensitive_paths
):
    _write_fixture(tmp_path)
    first = pilot.select_rows(tmp_path, sample_size=17, seed="odd-pilot")
    assert pilot.select_rows(tmp_path, sample_size=17, seed="odd-pilot") == first
    rows, manifest = first
    assert len(rows) == 17
    assert manifest["project_counts"] == {"wikipedia": 9, "wikivoyage": 8}
    assert pilot.select_rows(tmp_path, sample_size=17, seed="different")[0] != rows


def test_selection_reports_no_link_backed_rows(tmp_path):
    _write_fixture(tmp_path)
    for path in (tmp_path / "processed_v2/polygon_document_links").glob("*.parquet"):
        pq.write_table(pa.table({"document_id": ["unrelated"], "project": ["wikipedia"]}), path)
    with raises_exactly("No link-backed pilot rows are available"):
        pilot.select_rows(tmp_path)


def test_selection_rejects_an_incomplete_algorithm_result(tmp_path, monkeypatch):
    _write_fixture(tmp_path)
    monkeypatch.setattr(pilot, "_select_group_rows", lambda *args: [])
    with raises_exactly("Pilot selection did not produce the requested sample size"):
        pilot.select_rows(tmp_path)


def test_document_diversity_skips_repeated_documents_before_filling():
    rows = [
        dict(sentence_id=str(index), document_id=document, project="wikipedia", language="en")
        for index, document in enumerate(("a", "a", "b"))
    ]
    selected = pilot._diverse_rows([(str(i), row) for i, row in enumerate(rows)], 2)
    assert selected == [rows[0], rows[2]]
    with raises_exactly("Pilot group has too few retained rows"):
        pilot._fill_diverse_rows([], [], set(), 1)
    with raises_exactly("Pilot group has too few retained rows: ('wikipedia', 'en')"):
        pilot._select_group_rows({("wikipedia", "en"): []}, {("wikipedia", "en"): 1})


def test_equal_scores_have_a_stable_identity_tiebreaker():
    rows = [
        dict(sentence_id=value, document_id="d", project="wikipedia", language="en")
        for value in ("b", "a")
    ]
    retained = []
    for row in rows:
        pilot._retain_candidate(retained, "same-score", row, 2)
    assert [row["sentence_id"] for _, row in retained] == ["a", "b"]


@pytest.mark.parametrize("seed", ["first", "second", "third"])
def test_ordering_and_rebalancing_preserve_seeded_reproducibility(seed):
    rows = [
        dict(sentence_id=value, document_id="d", project="wikipedia", language="en")
        for value in ("a", "b", "c", "d")
    ]

    def order_key(row):
        identity = (row["sentence_id"], "d", "wikipedia", "en")
        return hashlib.sha256(f"{seed}-order\0{identity!r}".encode()).hexdigest(), identity

    assert pilot._ordered_rows(rows, seed) == sorted(rows, key=order_key)
    quotas = {
        (project, language): 2
        for project in ("wikipedia", "wikivoyage")
        for language in ("en", "fr", "de")
    }
    capacities = dict.fromkeys(quotas, 10)
    expected = min(
        ("en", "fr", "de"),
        key=lambda language: hashlib.sha256(f"{seed}-balance\0{language!r}".encode()).hexdigest(),
    )
    assert pilot._movable_language(quotas, capacities, "wikipedia", "wikivoyage", seed) == expected


@pytest.mark.parametrize("first,second", [(9, 1), (1, 9), (8, 1), (1, 8)])
def test_project_rebalancing_transfers_multiple_rows_without_losing_counts(
    first, second, monkeypatch
):
    quotas = {("wikipedia", "en"): first, ("wikivoyage", "en"): second}
    size = first + second
    expected = {("wikipedia", "en"): (size + 1) // 2, ("wikivoyage", "en"): size // 2}
    capacities = {group: max(quota, expected[group]) for group, quota in quotas.items()}
    original = pilot._movable_language
    calls = 0

    def bounded(*args):
        nonlocal calls
        calls += 1
        assert calls <= 10, "rebalancing must make progress on each transfer"
        return original(*args)

    monkeypatch.setattr(pilot, "_movable_language", bounded)
    pilot._balance_projects(quotas, capacities, size, "fixed")
    assert quotas == expected


def test_rebalance_updates_both_running_counts():
    quotas = {("wikipedia", "en"): 6, ("wikivoyage", "en"): 2}
    counts = Counter(wikipedia=6, wikivoyage=2)
    pilot._rebalance_project(
        quotas,
        dict.fromkeys(quotas, 8),
        counts,
        {"wikipedia": 4, "wikivoyage": 4},
        "wikipedia",
        "wikivoyage",
        "fixed",
    )
    assert counts == {"wikipedia": 4, "wikivoyage": 4}
    assert quotas == {("wikipedia", "en"): 4, ("wikivoyage", "en"): 4}


def test_zero_source_quota_cannot_be_moved():
    quotas = {("wikipedia", "en"): 0, ("wikivoyage", "en"): 1}
    assert not pilot._can_move_language(
        quotas, dict.fromkeys(quotas, 10), "wikipedia", "en", "wikipedia", "wikivoyage"
    )
    assert not pilot._can_move_language(
        quotas, dict.fromkeys(quotas, 10), "wikivoyage", "en", "wikipedia", "wikivoyage"
    )


def test_greedy_region_selection_is_bounded_even_when_coverage_is_incomplete():
    assert pilot._greedy_pilot_stems({}) == ()
    coverage = {f"region-{i:02d}": {("wikipedia", f"language-{i}")} for i in range(20)}
    selected = pilot._greedy_pilot_stems(coverage)
    assert len(selected) == 16
    assert len(set(selected)) == 16


def test_candidate_table_budget_is_cumulative_across_files(tmp_path):
    state = pilot._CandidateState()
    row = dict(
        sentence_id="s",
        document_id="d",
        project="wikipedia",
        language="en",
        text="Paris",
        segmentation_status="split",
    )
    for i in range(2):
        path = tmp_path / f"{i}.parquet"
        pq.write_table(pa.Table.from_pylist([{**row, "sentence_id": str(i)}]), path)
        pilot._read_candidate_table(
            tmp_path,
            "wikipedia",
            path,
            state,
            set(INPUT_COLUMNS),
            set(),
            {("d", "wikipedia", "en")},
            sample_size=10,
            seed="fixed",
            max_candidate_tables=2,
            max_candidate_batches=2,
        )
    assert state.tables_read == state.batches_read == 2
    assert state.capacities == {("wikipedia", "en"): 2}
    with raises_exactly("Pilot candidate scan cap reached before quotas were filled"):
        pilot._read_candidate_table(
            tmp_path,
            "wikipedia",
            tmp_path / "not-read",
            state,
            set(INPUT_COLUMNS),
            set(),
            set(),
            sample_size=10,
            seed="fixed",
            max_candidate_tables=2,
            max_candidate_batches=2,
        )


def test_processed_root_preserves_exact_path_spelling(tmp_path):
    processed = tmp_path / "processed_v2"
    processed.mkdir()
    assert str(pilot._processed_root(tmp_path)) == str(processed)
    assert str(pilot._processed_root(processed)) == str(processed)
    other = tmp_path / "other"
    other.mkdir()
    with raises_exactly(f"processed_v2 directory is missing: {other / 'processed_v2'}"):
        pilot._processed_root(other)


def test_link_file_selection_is_exact_sorted_and_requires_every_table(tmp_path):
    tables = [("wikipedia", tmp_path / "z.parquet"), ("wikivoyage", tmp_path / "a.parquet")]
    links = tmp_path / "polygon_document_links"
    links.mkdir()
    with raises_exactly(f"Selected sentence table has no link file: {links / 'a.parquet'}"):
        pilot._selected_link_paths(tmp_path, tables)
    for name in ("a.parquet", "z.parquet"):
        (links / name).touch()
    assert [str(path) for path in pilot._selected_link_paths(tmp_path, tables)] == [
        str(links / "a.parquet"),
        str(links / "z.parquet"),
    ]
    with raises_exactly("No polygon-document link files are available"):
        pilot._read_link_keys(tmp_path, [])


@pytest.mark.parametrize("languages", [True, False])
def test_link_columns_preserve_optional_language_without_inventing_columns(tmp_path, languages):
    data = {"document_id": ["d"], "project": ["wikipedia"], "irrelevant": [123]}
    if languages:
        data["language"] = ["fr"]
    path = tmp_path / "links.parquet"
    pq.write_table(pa.table(data), path)
    pairs, triples = set(), set()
    pilot._read_link_file(path, pairs, triples)
    assert pairs == {("d", "wikipedia")}
    assert triples == ({("d", "wikipedia", "fr")} if languages else set())
    assert pilot._available_columns(path, ["document_id", "missing"]) == {"document_id"}


def test_missing_link_or_sentence_columns_have_exact_diagnostics(tmp_path):
    path = tmp_path / "incomplete.parquet"
    pq.write_table(pa.table({"unrelated": [1]}), path)
    with raises_exactly(f"Link file is missing document_id/project columns: {path}"):
        pilot._read_link_file(path, set(), set())
    with raises_exactly(f"Sentence file is missing required columns: {path}"):
        pilot._require_candidate_columns(path, {"sentence_id"})


@pytest.mark.parametrize("region", [None, [], "region"])
def test_manifest_region_requires_an_object(region):
    with raises_exactly("Sentence manifest region is invalid"):
        pilot._manifest_region_identity(region)


@pytest.mark.parametrize(
    "region",
    [
        {"project": "unknown", "stem": "a"},
        {"project": 1, "stem": "a"},
        {"project": "wikipedia", "stem": None},
        {},
    ],
)
def test_manifest_region_requires_a_known_project_and_string_stem(region):
    with raises_exactly("Sentence manifest region has invalid project or stem"):
        pilot._manifest_region_identity(region)


def test_manifest_languages_filter_unsupported_and_non_string_values():
    assert pilot._region_target_languages({"supported_languages": ["en", "xx", 7]}) == {"en"}
    assert pilot._region_target_languages({}) == frozenset(TARGET_LANGUAGES)
    assert pilot._region_target_languages(None) == frozenset(TARGET_LANGUAGES)


def test_manifest_reader_reports_missing_regions_and_missing_selected_file(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with raises_exactly(f"Sentence manifest has no regions: {manifest}"):
        list(pilot._manifest_sentence_files(tmp_path, manifest))
    region = {"project": "wikipedia", "stem": "region"}
    path = tmp_path / "wikipedia/sentences/region.parquet"
    with raises_exactly(f"Sentence manifest file is missing: {path}"):
        pilot._manifest_region_path(tmp_path, region)


def test_manifest_skips_unsupported_regions_and_keeps_later_supported_regions(tmp_path):
    path = tmp_path / "wikipedia/sentences/yes.parquet"
    path.parent.mkdir(parents=True)
    path.touch()
    result = pilot._manifest_regions_by_stem(
        tmp_path,
        [
            {"project": "wikipedia", "stem": "no", "supported_languages": ["xx"]},
            {"project": "wikipedia", "stem": "yes", "supported_languages": ["fr"]},
        ],
    )
    assert result == {"yes": {"wikipedia": ("wikipedia", path, frozenset({"fr"}))}}


def test_diversity_counts_the_requested_prefix_and_accepts_exact_capacity():
    group = ("wikipedia", "en")
    candidates = {group: [("a", {"document_id": "a"}), ("b", {"document_id": "b"})]}
    assert pilot._quotas_are_diverse(candidates, {group: 2})
    assert not pilot._quotas_are_diverse(candidates, {group: 3})
    assert not pilot._quotas_are_diverse({}, {group: 1})


def test_split_coverage_requires_both_projects_even_at_zero_rows():
    with raises_exactly("No completed split rows are available for the pilot"):
        pilot._validate_split_coverage(0, {"wikipedia", "wikivoyage"})
    pilot._validate_split_coverage(1, {"wikipedia", "wikivoyage"})


def test_scan_cap_uses_inclusive_batch_and_table_limits():
    for tables, batches in ((2, 0), (0, 2)):
        state = pilot._CandidateState(tables_read=tables, batches_read=batches)
        with raises_exactly("Pilot candidate scan cap reached before quotas were filled"):
            pilot._raise_candidate_scan_cap(state, 2, 2)


def test_candidate_accounting_deduplicates_rows_but_counts_all_split_inputs():
    state = pilot._CandidateState()
    row = dict(
        sentence_id="s",
        document_id="d",
        project="wikipedia",
        language="en",
        text="Paris",
        segmentation_status="split",
    )
    for _ in range(3):
        pilot._read_candidate_row(
            row, "wikipedia", state, set(), {("d", "wikipedia", "en")}, sample_size=10, seed="fixed"
        )
    assert state.split_rows == 3
    assert state.split_projects == {"wikipedia"}
    assert state.capacities == {("wikipedia", "en"): 1}
    assert state.seen == {("s", "d", "wikipedia", "en")}
    assert len(state.candidates[("wikipedia", "en")]) == 1


def test_section_position_keeps_zero_and_explicit_indexes():
    assert pilot._section_position({"section_index": 0, "sentence_id": "s"}) == 0
    assert pilot._section_position({"section_index": 5, "sentence_id": "s"}) == 5


def test_language_quotas_fail_with_useful_capacity_diagnostics():
    with raises_exactly("Not enough link-backed rows for the requested pilot sample"):
        pilot._quotas({}, {}, sample_size=1, seed="fixed")
    with raises_exactly("Pilot source is missing one or more target languages"):
        pilot._language_targets({}, {}, 10)
    with raises_exactly("Pilot language quotas cannot be redistributed"):
        pilot._redistribute_language_carry(dict.fromkeys(TARGET_LANGUAGES, 0), 9)
    with raises_exactly("Pilot language quotas cannot fit the available project strata"):
        pilot._stratified_quotas({"en": 2}, {"en": [("wikipedia", "en")]}, {("wikipedia", "en"): 1})


def _write_fixture(root: Path) -> None:
    processed = root / "processed_v2"
    manifest_regions: list[dict[str, object]] = []
    for project, languages in (
        ("wikipedia", TARGET_LANGUAGES),
        ("wikivoyage", WIKIVOYAGE_LANGUAGES),
    ):
        stem = f"{project}-fixture"
        sentences: list[dict[str, object]] = []
        links: list[dict[str, object]] = []
        for language in languages:
            for document_index in range(6):
                document_id = f"Q{project[0]}{language}{document_index}:{project}:{language}:{document_index}:1"
                links.append(
                    {
                        "polygon_id": f"{stem}:way:{language}-{document_index}",
                        "document_id": document_id,
                        "project": project,
                        "language": language,
                    }
                )
                for section_index in (0, 1):
                    sentence_id = f"{project}-{language}-{document_index}-{section_index}"
                    sentences.append(
                        {
                            "sentence_id": sentence_id,
                            "document_id": document_id,
                            "project": project,
                            "language": language,
                            "text": f"{language} place {document_index} section {section_index}",
                            "segmentation_status": "split",
                            "section_index": section_index,
                            "sentence_index": document_index,
                        }
                    )
        sentence_dir = processed / project / "sentences"
        sentence_dir.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(sentences), sentence_dir / f"{stem}.parquet")
        link_dir = processed / "polygon_document_links"
        link_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(links), link_dir / f"{stem}.parquet")
        manifest_regions.append(
            {
                "project": project,
                "stem": stem,
                "supported_languages": list(languages),
            }
        )
    manifest_path = processed / "manifests" / "sentence_splitting.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"regions": manifest_regions}), encoding="utf-8")


def test_contract_matches_the_pinned_pipeline_interface() -> None:
    contract = build_contract()

    assert asdict(contract) == {
        "languages": TARGET_LANGUAGES,
        "threshold": 0.5,
        "validation_status": "pilot_unvalidated",
        "implementation_revision": "geographic-ner-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "label": LABEL,
    }


def test_selection_is_deterministic_balanced_and_link_backed(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    rows_one, manifest_one = select_rows(tmp_path, sample_size=36, seed="test-seed")
    rows_two, manifest_two = select_rows(tmp_path, sample_size=36, seed="test-seed")

    assert rows_one == rows_two
    assert manifest_one == manifest_two
    assert len(rows_one) == 36
    assert tuple(rows_one[0]) == INPUT_COLUMNS
    assert {row["project"] for row in rows_one} == {"wikipedia", "wikivoyage"}
    assert {row["language"] for row in rows_one} == set(TARGET_LANGUAGES)
    assert all(row["segmentation_status"] == "split" for row in rows_one)
    assert manifest_one["language_counts"] == {
        "ar": 2,
        "de": 4,
        "en": 5,
        "es": 4,
        "fr": 5,
        "hy": 2,
        "ja": 3,
        "pt": 3,
        "ru": 4,
        "zh": 4,
    }
    assert manifest_one["project_counts"] == {"wikipedia": 18, "wikivoyage": 18}
    assert manifest_one["unique_documents"] >= 10
    assert manifest_one["unique_section_positions"] >= 2
    assert manifest_one["link_relationship"]["all_selected_rows_link_backed"] is True


def test_selection_rejects_a_fixture_without_completed_split_rows(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    sentence_path = next((tmp_path / "processed_v2").glob("wikipedia/sentences/*.parquet"))
    table = pq.read_table(sentence_path)
    values = table.to_pydict()
    values["segmentation_status"] = ["unsupported_language"] * table.num_rows
    pq.write_table(pa.table(values), sentence_path)

    with raises_exactly("No completed split rows are available for the pilot"):
        select_rows(tmp_path, sample_size=36, seed="test-seed")


def test_selection_stops_when_bounded_candidate_scan_is_sufficient(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    rows, manifest = select_rows(
        tmp_path,
        sample_size=36,
        seed="test-seed",
        max_candidate_tables=2,
        max_candidate_batches=2,
    )

    assert len(rows) == 36
    assert manifest["source_files"] == [
        "wikipedia/sentences/wikipedia-fixture.parquet",
        "wikivoyage/sentences/wikivoyage-fixture.parquet",
    ]


def test_selection_reads_links_only_for_manifest_selected_tables(
    tmp_path: Path, case_sensitive_paths
) -> None:
    _write_fixture(tmp_path)
    pq.write_table(
        pa.table({"unrelated": ["must not be read"]}),
        tmp_path / "processed_v2/polygon_document_links/unselected.parquet",
    )
    sentence_dir = tmp_path / "processed_v2/wikipedia/sentences"
    pq.write_table(
        pa.table({"unrelated": ["must not be read"]}), sentence_dir / "unselected.parquet"
    )

    rows, _ = select_rows(tmp_path, sample_size=36, seed="test-seed")

    assert len(rows) == 36


def test_selection_rejects_a_bounded_scan_shortfall(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    with raises_exactly("Pilot candidate scan cap reached before quotas were filled"):
        select_rows(
            tmp_path,
            sample_size=36,
            seed="test-seed",
            max_candidate_tables=1,
            max_candidate_batches=1,
        )


def test_selection_accepts_processed_v2_root_and_manifest_fallback(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "processed_v2/manifests/sentence_splitting.json").unlink()

    rows, manifest = select_rows(tmp_path / "processed_v2", sample_size=36, seed="test-seed")

    assert len(rows) == 36
    assert manifest["source_files"] == [
        "wikipedia/sentences/wikipedia-fixture.parquet",
        "wikivoyage/sentences/wikivoyage-fixture.parquet",
    ]
    assert _processed_root(tmp_path / "processed_v2") == tmp_path / "processed_v2"


def test_language_shortfall_redistributes_to_available_language() -> None:
    groups = {language: [("wikipedia", language)] for language in TARGET_LANGUAGES}
    capacities = {
        group: 1 for groups_for_language in groups.values() for group in groups_for_language
    }
    capacities[("wikipedia", "en")] = 2
    targets = {language: 1 for language in TARGET_LANGUAGES}

    _fill_language_shortfall(targets, groups, capacities, sample_size=11)

    assert sum(targets.values()) == 11
    assert sum(value == 2 for value in targets.values()) == 1


def test_language_shortfall_requires_bounded_strict_progress() -> None:
    class ProgressCheckedTargets(dict[str, int]):
        sum_reads = 0
        updates = 0

        def values(self):
            self.sum_reads += 1
            if self.sum_reads > 1:
                raise AssertionError("shortfall loop re-read its own progress instead of bounding")
            return super().values()

        def __setitem__(self, language: str, value: int) -> None:
            previous = self.get(language)
            if previous is not None and value != previous + 1:
                raise AssertionError("shortfall loop made a non-progress quota update")
            self.updates += 1
            super().__setitem__(language, value)

    groups = {language: [("wikipedia", language)] for language in TARGET_LANGUAGES}
    capacities = {
        group: 1 for groups_for_language in groups.values() for group in groups_for_language
    }
    capacities[("wikipedia", "en")] = 2
    targets = ProgressCheckedTargets({language: 1 for language in TARGET_LANGUAGES})

    _fill_language_shortfall(targets, groups, capacities, sample_size=11)

    assert targets["en"] == 2
    assert targets.sum_reads == 1
    assert targets.updates == 1
    assert sum(super(ProgressCheckedTargets, targets).values()) == 11


def test_language_shortfall_rejects_exhausted_capacity() -> None:
    groups = {language: [("wikipedia", language)] for language in TARGET_LANGUAGES}
    capacities = {
        group: 1 for groups_for_language in groups.values() for group in groups_for_language
    }
    targets = {language: 1 for language in TARGET_LANGUAGES}

    with raises_exactly("Not enough link-backed rows for the requested pilot sample"):
        _fill_language_shortfall(targets, groups, capacities, sample_size=11)


def test_project_rebalancing_finds_and_rejects_movable_language() -> None:
    quotas = {("wikipedia", "en"): 1, ("wikivoyage", "en"): 0}
    capacities = {("wikipedia", "en"): 1, ("wikivoyage", "en"): 1}

    assert _movable_language(quotas, capacities, "wikipedia", "wikivoyage", "test-seed") == "en"

    with raises_exactly("Pilot sample cannot be balanced across projects"):
        _movable_language(
            quotas,
            {("wikipedia", "en"): 1, ("wikivoyage", "en"): 0},
            "wikipedia",
            "wikivoyage",
            "test-seed",
        )


def test_selection_prefers_diverse_documents_then_fills_same_document_rows() -> None:
    rows = [
        (
            "score-1",
            {"sentence_id": "one", "document_id": "doc", "project": "wikipedia", "language": "en"},
        ),
        (
            "score-2",
            {"sentence_id": "two", "document_id": "doc", "project": "wikipedia", "language": "en"},
        ),
    ]

    selected = _diverse_rows(rows, quota=2)

    assert [row["sentence_id"] for row in selected] == ["one", "two"]
    assert not _quotas_are_diverse({("wikipedia", "en"): rows}, {("wikipedia", "en"): 2})


def test_cli_writes_exact_input_columns_and_pinned_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_fixture(tmp_path)
    output_dir = tmp_path / "nested" / "pilot"

    assert (
        main(
            [
                "--data-root",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "--sample-size",
                "36",
                "--seed",
                "test-seed",
                "--max-candidate-tables",
                "2",
                "--max-candidate-batches",
                "2",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--data-root",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "--sample-size",
                "36",
                "--seed",
                "test-seed",
                "--max-candidate-tables",
                "2",
                "--max-candidate-batches",
                "2",
            ]
        )
        == 0
    )

    table = pq.read_table(output_dir / "input.parquet")
    assert tuple(table.schema.names) == INPUT_COLUMNS
    assert table.num_rows == 36
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "contract.json",
        "input.parquet",
        "selection.json",
    ]
    expected_contract = asdict(build_contract())
    expected_contract["languages"] = list(TARGET_LANGUAGES)
    assert json.loads((output_dir / "contract.json").read_text()) == expected_contract
    selection = json.loads((output_dir / "selection.json").read_text())
    assert set(selection) == {
        "seed",
        "sample_size",
        "target_languages",
        "source_files",
        "project_counts",
        "language_counts",
        "unique_documents",
        "unique_section_positions",
        "link_relationship",
        "contract_id",
    }
    assert selection["seed"] == "test-seed"
    assert selection["sample_size"] == 36
    assert selection["target_languages"] == list(TARGET_LANGUAGES)
    assert selection["source_files"] == [
        "wikipedia/sentences/wikipedia-fixture.parquet",
        "wikivoyage/sentences/wikivoyage-fixture.parquet",
    ]
    assert selection["link_relationship"] == {
        "all_selected_rows_link_backed": True,
        "selected_rows": 36,
    }
    assert selection["contract_id"] == build_contract().identity
    captured = capsys.readouterr()
    expected_stdout = json.dumps({"output_dir": str(output_dir), "rows": 36}, sort_keys=True)
    assert captured.out == f"{expected_stdout}\n{expected_stdout}\n"


@pytest.mark.parametrize(
    "option",
    ["--max-candidate-tables", "--max-candidate-batches"],
)
def test_cli_forwards_each_scan_cap(tmp_path: Path, option: str) -> None:
    _write_fixture(tmp_path)
    output_dir = tmp_path / "pilot"
    table_cap, batch_cap = ("1", "2") if option == "--max-candidate-tables" else ("2", "1")

    with raises_exactly("Pilot candidate scan cap reached before quotas were filled"):
        main(
            [
                "--data-root",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "--sample-size",
                "36",
                "--max-candidate-tables",
                table_cap,
                "--max-candidate-batches",
                batch_cap,
            ]
        )


def test_selection_manifest_falls_back_to_sentence_id_for_null_section_index() -> None:
    selected = [
        {
            "sentence_id": "sentence-a",
            "document_id": "Q1",
            "project": "wikipedia",
            "language": "en",
            "section_index": None,
        },
        {
            "sentence_id": "sentence-b",
            "document_id": "Q1",
            "project": "wikipedia",
            "language": "en",
            "section_index": None,
        },
    ]
    rows: list[dict[str, str]] = [
        {
            "sentence_id": "sentence-a",
            "document_id": "Q1",
            "project": "wikipedia",
            "language": "en",
        },
        {
            "sentence_id": "sentence-b",
            "document_id": "Q1",
            "project": "wikipedia",
            "language": "en",
        },
    ]

    manifest = _selection_manifest(
        selected,
        rows,
        source_files=("wikipedia/sentences/sample.parquet",),
        sample_size=2,
        seed="test-seed",
    )

    assert manifest["unique_section_positions"] == 2


def test_selection_applies_table_cap_before_link_scan(tmp_path: Path, monkeypatch) -> None:
    processed = tmp_path / "processed_v2"
    processed.mkdir()
    sentence_files = [
        ("wikipedia", processed / "wikipedia-1.parquet"),
        ("wikivoyage", processed / "wikivoyage-1.parquet"),
        ("wikipedia", processed / "wikipedia-2.parquet"),
    ]
    observed: dict[str, object] = {}

    monkeypatch.setattr(pilot, "_sentence_files", lambda root: iter(sentence_files))

    def observe_link_scan(root: Path, files: tuple[tuple[str, Path], ...]):
        observed["files"] = files
        raise RuntimeError("link scan reached")

    monkeypatch.setattr(pilot, "_read_link_keys", observe_link_scan)

    with raises_exactly("link scan reached", RuntimeError):
        pilot.select_rows(
            tmp_path,
            sample_size=1,
            max_candidate_tables=2,
            max_candidate_batches=1,
        )

    assert observed["files"] == tuple(sentence_files[:2])


def test_selection_supports_sentence_files_without_optional_position_columns(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    for sentence_path in sorted((tmp_path / "processed_v2").glob("*/sentences/*.parquet")):
        table = pq.read_table(sentence_path, columns=list(INPUT_COLUMNS))
        pq.write_table(table, sentence_path)

    rows, manifest = select_rows(tmp_path, sample_size=36, seed="test-seed")

    assert len(rows) == 36
    assert tuple(rows[0]) == INPUT_COLUMNS
    assert manifest["unique_section_positions"] == len(rows)


def test_batch_reader_preserves_requested_columns_and_batch_size(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeParquetFile:
        def __init__(self, path: Path) -> None:
            observed["path"] = path

        def iter_batches(self, **kwargs):
            observed.update(kwargs)
            return iter(())

    monkeypatch.setattr(pilot.pq, "ParquetFile", FakeParquetFile)

    assert list(_batches(Path("sample.parquet"), ("document_id", "text"))) == []
    assert observed == {
        "path": Path("sample.parquet"),
        "batch_size": 8192,
        "columns": ["document_id", "text"],
    }


def test_link_backing_requires_valid_ids_and_exact_membership() -> None:
    pairs: set[tuple[str, str]] = set()
    triples: set[tuple[str, str, str]] = set()
    _add_link_row(
        {"document_id": "doc", "project": "wikipedia", "language": "en"},
        pairs,
        triples,
    )
    _add_link_row(
        {"document_id": None, "project": "wikipedia", "language": "en"},
        pairs,
        triples,
    )

    assert pairs == {("doc", "wikipedia")}
    assert triples == {("doc", "wikipedia", "en")}

    row = {
        "sentence_id": "sentence",
        "document_id": "doc",
        "project": "wikipedia",
        "language": "en",
        "text": "text",
        "segmentation_status": "split",
    }
    assert _candidate_row(row, "wikipedia", pairs, set()) is True
    assert _candidate_row(row, "wikipedia", set(), triples) is True
    assert _candidate_row({**row, "project": "wikivoyage"}, "wikipedia", pairs, triples) is False
    assert _candidate_row({**row, "language": "xx"}, "wikipedia", pairs, triples) is False
    assert _candidate_row({**row, "text": None}, "wikipedia", pairs, triples) is False
    assert _candidate_is_linked("doc", "wikipedia", "en", pairs, set()) is True
    assert _candidate_is_linked("doc", "wikipedia", "en", set(), triples) is True
    assert _candidate_is_linked("other", "wikipedia", "en", set(), set()) is False


def test_candidate_retention_and_group_selection_preserve_identity_and_quota() -> None:
    def row(sentence_id: str, document_id: str) -> dict[str, str]:
        return {
            "sentence_id": sentence_id,
            "document_id": document_id,
            "project": "wikipedia",
            "language": "en",
        }

    retained: list[tuple[str, dict[str, str]]] = []
    _retain_candidate(retained, "z", row("z", "doc-z"), limit=2)
    _retain_candidate(retained, "a", row("a", "doc-a"), limit=2)
    _retain_candidate(retained, "b", row("b", "doc-b"), limit=2)

    assert [item[1]["sentence_id"] for item in retained] == ["a", "b"]
    assert _row_key(row("a", "doc-a")) != _row_key(row("a", "doc-b"))

    candidates = {
        ("wikipedia", "en"): [
            ("a", row("a", "doc-a")),
            ("b", row("b", "doc-b")),
        ]
    }
    selected = _select_group_rows(candidates, {("wikipedia", "en"): 2})

    assert [item["sentence_id"] for item in selected] == ["a", "b"]


def test_quota_boundaries_keep_capacity_and_project_balance() -> None:
    three_groups = (
        ("wikipedia", "en"),
        ("wikivoyage", "en"),
        ("other", "en"),
    )
    assert _language_quotas(
        1,
        three_groups,
        {group: 1 for group in three_groups},
    ) == {
        ("wikipedia", "en"): 0,
        ("wikivoyage", "en"): 0,
        ("other", "en"): 1,
    }
    assert _language_quotas(
        3,
        three_groups[:2],
        {("wikipedia", "en"): 2, ("wikivoyage", "en"): 1},
    ) == {("wikipedia", "en"): 1, ("wikivoyage", "en"): 1}

    quotas = {("wikipedia", "en"): 2, ("wikivoyage", "en"): 0}
    _balance_projects(
        quotas,
        {("wikipedia", "en"): 2, ("wikivoyage", "en"): 2},
        sample_size=2,
        seed="test-seed",
    )

    assert quotas == {("wikipedia", "en"): 1, ("wikivoyage", "en"): 1}


def test_language_and_stem_boundaries_preserve_diversity_rules() -> None:
    targets = {language: 1 for language in TARGET_LANGUAGES}
    targets["ar"] = 2
    groups_by_language = {
        language: [("wikipedia", language), ("wikivoyage", language)]
        for language in TARGET_LANGUAGES
    }
    groups_by_language["ar"] = [("wikipedia", "ar")]
    _reduce_single_project_targets(
        targets,
        groups_by_language,
    )
    assert targets["ar"] == 1
    assert targets["en"] == 2
    assert all(
        targets[language] == 1 for language in TARGET_LANGUAGES if language not in {"ar", "en"}
    )

    assert _pilot_stem_selection_is_complete(set(), selected=8, total=10) is True
    assert _pilot_stem_selection_is_complete({("wikipedia", "en")}, selected=8, total=10) is False
    assert _pilot_stem_selection_is_complete(set(), selected=7, total=10) is False
    assert _pilot_stem_selection_is_complete(set(), selected=8, total=8) is True


def test_cli_parser_preserves_required_paths_types_defaults_and_help() -> None:
    parser = _parser()
    args = parser.parse_args(["--data-root", "data", "--output-dir", "output"])

    assert args.data_root == Path("data")
    assert args.output_dir == Path("output")
    assert args.sample_size == 100
    assert args.seed == "geographic-ner-pilot-v1"
    assert args.max_candidate_tables == 256
    assert args.max_candidate_batches == 512
    assert (
        "Prepare a deterministic, link-backed geographic NER pilot shard." in parser.format_help()
    )

    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "output"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--data-root", "data"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_size": 0}, "Pilot sample_size must be a positive integer"),
        ({"seed": ""}, "Pilot seed must be a non-empty string"),
        ({"seed": 1}, "Pilot seed must be a non-empty string"),
        ({"max_candidate_tables": 0}, "Pilot max_candidate_tables must be a positive integer"),
        ({"max_candidate_batches": 0}, "Pilot max_candidate_batches must be a positive integer"),
    ],
)
def test_invalid_selection_inputs_keep_exact_diagnostics(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=rf"\A{message}\Z"):
        cast(Any, select_rows)(tmp_path, **kwargs)


def raises_exactly(message: str, kind: type[BaseException] = ValueError):
    """Pin an operator-facing diagnostic so message-only mutations cannot survive."""
    return pytest.raises(kind, match=rf"^{re.escape(message)}$")
