"""Tests for the function-level CRAP quality report."""

from __future__ import annotations

from pathlib import Path

import pytest
from radon.complexity import cc_visit

from scripts.quality.crap_score import (
    CrapEntry,
    crap_score,
    entries_from_reports,
    evaluate_threshold,
)


def test_crap_score_uses_standard_formula() -> None:
    assert crap_score(10, 0.75) == 11.5625


def test_crap_score_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="complexity"):
        crap_score(0, 0.5)
    with pytest.raises(ValueError, match="coverage"):
        crap_score(5, 1.1)


def test_evaluate_threshold_returns_worst_functions() -> None:
    entries = [
        CrapEntry("module.py", "safe", 2, 1.0),
        CrapEntry("module.py", "risky", 10, 0.0),
    ]

    assert [entry.name for entry in evaluate_threshold(entries, maximum=30.0)] == ["risky"]


def test_evaluate_threshold_rejects_exact_limit() -> None:
    entry = CrapEntry("module.py", "boundary", 6, 1.0)

    assert evaluate_threshold([entry], maximum=6.0) == [entry]


def test_evaluate_threshold_accepts_scores_below_limit() -> None:
    entry = CrapEntry("module.py", "safe", 5, 1.0)

    assert evaluate_threshold([entry], maximum=6.0) == []


def test_entries_from_reports_joins_radon_with_coverage_summary() -> None:
    entries = entries_from_reports(
        {
            "files": {
                "module.py": {
                    "functions": {"parse": {"summary": {"percent_statements_covered": 75.0}}}
                }
            }
        },
        {"module.py": [{"type": "function", "name": "parse", "complexity": 10, "lineno": 4}]},
    )

    assert entries == [CrapEntry("module.py", "parse", 10, 0.75, 4)]


def test_entries_from_reports_scores_uncovered_functions_as_zero() -> None:
    entries = entries_from_reports(
        {"files": {"module.py": {"functions": {}}}},
        {"module.py": [{"type": "function", "name": "uncovered", "complexity": 5, "lineno": 9}]},
    )

    assert entries == [CrapEntry("module.py", "uncovered", 5, 0.0, 9)]


def test_entries_from_reports_falls_back_to_line_coverage_for_helpers() -> None:
    """Helpers absent from coverage's function map still use executed lines."""
    entries = entries_from_reports(
        {
            "files": {
                "module.py": {
                    "executed_lines": [10, 11],
                    "missing_lines": [12],
                    "functions": {},
                }
            }
        },
        {
            "module.py": [
                {
                    "type": "function",
                    "name": "helper",
                    "complexity": 3,
                    "lineno": 10,
                    "endline": 12,
                }
            ]
        },
    )

    assert entries == [CrapEntry("module.py", "helper", 3, 2 / 3, 10)]


def test_retry_entrypoint_stays_below_complexity_six() -> None:
    source_path = Path("src/osm_polygon_wikidata_only/utils/retry.py")
    functions = {
        block.name: block.complexity
        for block in cc_visit(source_path.read_text(encoding="utf-8"))
        if block.name == "with_retries"
    }

    assert functions["with_retries"] < 6


def test_sync_execute_stays_below_complexity_six() -> None:
    source_path = Path("src/osm_polygon_wikidata_only/cli/run_sync.py")
    functions = {
        block.name: block.complexity
        for block in cc_visit(source_path.read_text(encoding="utf-8"))
        if block.name == "execute"
    }

    assert functions["execute"] < 6


def test_continent_assignment_stays_below_complexity_six() -> None:
    source_path = Path("src/osm_polygon_wikidata_only/hf/continent_stats.py")
    functions = {
        block.name: block.complexity
        for block in cc_visit(source_path.read_text(encoding="utf-8"))
        if block.name == "assign_continents"
    }

    assert functions["assign_continents"] < 6


def test_polygon_link_row_coercion_stays_below_complexity_six() -> None:
    source_path = Path("src/osm_polygon_wikidata_only/domain/polygon_document_links.py")
    functions = {
        block.name: block.complexity
        for block in cc_visit(source_path.read_text(encoding="utf-8"))
        if block.name == "_coerce_row"
    }

    assert functions["_coerce_row"] < 6


def test_polygon_link_validation_stays_below_complexity_six() -> None:
    source_path = Path("src/osm_polygon_wikidata_only/domain/polygon_document_links.py")
    functions = {
        block.name: block.complexity
        for block in cc_visit(source_path.read_text(encoding="utf-8"))
        if block.name == "validate_polygon_document_links"
    }

    assert functions["validate_polygon_document_links"] < 6


def test_json_file_cache_get_stays_below_complexity_six() -> None:
    source_path = Path("src/osm_polygon_wikidata_only/io/cache.py")
    functions = {
        block.name: block.complexity
        for block in cc_visit(source_path.read_text(encoding="utf-8"))
        if block.name == "get"
    }

    assert functions["get"] < 6


def test_retirement_reference_validation_stays_below_complexity_six() -> None:
    source_path = Path("src/osm_polygon_wikidata_only/augmentation/wikipedia_retirement.py")
    functions = {
        block.name: block.complexity
        for block in cc_visit(source_path.read_text(encoding="utf-8"))
        if block.name == "_assert_references_resolve"
    }

    assert functions["_assert_references_resolve"] < 6


def test_section_parser_stays_below_complexity_six() -> None:
    source_path = Path("src/osm_polygon_wikidata_only/augmentation/sections.py")
    functions = {
        block.name: block.complexity
        for block in cc_visit(source_path.read_text(encoding="utf-8"))
        if block.name == "_build_sections"
    }

    assert functions["_build_sections"] < 6


def test_v2_land_path_resolution_stays_below_complexity_six() -> None:
    source_path = Path("src/osm_polygon_wikidata_only/v2/maps.py")
    functions = {
        block.name: block.complexity
        for block in cc_visit(source_path.read_text(encoding="utf-8"))
        if block.name == "_resolve_land_context"
    }

    assert functions["_resolve_land_context"] < 6
