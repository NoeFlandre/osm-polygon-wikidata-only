"""Tests for the function-level CRAP quality report."""

from __future__ import annotations

from pathlib import Path

import pytest
from radon.complexity import cc_visit

from scripts.quality.crap_score import (
    CrapEntry,
    _entries_for_file,
    _line_numbers,
    _radon_file_parts,
    _radon_function_parts,
    crap_score,
    entries_from_reports,
    evaluate_threshold,
    main,
)


def test_crap_score_uses_standard_formula() -> None:
    assert crap_score(10, 0.75) == 11.5625


def test_crap_score_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="complexity"):
        crap_score(0, 0.5)
    with pytest.raises(ValueError, match="coverage"):
        crap_score(5, 1.1)


@pytest.mark.parametrize("complexity", [True, "5"])
def test_crap_score_rejects_non_integer_complexity(complexity: object) -> None:
    with pytest.raises(ValueError, match="complexity"):
        crap_score(complexity, 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("coverage", [True, "0.5", float("nan"), float("inf"), -0.1])
def test_crap_score_rejects_non_finite_or_non_fraction_coverage(coverage: object) -> None:
    with pytest.raises(ValueError, match="coverage"):
        crap_score(5, coverage)  # type: ignore[arg-type]


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


@pytest.mark.parametrize("maximum", [-1.0, float("nan")])
def test_evaluate_threshold_rejects_invalid_limits(maximum: float) -> None:
    with pytest.raises(ValueError, match="maximum"):
        evaluate_threshold([], maximum=maximum)


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


def test_entries_from_reports_resolves_a_unique_coverage_suffix() -> None:
    entries = entries_from_reports(
        {
            "files": {
                "/tmp/src/module.py": {
                    "functions": {"parse": {"summary": {"percent_covered": 100}}}
                }
            }
        },
        {"src/module.py": [{"type": "function", "name": "parse", "complexity": 1}]},
    )

    assert entries[0].coverage == 1.0


def test_entries_from_reports_rejects_ambiguous_or_missing_coverage_files() -> None:
    complexity = {"src/module.py": [{"type": "function", "name": "parse", "complexity": 1}]}
    ambiguous = {
        "files": {
            "/one/src/module.py": {"functions": {}},
            "/two/src/module.py": {"functions": {}},
        }
    }
    with pytest.raises(ValueError, match=r"coverage file src/module\.py"):
        entries_from_reports(ambiguous, complexity)
    with pytest.raises(ValueError, match=r"coverage file src/module\.py"):
        entries_from_reports({"files": {}}, complexity)


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"type": "function", "name": None, "complexity": 1}, "entry is malformed"),
        ({"type": "function", "name": "f", "complexity": True}, "entry is malformed"),
        (
            {"type": "function", "name": "f", "complexity": 1, "classname": 3},
            "classname is malformed",
        ),
        (
            {"type": "function", "name": "f", "complexity": 1, "lineno": True},
            "line is malformed",
        ),
        (
            {"type": "function", "name": "f", "complexity": 1, "lineno": 4, "endline": 3},
            "endline is malformed",
        ),
    ],
)
def test_entries_from_reports_rejects_malformed_radon_functions(
    block: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        entries_from_reports(
            {"files": {"module.py": {"functions": {}}}},
            {"module.py": [block]},
        )


def test_entries_from_reports_rejects_empty_or_non_object_blocks() -> None:
    with pytest.raises(ValueError, match="no function entries"):
        entries_from_reports({"files": {}}, {})
    with pytest.raises(ValueError, match="Radon function entry must be an object"):
        entries_from_reports(
            {"files": {"module.py": {"functions": {}}}},
            {"module.py": [None]},
        )


def test_line_numbers_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="executed_lines"):
        _line_numbers("not a list", "executed_lines")
    with pytest.raises(ValueError, match="missing_lines"):
        _line_numbers([1, False], "missing_lines")


def test_entries_from_reports_rejects_invalid_coverage_summary() -> None:
    with pytest.raises(ValueError, match="no percentage"):
        entries_from_reports(
            {
                "files": {
                    "module.py": {"functions": {"parse": {"summary": {"percent_covered": "100"}}}}
                }
            },
            {"module.py": [{"type": "function", "name": "parse", "complexity": 1}]},
        )


def test_crap_cli_reports_pass_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    coverage_path = tmp_path / "coverage.json"
    complexity_path = tmp_path / "complexity.json"
    coverage_path.write_text(
        '{"files":{"module.py":{"functions":{"parse":{"summary":{"percent_covered":100}}}}}}',
        encoding="utf-8",
    )
    complexity_path.write_text(
        '{"module.py":[{"type":"function","name":"parse","complexity":1,"lineno":4}]}',
        encoding="utf-8",
    )
    arguments = [
        "--coverage",
        str(coverage_path),
        "--complexity",
        str(complexity_path),
    ]

    assert main([*arguments, "--maximum", "1.0"]) == 1
    assert "CRAP threshold exceeded" in capsys.readouterr().out
    assert main([*arguments, "--maximum", "2.0"]) == 0
    assert "CRAP threshold passed" in capsys.readouterr().out


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


def test_entries_for_file_keeps_methods_and_skips_class_blocks() -> None:
    entries = _entries_for_file(
        "module.py",
        [
            {"type": "class", "name": "Parser", "complexity": 4, "lineno": 1},
            {"type": "method", "name": "parse", "classname": "Parser", "complexity": 2},
        ],
        {"module.py": {"functions": {"Parser.parse": {"summary": {"percent_covered": 100}}}}},
    )

    assert entries == [CrapEntry("module.py", "Parser.parse", 2, 1.0, 0)]


def test_radon_function_parts_qualifies_methods_and_defaults_endline() -> None:
    assert _radon_function_parts(
        {"name": "parse", "classname": "Parser", "complexity": 2, "lineno": 4}
    ) == ("Parser.parse", 2, 4, 4)


def test_radon_file_parts_validate_the_file_shape() -> None:
    assert _radon_file_parts("module.py", []) == ("module.py", [])


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
