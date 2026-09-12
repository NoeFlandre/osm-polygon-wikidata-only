"""Executable import-boundary contracts for the local package graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.quality.architecture import (
    ArchitectureViolation,
    build_import_graph,
    check_architecture,
    main,
)


def _write_package(tmp_path: Path, files: dict[str, str]) -> Path:
    package_root = tmp_path / "pkg"
    for relative_path, source in files.items():
        path = package_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return package_root


def _rules(violations: tuple[ArchitectureViolation, ...]) -> set[str]:
    return {violation.rule for violation in violations}


def test_resolves_absolute_relative_from_and_nested_imports(tmp_path: Path) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "support.py": "VALUE = 1\n",
            "io/__init__.py": "",
            "io/reader.py": "",
            "domain/__init__.py": "",
            "domain/other.py": "",
            "domain/reader.py": (
                "from pkg.domain.other import marker\n"
                "from ..support import VALUE\n"
                "\n"
                "def read():\n"
                "    import pkg.io.reader\n"
            ),
        },
    )

    graph = build_import_graph(package_root, "pkg")

    assert {(edge.source, edge.target) for edge in graph["pkg.domain.reader"]} == {
        ("pkg.domain.reader", "pkg.domain.other"),
        ("pkg.domain.reader", "pkg.support"),
        ("pkg.domain.reader", "pkg.io.reader"),
    }


def test_reports_import_cycles_with_a_stable_cycle_rule(tmp_path: Path) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "a.py": "from .b import value\n",
            "b.py": "from .a import value\n",
        },
    )

    violations = check_architecture(package_root, "pkg")

    cycle_violations = tuple(v for v in violations if v.rule == "cycle")
    assert len(cycle_violations) == 1
    violation = cycle_violations[0]
    assert violation.source == "pkg.a"
    assert violation.target == "pkg.b"
    assert violation.path == package_root / "a.py"
    assert violation.line == 1
    assert violation.detail == "pkg.a -> pkg.b -> pkg.a"
    assert str(violation) == (
        f"{package_root / 'a.py'}:1: cycle: pkg.a -> pkg.b (pkg.a -> pkg.b -> pkg.a)"
    )


def test_reports_one_deterministic_cycle_when_multiple_cycles_exist(tmp_path: Path) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "a.py": "from .b import value\n",
            "b.py": "from .a import value\n",
            "c.py": "from .d import value\n",
            "d.py": "from .c import value\n",
        },
    )

    cycle_violations = tuple(
        violation
        for violation in check_architecture(package_root, "pkg")
        if violation.rule == "cycle"
    )

    assert len(cycle_violations) == 1
    assert "pkg.a -> pkg.b -> pkg.a" in str(cycle_violations[0])


def test_domain_rejects_non_domain_local_imports(tmp_path: Path) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "domain/__init__.py": "",
            "domain/rules.py": (
                "from pkg.augmentation import helper\n"
                "from pkg.enrichment import parser\n"
                "from pkg.hf import publisher\n"
                "from pkg.io import reader\n"
                "from pkg.v2 import runner\n"
            ),
            "augmentation/__init__.py": "",
            "enrichment/__init__.py": "",
            "hf/__init__.py": "",
            "io/__init__.py": "",
            "v2/__init__.py": "",
        },
    )

    violations = check_architecture(package_root, "pkg")

    assert _rules(violations) == {"domain-purity"}
    detail = "domain may import only local domain modules; external imports are not represented"
    assert [
        (
            violation.rule,
            violation.source,
            violation.target,
            violation.path,
            violation.line,
            violation.detail,
        )
        for violation in violations
    ] == [
        (
            "domain-purity",
            "pkg.domain.rules",
            "pkg.augmentation",
            package_root / "domain/rules.py",
            1,
            detail,
        ),
        (
            "domain-purity",
            "pkg.domain.rules",
            "pkg.enrichment",
            package_root / "domain/rules.py",
            2,
            detail,
        ),
        (
            "domain-purity",
            "pkg.domain.rules",
            "pkg.hf",
            package_root / "domain/rules.py",
            3,
            detail,
        ),
        (
            "domain-purity",
            "pkg.domain.rules",
            "pkg.io",
            package_root / "domain/rules.py",
            4,
            detail,
        ),
        (
            "domain-purity",
            "pkg.domain.rules",
            "pkg.v2",
            package_root / "domain/rules.py",
            5,
            detail,
        ),
    ]


def test_boundary_rules_use_the_top_level_layer_for_nested_modules(tmp_path: Path) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "domain/__init__.py": "",
            "domain/rules/__init__.py": "",
            "domain/rules/reader.py": "import pkg.io\n",
            "io/__init__.py": "",
        },
    )

    violations = check_architecture(package_root, "pkg")

    assert violations == (
        ArchitectureViolation(
            "domain-purity",
            "pkg.domain.rules.reader",
            "pkg.io",
            package_root / "domain/rules/reader.py",
            1,
            "domain may import only local domain modules; external imports are not represented",
        ),
    )


def test_pipeline_rejects_cli_imports(tmp_path: Path) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "pipeline/__init__.py": "",
            "pipeline/runner.py": "from pkg.cli import main\n",
            "cli/__init__.py": "",
        },
    )

    violations = check_architecture(package_root, "pkg")

    assert _rules(violations) == {"pipeline-cli"}
    assert violations[0].source == "pkg.pipeline.runner"
    assert violations[0].target == "pkg.cli"


def test_v2_and_hf_pipeline_edges_are_allowed_orchestration_boundaries(
    tmp_path: Path,
) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "v2/__init__.py": "",
            "v2/runner.py": "from pkg.pipeline.orchestrator import collect\n",
            "hf/__init__.py": "",
            "hf/publication.py": "from pkg.pipeline.processor import ProcessResult\n",
            "pipeline/__init__.py": "",
            "pipeline/orchestrator.py": "",
            "pipeline/processor.py": "",
        },
    )

    assert check_architecture(package_root, "pkg") == ()


def test_external_imports_are_not_local_architecture_edges(tmp_path: Path) -> None:
    """Stdlib and third-party imports are intentionally outside this graph."""
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "domain/__init__.py": "",
            "domain/value.py": "import pathlib\nimport pyarrow\n",
        },
    )

    assert check_architecture(package_root, "pkg") == ()


def test_missing_source_root_fails_with_a_specific_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="source root does not exist"):
        check_architecture(tmp_path / "missing", "pkg")


def test_empty_source_root_cannot_pass_as_a_null_graph(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    with pytest.raises(ValueError, match="contains no Python modules"):
        check_architecture(empty_root, "pkg")


def test_main_reports_success_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "domain/__init__.py": "",
            "domain/value.py": "import pathlib\n",
        },
    )

    result = main(["--source-root", str(package_root), "--package", "pkg"])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "Architecture checks passed for pkg.\n"
    assert captured.err == ""


def test_main_reports_a_boundary_violation_and_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "pipeline/__init__.py": "",
            "pipeline/runner.py": "from pkg.cli import main\n",
            "cli/__init__.py": "",
        },
    )

    result = main(["--source-root", str(package_root), "--package", "pkg"])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == (
        f"{package_root / 'pipeline' / 'runner.py'}:1: pipeline-cli: "
        "pkg.pipeline.runner -> pkg.cli "
        "(pipeline must not import CLI modules)\n"
    )


def test_main_help_describes_public_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    captured = capsys.readouterr()
    assert exit_info.value.code == 0
    assert "Small AST-based import-boundary checker" in captured.out
    assert "--source-root" in captured.out
    assert "--package" in captured.out
    assert captured.err == ""


def test_main_defaults_to_repository_source_root(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = Path(__file__).parents[2]
    monkeypatch.chdir(repository)

    result = main([])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "Architecture checks passed for osm_polygon_wikidata_only.\n"
    assert captured.err == ""


def test_main_defaults_to_the_public_package_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package_root = _write_package(tmp_path, {"__init__.py": ""})

    result = main(["--source-root", str(package_root)])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "Architecture checks passed for osm_polygon_wikidata_only.\n"
    assert captured.err == ""


def test_repository_import_boundaries_are_clean() -> None:
    repository = Path(__file__).parents[2]
    package_root = repository / "src" / "osm_polygon_wikidata_only"

    assert check_architecture(package_root, "osm_polygon_wikidata_only") == ()
