from __future__ import annotations

from pathlib import Path

import pytest

from scripts.quality.architecture import ImportEdge, build_import_graph, check_architecture


def _write_package(tmp_path: Path, files: dict[str, str]) -> Path:
    package_root = tmp_path / "pkg"
    for relative_path, source in files.items():
        path = package_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return package_root


def _edge_facts(
    package_root: Path, edges: tuple[ImportEdge, ...]
) -> list[tuple[str, str, str, int]]:
    return sorted(
        (
            edge.source,
            edge.target,
            edge.path.relative_to(package_root).as_posix(),
            edge.line,
        )
        for edge in edges
    )


def test_public_graph_resolves_absolute_relative_and_symbol_imports_with_provenance(
    tmp_path: Path,
) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "support.py": "VALUE = 1\n",
            "other/__init__.py": "",
            "feature/__init__.py": ("from pkg import support, other\nfrom . import sibling\n"),
            "feature/sibling.py": "",
            "feature/sub/__init__.py": "from .. import sibling\n",
            "feature/sub/leaf.py": (
                "from .. import sibling\n"
                "from ...support import VALUE\n"
                "import pkg.support, pkg.feature.sub.leaf\n"
            ),
        },
    )

    graph = build_import_graph(package_root, "pkg")

    assert _edge_facts(package_root, graph["pkg.feature"]) == [
        ("pkg.feature", "pkg.feature.sibling", "feature/__init__.py", 2),
        ("pkg.feature", "pkg.other", "feature/__init__.py", 1),
        ("pkg.feature", "pkg.support", "feature/__init__.py", 1),
    ]
    assert _edge_facts(package_root, graph["pkg.feature.sub"]) == [
        ("pkg.feature.sub", "pkg.feature.sibling", "feature/sub/__init__.py", 1),
    ]
    assert _edge_facts(package_root, graph["pkg.feature.sub.leaf"]) == [
        ("pkg.feature.sub.leaf", "pkg.feature.sibling", "feature/sub/leaf.py", 1),
        ("pkg.feature.sub.leaf", "pkg.feature.sub.leaf", "feature/sub/leaf.py", 3),
        ("pkg.feature.sub.leaf", "pkg.support", "feature/sub/leaf.py", 2),
        ("pkg.feature.sub.leaf", "pkg.support", "feature/sub/leaf.py", 3),
    ]


def test_public_boundary_diagnostics_preserve_source_target_path_and_line(
    tmp_path: Path,
) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "domain/__init__.py": "",
            "domain/rules.py": ("from pkg.pipeline.runner import run\nimport pkg.hf\n"),
            "pipeline/__init__.py": "",
            "pipeline/runner.py": "",
            "hf/__init__.py": "",
        },
    )

    violations = check_architecture(package_root, "pkg")
    facts = sorted(
        (
            violation.rule,
            violation.source,
            violation.target,
            violation.path.relative_to(package_root).as_posix(),
            violation.line,
            violation.detail,
        )
        for violation in violations
    )

    assert facts == [
        (
            "domain-purity",
            "pkg.domain.rules",
            "pkg.hf",
            "domain/rules.py",
            2,
            "domain may import only local domain modules; external imports are not represented",
        ),
        (
            "domain-purity",
            "pkg.domain.rules",
            "pkg.pipeline.runner",
            "domain/rules.py",
            1,
            "domain may import only local domain modules; external imports are not represented",
        ),
    ]
    assert str(violations[0]).endswith(
        "domain-purity: pkg.domain.rules -> pkg.hf "
        "(domain may import only local domain modules; external imports are not represented)"
    )


def test_unparsable_module_reports_its_real_path_in_the_syntax_error(tmp_path: Path) -> None:
    package_root = _write_package(
        tmp_path,
        {"__init__.py": "", "broken.py": "def missing_body(\n"},
    )

    with pytest.raises(SyntaxError) as error:
        build_import_graph(package_root, "pkg")

    assert error.value.filename == str(package_root / "broken.py")


def test_modules_are_read_using_their_declared_source_encoding(tmp_path: Path) -> None:
    package_root = _write_package(tmp_path, {"__init__.py": "", "support.py": "VALUE = 1\n"})
    latin_1_module = package_root / "legacy.py"
    latin_1_module.write_bytes(
        "# -*- coding: latin-1 -*-\nfrom pkg import support\nLABEL = 'café'\n".encode("latin-1")
    )

    graph = build_import_graph(package_root, "pkg")

    assert _edge_facts(package_root, graph["pkg.legacy"]) == [
        ("pkg.legacy", "pkg.support", "legacy.py", 2),
    ]


def test_relative_import_deeper_than_its_package_creates_no_edge(tmp_path: Path) -> None:
    package_root = _write_package(
        tmp_path,
        {
            "__init__.py": "",
            "support.py": "VALUE = 1\n",
            "module.py": "from .. import support\n",
        },
    )

    graph = build_import_graph(package_root, "pkg")

    assert graph["pkg.module"] == ()


def test_top_level_package_import_resolves_to_the_package_module(tmp_path: Path) -> None:
    package_root = _write_package(
        tmp_path,
        {"__init__.py": "", "module.py": "import pkg\n"},
    )

    graph = build_import_graph(package_root, "pkg")

    assert _edge_facts(package_root, graph["pkg.module"]) == [
        ("pkg.module", "pkg", "module.py", 1),
    ]
