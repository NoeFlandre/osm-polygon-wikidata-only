"""Tests for installed distribution smoke validation."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.quality import package_smoke


def test_verify_resources_accepts_nested_files_and_reports_missing_files(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    resource = package_root / "assets" / "hero.png"
    resource.parent.mkdir(parents=True)
    resource.write_bytes(b"asset")

    package_smoke.verify_resources(package_root, ("assets/hero.png",))

    with pytest.raises(
        package_smoke.PackageSmokeError,
        match=r"Missing packaged resources: missing\.txt, other\.txt",
    ):
        package_smoke.verify_resources(package_root, ("missing.txt", "other.txt"))


def test_verify_resources_splits_resource_paths_for_traversables() -> None:
    calls: list[tuple[str, ...]] = []
    resource = SimpleNamespace(is_file=lambda: True)

    def joinpath(*parts: str) -> SimpleNamespace:
        calls.append(parts)
        return resource

    package_root = SimpleNamespace(joinpath=joinpath)

    package_smoke.verify_resources(package_root, ("assets/hero.png",))
    assert calls == [("assets", "hero.png")]


def test_top_level_names_cover_entry_point_bindings() -> None:
    source = """
import os.path as path
import os.path
import sys
from typing import Final as TypeAlias
from package import *
from package import value

async def async_main():
    pass

class Container:
    pass

value = 1
typed: str = "value"
(first, second) = (1, 2)
"""

    assert package_smoke._top_level_names(source, "package.cli") == {
        "path",
        "os",
        "sys",
        "TypeAlias",
        "value",
        "async_main",
        "Container",
        "typed",
    }


def test_top_level_names_reports_invalid_source() -> None:
    with pytest.raises(
        package_smoke.PackageSmokeError, match="Invalid entry-point module"
    ) as raised:
        package_smoke._top_level_names("def broken(", "package.cli")
    assert isinstance(raised.value.__cause__, SyntaxError)
    assert raised.value.__cause__.filename == "package.cli"


def test_module_resource_resolves_package_modules_and_rejects_other_packages(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    package_init = package_root / "__init__.py"
    package_init.write_text("", encoding="utf-8")
    nested = package_root / "nested"
    nested.mkdir()
    nested_init = nested / "__init__.py"
    nested_init.write_text("", encoding="utf-8")
    dotted_module = package_root / "cli.py"
    dotted_module.write_text("", encoding="utf-8")

    assert package_smoke._module_resource(package_root, "package", "package") == package_init
    assert package_smoke._module_resource(package_root, "package", "package.nested") == nested_init
    assert (
        package_smoke._module_resource(package_root, "package.nested", "package.nested.cli")
        == dotted_module
    )
    assert package_smoke._module_resource(package_root, "package", "other.cli") is None


def test_read_entry_point_source_requires_utf8_encoding() -> None:
    encodings: list[str | None] = []

    class Resource:
        def read_text(self, *, encoding: str | None) -> str:
            encodings.append(encoding)
            return "source"

    entry_point = EntryPoint("cli", "package.cli:main", "console_scripts")
    assert package_smoke._read_entry_point_source(Resource(), entry_point) == "source"
    assert encodings == ["utf-8"]


def test_verify_entry_point_target_preserves_metadata_and_uses_first_attr_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    (package_root / "cli.py").write_text("", encoding="utf-8")
    entry_point = EntryPoint("cli", "package.cli:main.wrapper.call", "console_scripts")
    seen: list[EntryPoint] = []

    def read_source(_resource: object, received: EntryPoint) -> str:
        seen.append(received)
        return "def main():\n    return 0\n"

    monkeypatch.setattr(package_smoke, "_read_entry_point_source", read_source)

    package_smoke._verify_entry_point_target(package_root, "package", entry_point)
    assert seen == [entry_point]


def test_verify_entry_point_target_rejects_empty_attr_even_if_fallback_name_exists(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    (package_root / "cli.py").write_text("XXXX = object()\n", encoding="utf-8")
    entry_point = SimpleNamespace(
        name="cli",
        value="package.cli:",
        module="package.cli",
        attr=None,
    )

    with pytest.raises(package_smoke.PackageSmokeError, match="target"):
        package_smoke._verify_entry_point_target(package_root, "package", entry_point)


def test_verify_entry_points_requires_expected_names_and_targets() -> None:
    valid = (EntryPoint("cli", "package.cli:main", "console_scripts"),)
    package_smoke.verify_entry_points(valid, ("cli",))

    with pytest.raises(
        package_smoke.PackageSmokeError,
        match=r"Missing console entry points: missing-a, missing-b",
    ):
        package_smoke.verify_entry_points(valid, ("cli", "missing-a", "missing-b"))

    malformed = (
        EntryPoint("cli-a", "package.cli", "console_scripts"),
        EntryPoint("cli-b", "package.other", "console_scripts"),
    )
    with pytest.raises(
        package_smoke.PackageSmokeError,
        match=r"Malformed console entry points: cli-a=package\.cli, cli-b=package\.other",
    ):
        package_smoke.verify_entry_points(malformed, ("cli-a", "cli-b"))


def test_verify_distribution_checks_metadata_and_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "package"
    resource = package_root / "data" / "schema.json"
    resource.parent.mkdir(parents=True)
    resource.write_text("{}", encoding="utf-8")
    (package_root / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    distribution = SimpleNamespace(
        entry_points=(EntryPoint("cli", "package.cli:main", "console_scripts"),)
    )
    calls: list[tuple[str, str]] = []

    def get_distribution(name: str) -> SimpleNamespace:
        calls.append(("distribution", name))
        return distribution

    def get_resources(name: str) -> Path:
        calls.append(("package", name))
        return package_root

    monkeypatch.setattr(package_smoke.metadata, "distribution", get_distribution)
    monkeypatch.setattr(package_smoke.resources, "files", get_resources)

    package_smoke.verify_distribution("package", "package", ("data/schema.json",), ("cli",))
    assert calls == [("distribution", "package"), ("package", "package")]


@pytest.mark.parametrize(
    ("entry_point", "message"),
    [
        ("package.missing:main", "module"),
        ("package.cli:missing", "target"),
    ],
)
def test_verify_distribution_rejects_unavailable_entry_point_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    message: str,
) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    (package_root / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    distribution = SimpleNamespace(
        entry_points=(EntryPoint("cli", entry_point, "console_scripts"),)
    )
    monkeypatch.setattr(package_smoke.metadata, "distribution", lambda _name: distribution)
    monkeypatch.setattr(package_smoke.resources, "files", lambda _name: package_root)

    with pytest.raises(package_smoke.PackageSmokeError, match=message):
        package_smoke.verify_distribution("package", "package", (), ("cli",))


def test_verify_distribution_reports_missing_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> object:
        raise package_smoke.metadata.PackageNotFoundError("missing")

    monkeypatch.setattr(package_smoke.metadata, "distribution", missing)

    with pytest.raises(package_smoke.PackageSmokeError, match="distribution"):
        package_smoke.verify_distribution("missing", "package", (), ())


def test_verify_distribution_reports_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = SimpleNamespace(entry_points=())
    monkeypatch.setattr(package_smoke.metadata, "distribution", lambda _name: distribution)

    def missing(_name: str) -> object:
        raise ModuleNotFoundError("package")

    monkeypatch.setattr(package_smoke.resources, "files", missing)

    with pytest.raises(package_smoke.PackageSmokeError, match="package"):
        package_smoke.verify_distribution("distribution", "package", (), ())


def test_main_forwards_the_declared_artifact_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []

    def capture(
        distribution_name: str,
        package_name: str,
        resource_paths: tuple[str, ...],
        entry_point_names: tuple[str, ...],
    ) -> None:
        calls.append((distribution_name, package_name, resource_paths, entry_point_names))

    monkeypatch.setattr(package_smoke, "verify_distribution", capture)

    assert (
        package_smoke.main(
            [
                "--distribution",
                "distribution",
                "--package",
                "package",
                "--resource",
                "data/schema.json",
                "--entry-point",
                "cli",
            ]
        )
        == 0
    )
    assert calls == [
        ("distribution", "package", ("data/schema.json",), ("cli",)),
    ]


def test_main_defaults_optional_contracts_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def capture(
        _distribution_name: str,
        _package_name: str,
        resource_paths: tuple[str, ...],
        entry_point_names: tuple[str, ...],
    ) -> None:
        calls.append((resource_paths, entry_point_names))

    monkeypatch.setattr(package_smoke, "verify_distribution", capture)

    assert package_smoke.main(["--distribution", "distribution", "--package", "package"]) == 0
    assert calls == [((), ())]


@pytest.mark.parametrize(
    "arguments",
    [[], ["--distribution", "distribution"]],
)
def test_main_requires_distribution_and_package(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        package_smoke.main(arguments)
    assert error.value.code == 2


def test_main_help_includes_the_command_description(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        package_smoke.main(["--help"])

    assert error.value.code == 0
    assert "Validate installed package metadata" in capsys.readouterr().out
