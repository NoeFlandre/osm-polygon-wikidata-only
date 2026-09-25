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


def _verify_entry_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modules: dict[str, str],
    entry_point: object,
    *,
    package_name: str = "package",
) -> None:
    """Run ``verify_distribution`` against a fake installed package tree."""
    package_root = tmp_path / "package"
    for relative, source in modules.items():
        module = package_root / relative
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text(source, encoding="utf-8")
    package_root.mkdir(exist_ok=True)
    distribution = SimpleNamespace(entry_points=(entry_point,))
    monkeypatch.setattr(package_smoke.metadata, "distribution", lambda _name: distribution)
    monkeypatch.setattr(package_smoke.resources, "files", lambda _name: package_root)
    package_smoke.verify_distribution("distribution", package_name, (), ("cli",))


def _console(value: str) -> EntryPoint:
    return EntryPoint("cli", value, "console_scripts")


@pytest.mark.parametrize(
    ("source", "attr"),
    [
        ("import os.path as path\n", "path"),
        ("import os.path\n", "os"),
        ("from typing import Final as TypeAlias\n", "TypeAlias"),
        ("from package import value\n", "value"),
        ("async def async_main():\n    pass\n", "async_main"),
        ("class Container:\n    pass\n", "Container"),
        ("value = 1\n", "value"),
        ("typed: str = 'value'\n", "typed"),
        ("def main():\n    return 0\n", "main.wrapper.call"),
    ],
)
def test_entry_point_target_may_be_any_top_level_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, attr: str
) -> None:
    _verify_entry_point(tmp_path, monkeypatch, {"cli.py": source}, _console(f"package.cli:{attr}"))


@pytest.mark.parametrize(
    ("source", "attr"),
    [
        ("(first, second) = (1, 2)\n", "first"),
        ("from package import *\n", "*"),
        ("XXXX = object()\n", None),
        ("import os.path\n", "path"),
    ],
)
def test_entry_point_target_must_be_a_top_level_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, attr: str | None
) -> None:
    entry_point = SimpleNamespace(
        name="cli", value=f"package.cli:{attr or ''}", module="package.cli", attr=attr
    )
    with pytest.raises(package_smoke.PackageSmokeError, match="target unavailable"):
        _verify_entry_point(tmp_path, monkeypatch, {"cli.py": source}, entry_point)


@pytest.mark.parametrize(
    ("package_name", "module", "path"),
    [
        ("package", "package", "__init__.py"),
        ("package", "package.nested", "nested/__init__.py"),
        ("package.nested", "package.nested.cli", "cli.py"),
        ("package", "package.nested.cli", "nested/cli.py"),
    ],
)
def test_entry_point_module_resolves_inside_the_installed_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package_name: str, module: str, path: str
) -> None:
    _verify_entry_point(
        tmp_path,
        monkeypatch,
        {path: "def main():\n    return 0\n"},
        _console(f"{module}:main"),
        package_name=package_name,
    )


def test_entry_point_module_outside_the_package_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(package_smoke.PackageSmokeError, match="module unavailable"):
        _verify_entry_point(
            tmp_path, monkeypatch, {"cli.py": "def main(): pass\n"}, _console("other.cli:main")
        )


def test_entry_point_module_must_be_valid_utf8_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        package_smoke.PackageSmokeError, match="Invalid entry-point module"
    ) as raised:
        _verify_entry_point(
            tmp_path, monkeypatch, {"cli.py": "def broken("}, _console("package.cli:main")
        )
    assert isinstance(raised.value.__cause__, SyntaxError)
    assert raised.value.__cause__.filename == "package.cli"

    (tmp_path / "package" / "cli.py").write_bytes(b"main = '\xff'\n")
    with pytest.raises(package_smoke.PackageSmokeError, match="module unreadable"):
        _verify_entry_point(tmp_path, monkeypatch, {}, _console("package.cli:main"))


def test_entry_point_module_is_read_as_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    encodings: list[str | None] = []

    class Module:
        def is_file(self) -> bool:
            return True

        def read_text(self, *, encoding: str | None = None) -> str:
            encodings.append(encoding)
            return "def main():\n    return 0\n"

    package_root = SimpleNamespace(joinpath=lambda *_parts: Module())
    distribution = SimpleNamespace(entry_points=(_console("package.cli:main"),))
    monkeypatch.setattr(package_smoke.metadata, "distribution", lambda _name: distribution)
    monkeypatch.setattr(package_smoke.resources, "files", lambda _name: package_root)

    package_smoke.verify_distribution("distribution", "package", (), ("cli",))
    assert encodings == ["utf-8"]


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
    [
        [],
        ["--distribution", "distribution"],
        ["--package", "package"],
    ],
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
