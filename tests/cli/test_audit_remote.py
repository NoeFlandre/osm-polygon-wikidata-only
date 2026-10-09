"""Contracts for the read-only argparse/Rich/tqdm operator audit."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _module() -> Any:
    from osm_polygon_wikidata_only.cli import audit_remote

    return audit_remote


def test_audit_help_is_public_and_focused(capsys: pytest.CaptureFixture[str]) -> None:
    module = _module()
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--help"])
    plain_stdout = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", capsys.readouterr().out)

    assert exit_info.value.code == 0
    assert "Read-only audit of remote versus local canonical dataset files" in plain_stdout
    assert "--data-root" in plain_stdout
    assert "--repo-id" in plain_stdout
    assert "--hf-token" in plain_stdout
    assert "sync-dir" not in plain_stdout


def test_standalone_help_uses_the_shared_option_text(capsys: pytest.CaptureFixture[str]) -> None:
    module = _module()
    with pytest.raises(SystemExit):
        module.main(["--help"])
    stdout = capsys.readouterr().out

    assert "Local dataset root; defaults to env var" in stdout
    assert "Hugging Face dataset repository" in stdout
    assert "Hugging Face token" in stdout


def test_standalone_usage_error_exits_with_status_two(capsys: pytest.CaptureFixture[str]) -> None:
    module = _module()
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--no-such-option"])

    assert exit_info.value.code == 2
    assert "usage: osm-polygon-wikidata-only-audit-remote" in capsys.readouterr().err


def test_unresolvable_data_root_is_reported_with_status_one(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()

    def fail(*_args: object, **_kwargs: object) -> object:
        raise module.DataRootError("bad root")

    monkeypatch.setattr(module, "resolve_data_root", fail)

    assert module.main(["--data-root", str(tmp_path)]) == 1
    assert capsys.readouterr().err == (
        "osm-polygon-wikidata-only audit-remote: error: cannot resolve data root: bad root\n"
        f"{module.STANDALONE_PROG}: warning: deprecated; use "
        "'osm-polygon-wikidata-only audit-remote' instead.\n"
    )


def test_audit_uses_sorted_progress_and_renders_reconciliation(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    processed = tmp_path / "processed" / "polygons"
    processed.mkdir(parents=True)
    for stem in ("zambia-latest", "andorra-latest"):
        (processed / f"{stem}.parquet").touch()

    data_root = SimpleNamespace(processed_polygons=processed)
    monkeypatch.setattr(module, "resolve_data_root", lambda *_args, **_kwargs: data_root)
    monkeypatch.setattr(
        module.RemoteInventory,
        "fetch",
        lambda **_kwargs: SimpleNamespace(paths=frozenset()),
    )

    observed: list[str] = []

    def current(_data_root: object, stem: str) -> bool:
        observed.append(stem)
        return stem == "andorra-latest"

    monkeypatch.setattr(module, "augmentation_is_current", current)

    class Planner:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            assert kwargs["stems"] == {"andorra-latest", "zambia-latest"}
            assert kwargs["augmentation_current"] == {
                "andorra-latest": True,
                "zambia-latest": False,
            }

        def plan(self) -> object:
            return SimpleNamespace(
                missing=frozenset(
                    {
                        ("andorra-latest", "polygons"),
                        ("zambia-latest", "wikipedia/documents"),
                    }
                ),
                unexpected=frozenset({"unexpected/file.parquet"}),
                repository_refresh=frozenset({"README.md"}),
            )

    monkeypatch.setattr(module, "ReconciliationPlanner", Planner)
    monkeypatch.setattr(
        module,
        "tqdm",
        lambda items, **kwargs: (
            observed.append(f"progress:{kwargs['desc']}:{kwargs['unit']}") or items
        ),
    )

    assert module.main(["--data-root", str(tmp_path)]) == 0
    stdout = capsys.readouterr().out
    assert observed == [
        "progress:Checking local augmentation:region",
        "andorra-latest",
        "zambia-latest",
    ]
    assert "Local finalized regions" in stdout
    assert "2" in stdout
    assert "andorra-latest.parquet" in stdout
    assert "zambia-latest.parquet" in stdout
    assert "unexpected/file.parquet" in stdout
    assert "README.md" in stdout


def test_audit_reports_inventory_failure_without_traceback(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    from osm_polygon_wikidata_only.hf._uploader.errors import UploadError

    data_root = SimpleNamespace(processed_polygons=tmp_path)
    monkeypatch.setattr(module, "resolve_data_root", lambda *_args, **_kwargs: data_root)

    def fail(**_kwargs: object) -> object:
        raise UploadError("inventory unavailable")

    monkeypatch.setattr(module.RemoteInventory, "fetch", fail)

    assert module.main(["--data-root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.err == (
        "osm-polygon-wikidata-only audit-remote: error: "
        "failed to fetch remote inventory: inventory unavailable\n"
        f"{module.STANDALONE_PROG}: warning: deprecated; use "
        "'osm-polygon-wikidata-only audit-remote' instead.\n"
    )
    assert "Traceback" not in captured.out + captured.err


def test_audit_does_not_suppress_unexpected_data_root_errors(
    tmp_path: Path, monkeypatch: Any
) -> None:
    module = _module()
    error = TypeError("data-root programming bug")
    monkeypatch.setattr(module, "repository_root", lambda: tmp_path)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(module, "resolve_data_root", fail)

    with pytest.raises(TypeError, match="data-root programming bug"):
        module._resolve_root(tmp_path)


def test_audit_does_not_suppress_unexpected_inventory_errors(monkeypatch: Any) -> None:
    module = _module()

    def fail(**_kwargs: object) -> object:
        raise RuntimeError("inventory programming bug")

    monkeypatch.setattr(module.RemoteInventory, "fetch", fail)

    with pytest.raises(RuntimeError, match="inventory programming bug"):
        module._fetch_inventory(module.Console(), "owner/repo", None)


def test_audit_does_not_suppress_unexpected_reconciliation_errors(
    monkeypatch: Any,
) -> None:
    module = _module()
    error = KeyError("reconciliation programming bug")

    class BrokenPlanner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def plan(self) -> object:
            raise error

    monkeypatch.setattr(module, "ReconciliationPlanner", BrokenPlanner)

    with pytest.raises(KeyError, match="reconciliation programming bug"):
        module._build_plan(object(), object(), [], {})
