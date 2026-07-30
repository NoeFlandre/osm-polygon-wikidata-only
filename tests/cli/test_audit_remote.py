"""Contracts for the read-only Typer/Rich/tqdm operator audit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner


def _module() -> Any:
    from osm_polygon_wikidata_only.cli import audit_remote

    return audit_remote


def test_audit_help_is_public_and_focused() -> None:
    module = _module()
    result = CliRunner().invoke(module.app, ["--help"])

    assert result.exit_code == 0
    assert "Audit remote versus local canonical dataset files" in result.stdout
    assert "--data-root" in result.stdout
    assert "--repo-id" in result.stdout
    assert "--hf-token" in result.stdout
    assert "sync-dir" not in result.stdout


def test_audit_uses_sorted_progress_and_renders_reconciliation(
    tmp_path: Path, monkeypatch: Any
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

    result = CliRunner().invoke(module.app, ["--data-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert observed == [
        "progress:Checking local augmentation:region",
        "andorra-latest",
        "zambia-latest",
    ]
    assert "Local finalized regions" in result.stdout
    assert "2" in result.stdout
    assert "andorra-latest.parquet" in result.stdout
    assert "zambia-latest.parquet" in result.stdout
    assert "unexpected/file.parquet" in result.stdout
    assert "README.md" in result.stdout


def test_audit_reports_inventory_failure_without_traceback(
    tmp_path: Path, monkeypatch: Any
) -> None:
    module = _module()
    data_root = SimpleNamespace(processed_polygons=tmp_path)
    monkeypatch.setattr(module, "resolve_data_root", lambda *_args, **_kwargs: data_root)

    def fail(**_kwargs: object) -> object:
        raise RuntimeError("inventory unavailable")

    monkeypatch.setattr(module.RemoteInventory, "fetch", fail)
    result = CliRunner().invoke(module.app, ["--data-root", str(tmp_path)])

    assert result.exit_code == 1
    assert "Failed to fetch remote inventory" in result.stdout
    assert "inventory unavailable" in result.stdout
    assert "Traceback" not in result.stdout
