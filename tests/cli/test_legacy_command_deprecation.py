"""Legacy command paths keep their stdout, JSON and exit codes (issue #174).

The standalone console scripts and ``scripts/audit_containment.py`` remain as
compatibility paths. These tests pin what each one writes to stdout and the
status it returns, and check that its replacement subcommand is named in
``docs/cli-reference.md``.
"""

from __future__ import annotations

import hashlib
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.cli import audit_containment
from tests.cli.test_enforce_integrity_cli import _seed_defect

REPOSITORY = Path(__file__).resolve().parents[2]
SHIM = REPOSITORY / "scripts" / "audit_containment.py"
CLI_REFERENCE = REPOSITORY / "docs" / "cli-reference.md"

MAIN = "osm-polygon-wikidata-only"
ENFORCE = "osm-polygon-wikidata-only-enforce-integrity"
AUDIT_REMOTE = "osm-polygon-wikidata-only-audit-remote"
TRACKIO = "osm-polygon-wikidata-only-trackio"
TRACKIO_V2 = "osm-polygon-wikidata-and-wikipedia-trackio"

ENFORCE_JSON = (
    '{"audit_path": null, "dry_run": true, "polygon_articles_rejected": 1, '
    '"wikivoyage_documents_rejected": 0, "wikivoyage_sections_cascaded": 0}\n'
)
AUDIT_REMOTE_MISSING_ROOT = (
    "Error resolving data root: Data root absent (explicit --data-root) does not \nexist.\n"
)
CONTAINMENT_ABSENT_ROOT_SHA256 = "876a4a8d2694b29d76e6d211877b95c0beb87b4d5d8a8e87fae4ac146e1b5685"
EMPTY_CONTAINMENT_JSON = (
    '{\n  "blocked_parents": [],\n  "reports": [],\n  "retired_children": [],\n'
    '  "safe_parents": []\n}\n'
)


def _console(name: str) -> str:
    path = Path(sys.executable).with_name(name)
    assert path.is_file(), f"console script not installed: {path}"
    return str(path)


def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "COLUMNS": "80", "NO_COLOR": "1", "TERM": "dumb"}
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=300,
    )


def test_enforce_integrity_standalone_pins_json_stdout_and_exit_code(tmp_path: Path) -> None:
    _seed_defect(tmp_path / "data")

    proc = _run([_console(ENFORCE), "--data-root", "data", "--dry-run", "--json"], tmp_path)

    assert proc.returncode == 0
    assert proc.stdout == ENFORCE_JSON


def test_enforce_integrity_replacement_prints_the_same_json(tmp_path: Path) -> None:
    _seed_defect(tmp_path / "data")

    proc = _run(
        [_console(MAIN), "enforce-integrity", "--data-root", "data", "--dry-run", "--json"],
        tmp_path,
    )

    assert proc.returncode == 0
    assert proc.stdout == ENFORCE_JSON


def test_enforce_integrity_standalone_missing_root_exits_1_with_empty_stdout(
    tmp_path: Path,
) -> None:
    proc = _run([_console(ENFORCE), "--data-root", "absent"], tmp_path)

    assert proc.returncode == 1
    assert proc.stdout == ""
    assert (
        f"{ENFORCE}: error: Data root absent (explicit --data-root) does not exist." in proc.stderr
    )


def test_enforce_integrity_standalone_help_pins_usage_line(tmp_path: Path) -> None:
    proc = _run([_console(ENFORCE), "--help"], tmp_path)

    assert proc.returncode == 0
    assert proc.stdout.startswith(f"usage: {ENFORCE} [-h]\n")


def test_audit_remote_standalone_missing_root_pins_stdout_and_exit_code(tmp_path: Path) -> None:
    proc = _run(
        [_console(AUDIT_REMOTE), "--data-root", "absent", "--repo-id", "o/r", "--hf-token", "t"],
        tmp_path,
    )

    assert proc.returncode == 1
    assert proc.stdout == AUDIT_REMOTE_MISSING_ROOT


def test_audit_remote_standalone_help_exits_0(tmp_path: Path) -> None:
    proc = _run([_console(AUDIT_REMOTE), "--help"], tmp_path)

    assert proc.returncode == 0
    assert f"Usage: {AUDIT_REMOTE} [OPTIONS]" in proc.stdout


@pytest.mark.parametrize(
    ("name", "help_text"),
    [
        (TRACKIO, "Publish one static run and exactly three plots."),
        (TRACKIO_V2, "Publish the V2 card metrics and three static plots."),
    ],
)
def test_trackio_standalone_help_exits_0(tmp_path: Path, name: str, help_text: str) -> None:
    proc = _run([_console(name), "--help"], tmp_path)

    assert proc.returncode == 0
    assert f"Usage: {name} [OPTIONS]" in proc.stdout
    assert help_text in proc.stdout


@pytest.mark.parametrize("name", [TRACKIO, TRACKIO_V2])
def test_trackio_standalone_usage_error_exits_2_with_empty_stdout(
    tmp_path: Path, name: str
) -> None:
    proc = _run([_console(name), "--no-such-option"], tmp_path)

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "No such option: --no-such-option" in proc.stderr


def test_containment_script_absent_root_pins_stdout_and_exit_code_2(tmp_path: Path) -> None:
    proc = _run([sys.executable, str(SHIM), "absent"], tmp_path)

    assert proc.returncode == 2
    assert hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest() == (
        CONTAINMENT_ABSENT_ROOT_SHA256
    )


def test_containment_script_help_exits_0(tmp_path: Path) -> None:
    proc = _run([sys.executable, str(SHIM), "--help"], tmp_path)

    assert proc.returncode == 0
    assert proc.stdout.startswith("usage: audit_containment.py [-h] [--output OUTPUT] data_root\n")


def test_containment_script_pins_json_and_exit_0_when_no_rule_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(audit_containment, "CONTAINMENT_RULES", ())
    monkeypatch.setattr(audit_containment, "load_retired_children", lambda _processed: set())
    monkeypatch.setattr(sys, "argv", [str(SHIM), str(tmp_path)])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SHIM), run_name="__main__")

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == EMPTY_CONTAINMENT_JSON


@pytest.mark.parametrize(
    "subcommand",
    ["enforce-integrity", "audit-remote", "trackio-snapshot", "audit-containment"],
)
def test_replacement_subcommand_is_named_in_cli_reference(subcommand: str) -> None:
    text = CLI_REFERENCE.read_text(encoding="utf-8")

    assert f"uv run {MAIN} {subcommand}" in text


NOTICE_TARGETS = {
    ENFORCE: "osm-polygon-wikidata-only enforce-integrity",
    AUDIT_REMOTE: "osm-polygon-wikidata-only audit-remote",
    TRACKIO: "osm-polygon-wikidata-only trackio-snapshot",
    TRACKIO_V2: "osm-polygon-wikidata-only trackio-snapshot --dataset-version v2",
}


@pytest.mark.parametrize(("name", "replacement"), list(NOTICE_TARGETS.items()))
def test_standalone_executable_notice_goes_to_stderr_only(
    tmp_path: Path, name: str, replacement: str
) -> None:
    proc = _run([_console(name), "--help"], tmp_path)

    assert proc.returncode == 0
    assert f"{name}: warning: deprecated" in proc.stderr
    assert f"'{replacement}'" in proc.stderr
    assert "deprecated" not in proc.stdout


def test_containment_script_notice_goes_to_stderr_only(tmp_path: Path) -> None:
    proc = _run([sys.executable, str(SHIM), "--help"], tmp_path)

    assert proc.returncode == 0
    assert "scripts/audit_containment.py: warning: deprecated" in proc.stderr
    assert "'osm-polygon-wikidata-only audit-containment'" in proc.stderr
    assert "deprecated" not in proc.stdout


@pytest.mark.parametrize(
    "subcommand",
    [
        ["enforce-integrity", "--help"],
        ["audit-remote", "--help"],
        ["trackio-snapshot", "--help"],
        ["audit-containment", "--help"],
    ],
)
def test_replacement_subcommands_emit_no_deprecation_notice(
    tmp_path: Path, subcommand: list[str]
) -> None:
    proc = _run([_console(MAIN), *subcommand], tmp_path)

    assert proc.returncode == 0
    assert "deprecated" not in proc.stderr


def test_cli_reference_maps_each_legacy_name_to_its_replacement() -> None:
    text = CLI_REFERENCE.read_text(encoding="utf-8")
    start = text.index("## Legacy commands")
    end = text.find("\n## ", start + 1)
    section = text[start : len(text) if end == -1 else end]

    for legacy, replacement in {
        **NOTICE_TARGETS,
        "scripts/audit_containment.py": "osm-polygon-wikidata-only audit-containment",
    }.items():
        assert any(legacy in line and replacement in line for line in section.splitlines()), legacy
