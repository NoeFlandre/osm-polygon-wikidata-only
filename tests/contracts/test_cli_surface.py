"""Characterization tests for the stable CLI surface.

These tests pass against the existing baseline and freeze the
documented CLI surface. They are NOT forced red.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from osm_polygon_wikidata_only.cli.parser import build_parser

EXPECTED_COMMANDS = {
    "process-pbf",
    "process-dir",
    "sync-dir",
    "augment-region",
    "augment-dir",
}

EXPECTED_COMMON_FLAGS = {
    "--data-root",
    "--repo-id",
    "--user-agent",
    "--languages",
    "--all-languages",
    "--no-full-text",
    "--max-articles-per-qid",
    "--enrichment-batch-size",
    "--enrichment-site-workers",
    "--limit",
    "--skip-existing",
    "--force",
    "--push",
    "--commit-message",
    "--upload-threads",
    "--hf-token",
    "--log-level",
    "--dry-run",
}


def _subcommands(parser: argparse.ArgumentParser) -> set[str]:
    sub_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return set(sub_action.choices)


def _flag_names(parser: argparse.ArgumentParser) -> set[str]:
    flags: set[str] = set()
    for action in parser._actions:
        flags.update(action.option_strings)
        for sub_action in getattr(action, "_get_kwargs_actions", lambda: [])():
            flags.update(getattr(sub_action, "option_strings", []))
    sub_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    for sub in sub_action.choices.values():
        for sub_parser_action in sub._actions:
            flags.update(sub_parser_action.option_strings)
    return flags


def test_parser_exposes_documented_subcommands() -> None:
    assert _subcommands(build_parser()) == EXPECTED_COMMANDS


def test_parser_exposes_documented_common_flags() -> None:
    flags = _flag_names(build_parser())
    missing = EXPECTED_COMMON_FLAGS - flags
    assert not missing, f"missing documented flags: {missing}"


def test_sync_dir_accepts_skip_existing_and_push(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["sync-dir", str(tmp_path), "--skip-existing", "--push", "--dry-run"]
    )
    assert args.command == "sync-dir"
    assert args.input == tmp_path
    assert args.skip_existing is True
    assert args.push is True
    assert args.dry_run is True


def test_augment_region_requires_stem_argument() -> None:
    args = build_parser().parse_args(["augment-region", "andorra-latest"])
    assert args.command == "augment-region"
    assert args.stem == "andorra-latest"


def test_process_pbf_no_full_text_disables_field(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "process-pbf",
            str(tmp_path / "x.osm.pbf"),
            "--no-full-text",
            "--languages",
            "en,fr",
            "--max-articles-per-qid",
            "3",
            "--limit",
            "50",
        ]
    )
    assert args.no_full_text is True
    assert args.languages == "en,fr"
    assert args.max_articles_per_qid == 3
    assert args.limit == 50


def test_log_level_accepts_documented_values() -> None:
    for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        args = build_parser().parse_args(
            ["process-pbf", str(Path("/tmp/x.osm.pbf")), "--log-level", level]
        )
        assert args.log_level == level


def test_log_level_rejects_unknown_value() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["process-pbf", str(Path("/tmp/x.osm.pbf")), "--log-level", "TRACE"]
        )


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "osm-polygon-wikidata-only" in captured.out
