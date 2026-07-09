"""Tests for the CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from osm_polygon_wikidata_only.cli.commands import build_parser


def test_parser_has_two_subcommands() -> None:
    parser = build_parser()
    sub_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub_action.choices) == {"process-pbf", "process-dir"}


def test_parser_process_pbf_accepts_input(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(["process-pbf", str(tmp_path / "x.osm.pbf"), "--all-languages"])
    assert args.command == "process-pbf"
    assert args.all_languages is True
    assert args.no_full_text is False


def test_parser_process_pbf_no_full_text_disables_field(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "process-pbf",
            str(tmp_path / "x.osm.pbf"),
            "--no-full-text",
            "--languages",
            "en,fr",
        ]
    )
    assert args.no_full_text is True
    assert args.languages == "en,fr"


def test_parser_process_dir_default_skip() -> None:
    parser = build_parser()
    args = parser.parse_args(["process-dir", "/tmp/abc"])
    assert args.skip_existing is False
    assert args.force is False


def test_parser_push_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "process-pbf",
            "/tmp/x.osm.pbf",
            "--push",
            "--repo-id",
            "foo/bar",
        ]
    )
    assert args.push is True
    assert args.repo_id == "foo/bar"
