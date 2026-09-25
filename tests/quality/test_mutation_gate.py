"""Fail-closed contracts for the mutation gate and its equivalence reviews."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import tomllib
from pathlib import Path

import pytest

from scripts.quality import mutation_gate
from scripts.quality.mutation_equivalents import reviewed_equivalents

REPOSITORY = Path(__file__).resolve().parents[2]
NAME = "scripts.example.x_query__mutmut_1"
SOURCE = "def query():\n    return False\n"
GENERATED = "def x_query__mutmut_1():\n    return None\n"


def _gate(report: str, monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(mutation_gate.sys, "stdin", io.StringIO(report))
    return mutation_gate.main(argv)


def test_mutation_gate_passes_only_when_every_mutant_is_killed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _gate("one: killed\n", monkeypatch) == 0
    assert capsys.readouterr().out == "Mutation gate passed: 1 mutants killed\n"
    for report, message in (
        ("", "No mutants"),
        ("one: killed\ntwo: survived\n", "two: survived"),
        ("one: timeout\n", "one: timeout"),
        ("one: changed behavior\n", "Unknown mutation status"),
        (": killed\n", "missing mutant name"),
    ):
        with pytest.raises(mutation_gate.MutationGateError, match=message):
            _gate(report, monkeypatch)


def test_equivalence_reviews_are_exact_and_source_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "scripts/example.py"
    source.parent.mkdir()
    source.write_text(SOURCE)
    generated = tmp_path / "mutants/scripts/example.py"
    generated.parent.mkdir(parents=True)
    generated.write_text(GENERATED)
    entry = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mutant_sha256": hashlib.sha256(
            ast.dump(ast.parse(GENERATED).body[0]).encode()
        ).hexdigest(),
        "reason": "Fixture only: the caller uses truthiness, so both values match.",
    }
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps({NAME: entry}))
    monkeypatch.chdir(tmp_path)

    assert (
        _gate(f"other: killed\n{NAME}: survived\n", monkeypatch, "--equivalents", "reviews.json")
        == 0
    )
    assert capsys.readouterr().out.endswith("1 mutants killed; 1 reviewed equivalents\n")

    def validate(results: list[tuple[str, str]]) -> frozenset[str]:
        return reviewed_equivalents(
            results, reviews, source_root=tmp_path, mutants_root=tmp_path / "mutants"
        )

    with pytest.raises(ValueError, match="Stale"):
        validate([(NAME, "killed")])
    with pytest.raises(ValueError, match="Stale"):
        validate([])
    generated.write_text(GENERATED.replace("None", "True"))
    with pytest.raises(ValueError, match="Mutation changed"):
        validate([(NAME, "survived")])
    source.write_text(SOURCE + "# changed caller\n")
    with pytest.raises(ValueError):
        validate([(NAME, "survived")])


def test_mutation_scope_names_only_existing_sources_and_tests() -> None:
    mutation = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "mutmut"
    ]
    configured = [*mutation["source_paths"], *mutation["pytest_add_cli_args_test_selection"]]

    assert [path for path in configured if not (REPOSITORY / path).is_file()] == []
