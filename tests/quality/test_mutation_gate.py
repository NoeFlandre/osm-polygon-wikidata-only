"""Fail-closed contracts for the mutation gate and its equivalence reviews."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import tomllib
from pathlib import Path

import pytest

from scripts.quality import mutation_equivalents, mutation_gate
from scripts.quality.mutation_equivalents import _review_fields, reviewed_equivalents

REPOSITORY = Path(__file__).resolve().parents[2]
NAME = "scripts.example.x_query__mutmut_1"
FILE_NAME = "scripts.example.x_query__mutmut_2"
SOURCE = "def query():\n    return False\n"
GENERATED = "def x_query__mutmut_1():\n    return None\n"
FILE_GENERATED = "def x_query__mutmut_2():\n    return True\n"


def _gate(report: str, monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(mutation_gate.sys, "stdin", io.StringIO(report))
    return mutation_gate.main(argv)


def test_mutation_gate_passes_only_when_every_mutant_is_killed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mutation_equivalents, "REVIEWED_EQUIVALENTS", {})
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
    monkeypatch.setattr(mutation_equivalents, "REVIEWED_EQUIVALENTS", {})

    assert (
        _gate(f"other: killed\n{NAME}: survived\n", monkeypatch, "--equivalents", "reviews.json")
        == 0
    )
    assert capsys.readouterr().out.endswith("1 mutants killed; 1 reviewed equivalents\n")

    monkeypatch.setattr(mutation_equivalents, "REVIEWED_EQUIVALENTS", {NAME: entry})
    assert _gate(f"{NAME}: survived\n", monkeypatch) == 0
    assert (
        capsys.readouterr().out
        == "Mutation gate passed: 0 mutants killed; 1 reviewed equivalents\n"
    )
    monkeypatch.setattr(mutation_equivalents, "REVIEWED_EQUIVALENTS", {})

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


def test_mutation_gate_combines_builtin_and_json_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "scripts/example.py"
    source.parent.mkdir()
    source.write_text(SOURCE)
    generated = tmp_path / "mutants/scripts/example.py"
    generated.parent.mkdir(parents=True)
    generated.write_text(f"{GENERATED}\n{FILE_GENERATED}")

    def review(mutant: str, reason: str) -> dict[str, str]:
        return {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "mutant_sha256": hashlib.sha256(
                ast.dump(ast.parse(mutant).body[0]).encode()
            ).hexdigest(),
            "reason": reason,
        }

    built_in_review = review(GENERATED, "Fixture built-in equivalent.")
    file_review = review(FILE_GENERATED, "Fixture JSON equivalent.")
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps({FILE_NAME: file_review}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mutation_equivalents, "REVIEWED_EQUIVALENTS", {NAME: built_in_review})

    assert (
        _gate(
            f"{NAME}: survived\n{FILE_NAME}: survived\n",
            monkeypatch,
            "--equivalents",
            "reviews.json",
        )
        == 0
    )
    assert capsys.readouterr().out == (
        "Mutation gate passed: 0 mutants killed; 2 reviewed equivalents\n"
    )


def test_mutation_scope_names_only_existing_sources_and_tests() -> None:
    mutation = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "mutmut"
    ]
    configured = [*mutation["source_paths"], *mutation["pytest_add_cli_args_test_selection"]]

    assert [path for path in configured if not (REPOSITORY / path).is_file()] == []


@pytest.mark.parametrize("reason", [None, "", " \t"])
def test_equivalence_reviews_reject_empty_or_non_string_text(reason: object) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        _review_fields(
            {
                "source_sha256": "source",
                "mutant_sha256": "mutant",
                "reason": reason,
            }
        )
