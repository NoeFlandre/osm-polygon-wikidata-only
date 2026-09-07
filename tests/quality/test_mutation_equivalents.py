"""Exact, source-bound mutation exemptions must fail closed."""

import ast
import hashlib
import json

import pytest

from scripts.quality import mutation_gate

NAME = "scripts.example.x_query__mutmut_1"
SOURCE = "def query():\n    return False\n"
GENERATED = "def x_query__mutmut_1():\n    return None\n"


@pytest.fixture
def review(tmp_path):
    source = tmp_path / "scripts/example.py"
    source.parent.mkdir()
    source.write_text(SOURCE)
    generated = tmp_path / "mutants/scripts/example.py"
    generated.parent.mkdir(parents=True)
    generated.write_text(GENERATED)
    node = ast.parse(GENERATED).body[0]
    entry = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mutant_sha256": hashlib.sha256(ast.dump(node).encode()).hexdigest(),
        "reason": "Fixture only: the caller uses truthiness, so both values select the same branch.",
    }
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps({NAME: entry}))
    return path, source, generated, entry


def validate(review, results=None):
    path, source, generated, _ = review
    from scripts.quality.mutation_equivalents import reviewed_equivalents

    return reviewed_equivalents(
        [(NAME, "survived")] if results is None else results,
        path,
        source_root=source.parents[1],
        mutants_root=generated.parents[1],
    )


def test_exact_review_accepts_only_the_unchanged_survivor(review):
    assert validate(review) == frozenset({NAME})


@pytest.mark.parametrize("target", [1, 2])
def test_changed_source_or_mutation_requires_new_review(review, target):
    review[target].write_text(SOURCE if target == 2 else SOURCE + "# changed caller\n")
    with pytest.raises(ValueError):
        validate(review)


@pytest.mark.parametrize("status", ["killed", "timeout", "no tests", "not checked"])
def test_review_for_a_non_survivor_is_stale(review, status):
    with pytest.raises(ValueError, match="Stale"):
        validate(review, [(NAME, status)])


def test_review_missing_from_report_is_stale(review):
    with pytest.raises(ValueError, match="Stale"):
        validate(review, [])


@pytest.mark.parametrize(
    "field,value", [("reason", " "), ("source_sha256", ""), ("mutant_sha256", "x"), ("reason", 1)]
)
def test_incomplete_review_is_rejected(review, field, value):
    review[3][field] = value
    review[0].write_text(json.dumps({NAME: review[3]}))
    with pytest.raises(ValueError):
        validate(review)


@pytest.mark.parametrize("payload", [[], None, {NAME: None}, {NAME: {}}, {"../escape": {}}])
def test_malformed_review_is_rejected(review, payload):
    review[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate(review)


def test_duplicate_json_keys_are_rejected(review):
    value = json.dumps(review[3])
    review[0].write_text(f'{{"{NAME}": {value}, "{NAME}": {value}}}')
    with pytest.raises(ValueError, match="Duplicate"):
        validate(review)


def test_changed_mutant_body_requires_new_review(review):
    review[2].write_text(GENERATED.replace("None", "True"))
    with pytest.raises(ValueError, match="Mutation changed"):
        validate(review)


def test_duplicate_generated_function_is_rejected(review):
    review[2].write_text(GENERATED + GENERATED)
    with pytest.raises(ValueError, match="Expected one"):
        validate(review)


@pytest.mark.parametrize(
    "name", ["single", "scripts.*.x_query__mutmut_1", "scripts../escape.x", "scripts.example.x-bad"]
)
def test_review_names_cannot_be_patterns_or_paths(review, name):
    review[0].write_text(json.dumps({name: review[3]}))
    with pytest.raises(ValueError, match="Invalid exact"):
        validate(review, [(name, "survived")])


def test_package_sources_are_resolved_under_src(review):
    name = "package.example.x_query__mutmut_1"
    root = review[1].parents[1]
    target = root / "src/package/example.py"
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE)
    mutant = root / "mutants/src/package/example.py"
    mutant.parent.mkdir(parents=True)
    mutant.write_text(GENERATED)
    review[0].write_text(json.dumps({name: review[3]}))
    assert validate(review, [(name, "survived")]) == frozenset({name})


def test_an_empty_review_does_not_exempt_anything(review):
    review[0].write_text("{}")
    assert validate(review) == frozenset()


def test_unknown_review_fields_are_rejected(review):
    review[3]["typo"] = "unexpected"
    review[0].write_text(json.dumps({NAME: review[3]}))
    with pytest.raises(ValueError, match="exact"):
        validate(review)


def test_cli_reports_reviewed_equivalents_separately(review, monkeypatch, capsys):
    import io

    monkeypatch.chdir(review[1].parents[1])
    monkeypatch.setattr(
        mutation_gate.sys, "stdin", io.StringIO(f"other: killed\n{NAME}: survived\n")
    )
    assert mutation_gate.main(["--equivalents", str(review[0])]) == 0
    assert (
        capsys.readouterr().out
        == "Mutation gate passed: 1 mutants killed; 1 reviewed equivalents\n"
    )
