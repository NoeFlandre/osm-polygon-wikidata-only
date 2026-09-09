"""Quality-gate contracts for the geographic NER pilot."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from scripts.quality.scope_manifest import NER_SCOPE, validate_scope

ROOT = Path(__file__).resolve().parents[2]

LEGACY_MUTATION_SOURCES = frozenset(
    {
        "src/osm_polygon_wikidata_only/v2/deduplication.py",
        "src/osm_polygon_wikidata_only/v2/fingerprints.py",
        "src/osm_polygon_wikidata_only/v2/wikipedia_tags.py",
        "src/osm_polygon_wikidata_only/enrichment/wikipedia/parsing.py",
        "src/osm_polygon_wikidata_only/enrichment/wikidata/parsing.py",
        "src/osm_polygon_wikidata_only/io/pbf_reader.py",
        "src/osm_polygon_wikidata_only/hf/_upload_state.py",
        "src/osm_polygon_wikidata_only/hf/_upload_retry.py",
        "src/osm_polygon_wikidata_only/v2/sentence_logic.py",
        "src/osm_polygon_wikidata_only/grid5000/sentence_protocol.py",
    }
)


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _justfile() -> str:
    return (ROOT / "Justfile").read_text(encoding="utf-8")


def _recipe(text: str, name: str) -> str:
    lines = text.splitlines()
    start = lines.index(f"{name}:")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith((" ", "\t")):
            break
        body.append(line)
    return "\n".join(body)


def _locked_packages(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        *re.findall(r'^name = "([^"]+)"$', text, re.MULTILINE),
        *re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]+)==", text, re.MULTILINE),
    }


def test_mutation_scope_keeps_legacy_paths_and_covers_ner() -> None:
    mutmut = _pyproject()["tool"]["mutmut"]
    source_paths = set(mutmut["source_paths"])
    test_selection = set(mutmut["pytest_add_cli_args_test_selection"])
    assert LEGACY_MUTATION_SOURCES <= source_paths
    assert set(NER_SCOPE.source_paths) <= source_paths
    assert set(NER_SCOPE.test_paths) <= test_selection


def test_ner_quality_scope_paths_exist() -> None:
    validate_scope(NER_SCOPE, ROOT)


def test_mutation_config_matches_the_manifest_for_ner_paths() -> None:
    mutation = _pyproject()["tool"]["mutmut"]
    ner_sources = {
        path
        for path in mutation["source_paths"]
        if path.startswith("src/osm_polygon_wikidata_only/ner/")
        or path
        in {
            "src/osm_polygon_wikidata_only/grid5000/ner_controller.py",
            "scripts/grid5000_geographic_ner.py",
            "scripts/prepare_geographic_ner_pilot.py",
        }
    }
    ner_tests = {
        path
        for path in mutation["pytest_add_cli_args_test_selection"]
        if path.startswith("tests/ner/") or path == "tests/grid5000/test_ner_controller.py"
    }
    assert ner_sources == set(NER_SCOPE.source_paths)
    assert ner_tests == set(NER_SCOPE.test_paths)


def test_mutation_statistics_accept_generated_dataclass_frames(monkeypatch) -> None:
    """Contract validation must remain discoverable through dataclass __init__."""
    import mutmut
    from mutmut.__main__ import Config, record_trampoline_hit

    config = SimpleNamespace(
        max_stack_depth=_pyproject()["tool"]["mutmut"]["max_stack_depth"],
        resolved_mutated_source_paths=[ROOT / "src"],
        track_dependencies=False,
    )
    hits: set[str] = set()
    monkeypatch.setattr(Config, "get", lambda: config)
    monkeypatch.setattr(mutmut, "_stats", hits)

    @dataclass(frozen=True)
    class ContractProbe:
        def __post_init__(self) -> None:
            record_trampoline_hit("ner.contract.validation")

    ContractProbe()
    assert hits == {"ner.contract.validation"}


def test_mutation_sandbox_includes_the_gpu_dependency_lock() -> None:
    copied = _pyproject()["tool"]["mutmut"]["also_copy"]
    lock = Path("requirements/geographic-ner-gpu.txt")
    assert any(lock == Path(path) or Path(path) in lock.parents for path in copied)


def test_coverage_discovers_pilot_script_execution() -> None:
    """Covered-line mutation selection must not silently omit executable scripts."""
    import coverage

    from scripts import prepare_geographic_ner_pilot

    measured = coverage.Coverage(config_file=str(ROOT / "pyproject.toml"), data_file=None)
    measured.start()
    try:
        prepare_geographic_ner_pilot._score("seed", "identity")
    finally:
        measured.stop()
    assert measured.get_data().lines(prepare_geographic_ner_pilot.__file__)


def test_quality_tests_collect_without_a_shell_pythonpath() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/quality/test_mutation_gate.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ner_crap_scope_is_explicit_and_in_every_strength_aggregate() -> None:
    text = _justfile()
    recipe = _recipe(text, "crap-ner")
    assert "--scope ner --kind source" in recipe
    assert "--scope ner --kind test" in recipe
    for path in (*NER_SCOPE.source_paths[:-1], *NER_SCOPE.test_paths):
        assert path not in recipe
    assert "--maximum 6" in recipe
    assert re.search(r"^crap-all:.*\bcrap-ner\b", text, re.MULTILINE)
    assert re.search(r"^quality-strength:.*\bcrap-ner\b", text, re.MULTILINE)
    assert re.search(r"^quality-advanced:.*\bcrap-ner\b", text, re.MULTILINE)


def test_ner_runtime_imports_are_declared_and_locked() -> None:
    project = _pyproject()["project"]
    dependencies = {
        str(value).split("[", 1)[0].split(">", 1)[0].split("=", 1)[0]
        for value in project["dependencies"]
    }
    assert {"huggingface-hub", "osmium", "pyarrow"} <= dependencies
    assert {"huggingface-hub", "osmium", "pyarrow"} <= _locked_packages(ROOT / "uv.lock")
    assert {"huggingface-hub", "pyarrow", "torch", "transformers"} <= _locked_packages(
        ROOT / "requirements/geographic-ner-gpu.txt"
    )


def test_runbook_shell_examples_parse_without_executing() -> None:
    runbook = (ROOT / "docs/geographic-ner.md").read_text(encoding="utf-8")
    examples = re.findall(r"```sh\n(.*?)\n```", runbook, re.DOTALL)
    assert examples
    for example in examples:
        result = subprocess.run(
            ["bash", "-n"], input=example, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr


def test_runbook_records_the_completed_pilot_boundary() -> None:
    runbook = (ROOT / "docs/geographic-ner.md").read_text(encoding="utf-8")
    for expected in (
        "Pilot execution record",
        "OAR job 4096718",
        "1,000",
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        "geographic_ner/geographic-ner-pilot-20260908/",
    ):
        assert expected in runbook
    assert "full dataset has not been run" in runbook.lower()
