"""Startup import guards for metadata-only CLI invocations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE = """
import json
import sys

sys.argv = ["osm-polygon-wikidata-only", sys.argv[1]]
from osm_polygon_wikidata_only.cli.app import run

try:
    run()
except SystemExit as error:
    if error.code not in (None, 0):
        raise

prefixes = (
    "pyarrow",
    "httpx",
    "huggingface_hub",
    "osmium",
    "numpy",
    "matplotlib",
    "typer",
    "osm_polygon_wikidata_only.cli.commands",
    "osm_polygon_wikidata_only.cli.dependencies",
    "osm_polygon_wikidata_only.cli.grid5000",
    "osm_polygon_wikidata_only.cli.audit_containment",
    "osm_polygon_wikidata_only.cli.audit_remote",
    "osm_polygon_wikidata_only.cli.enforce_integrity",
    "osm_polygon_wikidata_only.cli.tools",
    "osm_polygon_wikidata_only.grid5000.sentence_controller",
    "osm_polygon_wikidata_only.grid5000.sentence_job",
    "osm_polygon_wikidata_only.grid5000.sentence_protocol",
    "osm_polygon_wikidata_only.pipeline",
)
loaded = sorted(
    prefix
    for prefix in prefixes
    if any(name == prefix or name.startswith(prefix + ".") for name in sys.modules)
)
print("CLI_IMPORTS=" + json.dumps(loaded))
print(
    "CLI_PARSER_LOADED="
    + json.dumps("osm_polygon_wikidata_only.cli.parser" in sys.modules)
)
"""

_PACKAGE_ENTRYPOINT_PROBE = """
import json
import sys

sys.argv = ["osm-polygon-wikidata-only", "--version"]
from osm_polygon_wikidata_only import main

status = main()
loaded = sorted(name for name in sys.modules if name.startswith("osm_polygon_wikidata_only.cli"))
print("CLI_ENTRYPOINT_MODULES=" + json.dumps(loaded))
heavy_prefixes = (
    "pyarrow",
    "httpx",
    "huggingface_hub",
    "osmium",
    "numpy",
    "matplotlib",
    "typer",
    "osm_polygon_wikidata_only.pipeline",
    "osm_polygon_wikidata_only.grid5000.sentence_controller",
    "osm_polygon_wikidata_only.grid5000.sentence_job",
    "osm_polygon_wikidata_only.grid5000.sentence_protocol",
)
heavy = sorted(
    prefix
    for prefix in heavy_prefixes
    if any(name == prefix or name.startswith(prefix + ".") for name in sys.modules)
)
print("CLI_HEAVY_MODULES=" + json.dumps(heavy))
sys.exit(status)
"""


@pytest.mark.parametrize("argument", ["--version", "--help"])
def test_metadata_cli_invocations_skip_command_implementation_imports(argument: str) -> None:
    repository = Path(__file__).resolve().parents[2]
    source_root = repository / "src"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    python_path = [str(source_root), environment.get("PYTHONPATH", "")]
    environment["PYTHONPATH"] = os.pathsep.join(path for path in python_path if path)

    result = subprocess.run(
        [sys.executable, "-c", _PROBE, argument],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    import_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("CLI_IMPORTS=")),
        None,
    )
    assert import_line is not None, result.stdout
    loaded = json.loads(import_line.removeprefix("CLI_IMPORTS="))
    assert loaded == [], f"{argument} imported command stacks: {loaded}"
    parser_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("CLI_PARSER_LOADED=")),
        None,
    )
    assert parser_line is not None, result.stdout
    parser_loaded = json.loads(parser_line.removeprefix("CLI_PARSER_LOADED="))
    if argument == "--version":
        assert not parser_loaded, "--version should not construct the command parser"
    else:
        assert parser_loaded, "--help should use the command parser"


def test_package_version_entrypoint_skips_cli_modules() -> None:
    repository = Path(__file__).resolve().parents[2]
    source_root = repository / "src"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    python_path = [str(source_root), environment.get("PYTHONPATH", "")]
    environment["PYTHONPATH"] = os.pathsep.join(path for path in python_path if path)

    result = subprocess.run(
        [sys.executable, "-c", _PACKAGE_ENTRYPOINT_PROBE],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("osm-polygon-wikidata-only ")
    import_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("CLI_ENTRYPOINT_MODULES=")),
        None,
    )
    assert import_line is not None, result.stdout
    loaded = json.loads(import_line.removeprefix("CLI_ENTRYPOINT_MODULES="))
    assert loaded == []
    heavy_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("CLI_HEAVY_MODULES=")),
        None,
    )
    assert heavy_line is not None, result.stdout
    heavy = json.loads(heavy_line.removeprefix("CLI_HEAVY_MODULES="))
    assert heavy == [], f"--version imported heavy modules: {heavy}"
