"""Executable contracts for portable quality-recipe runtime paths."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROUTING_ENV_NAMES = frozenset(
    {
        "TMPDIR",
        "UV_CACHE_DIR",
        "RUFF_CACHE_DIR",
        "HYPOTHESIS_STORAGE_DIRECTORY",
        "MPLCONFIGDIR",
        "UV_PYTHON_INSTALL_DIR",
    }
)


def _run_just(
    *arguments: str, env: dict[str, str | None] | None = None
) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    for name in tuple(child_env):
        if name.startswith("QUALITY_") or name in _CACHE_ROUTING_ENV_NAMES:
            child_env.pop(name)
    if env:
        for name, value in env.items():
            if value is None:
                child_env.pop(name, None)
            else:
                child_env[name] = value
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        ["just", *arguments],
        cwd=ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_quality_runtime_recipe_defaults_to_the_checkout_parent() -> None:
    result = _run_just("--evaluate", "QUALITY_RUNTIME_DIR")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "../quality-runtime"


def test_quality_recipe_quotes_configured_runtime_with_spaces(tmp_path: Path) -> None:
    runtime = tmp_path / "quality runtime"
    child_env = {
        "QUALITY_RUNTIME_DIR": str(runtime),
        "QUALITY_REPORT_DIR": None,
        "QUALITY_CACHE_DIR": None,
        "QUALITY_TMP_DIR": None,
        "UV_CACHE_DIR": None,
        "TMPDIR": "/var/folders/hostile",
    }

    result = _run_just("--dry-run", "package-smoke", env=child_env)

    assert result.returncode == 0, result.stderr
    rendered = result.stdout + result.stderr
    assert f'mktemp -d "{runtime}/tmp/package-smoke.XXXXXX"' in rendered
    assert "uv venv --python" in rendered
    assert "sys.executable" in rendered
    assert "/var/folders/hostile" not in rendered

    uv_cache = _run_just("--evaluate", "UV_CACHE_DIR", env=child_env)
    assert uv_cache.returncode == 0, uv_cache.stderr
    assert uv_cache.stdout.strip() == str(runtime / "cache" / "uv-cache")


def test_package_smoke_recipe_template_is_executable_with_portable_mktemp(tmp_path: Path) -> None:
    runtime = tmp_path / "quality runtime"
    result = _run_just(
        "--dry-run",
        "package-smoke",
        env={"QUALITY_RUNTIME_DIR": str(runtime), "UV_CACHE_DIR": None},
    )
    assert result.returncode == 0, result.stderr
    rendered = result.stdout + result.stderr
    line = next(line for line in rendered.splitlines() if "mktemp -d" in line)
    command_text = line[line.index("mktemp -d") :]
    template = shlex.split(command_text)[2]
    Path(template).parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["mktemp", "-d", template],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    created = Path(result.stdout.strip())
    assert created.is_dir()
    shutil.rmtree(created)


def test_quality_recipe_exports_normalized_runtime_environment(tmp_path: Path) -> None:
    runtime = tmp_path / "quality runtime"
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    capture = tmp_path / "environment.txt"
    stub = stub_dir / "uv"
    stub.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$TMPDIR" "$UV_CACHE_DIR" "$RUFF_CACHE_DIR" '
        '"$HYPOTHESIS_STORAGE_DIRECTORY" "$MPLCONFIGDIR" '
        '"$UV_PYTHON_INSTALL_DIR" > "$QUALITY_ENV_CAPTURE"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)

    result = _run_just(
        "build",
        env={
            "QUALITY_RUNTIME_DIR": str(runtime),
            "QUALITY_REPORT_DIR": None,
            "QUALITY_CACHE_DIR": None,
            "QUALITY_TMP_DIR": None,
            "UV_CACHE_DIR": None,
            "RUFF_CACHE_DIR": None,
            "HYPOTHESIS_STORAGE_DIRECTORY": None,
            "MPLCONFIGDIR": None,
            "UV_PYTHON_INSTALL_DIR": None,
            "TMPDIR": "/var/folders/hostile",
            "QUALITY_ENV_CAPTURE": str(capture),
            "PATH": f"{stub_dir}:{os.environ['PATH']}",
        },
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        str(runtime / "tmp"),
        str(runtime / "cache" / "uv-cache"),
        str(runtime / "cache" / "ruff"),
        str(runtime / "cache" / "hypothesis"),
        str(runtime / "cache" / "matplotlib"),
        str(runtime / "cache" / "python"),
    ]


def test_baseline_recipe_executes_pytest() -> None:
    result = _run_just("--dry-run", "baseline")

    assert result.returncode == 0, result.stderr
    rendered = result.stdout + result.stderr
    assert "uv run python -m pytest" in rendered
    assert "--no-cov" in rendered
    assert "--basetemp" in rendered
    assert "baseline-pytest" in rendered


def test_preprocessing_coverage_is_reused_by_its_crap_recipe(tmp_path: Path) -> None:
    runtime = tmp_path / "quality runtime"
    child_env = {
        "QUALITY_RUNTIME_DIR": str(runtime),
        "QUALITY_REPORT_DIR": None,
        "QUALITY_CACHE_DIR": None,
        "QUALITY_TMP_DIR": None,
        "UV_CACHE_DIR": None,
        "TMPDIR": "/var/folders/hostile",
    }

    preprocessing = _run_just("--dry-run", "preprocessing-check", env=child_env)
    assert preprocessing.returncode == 0, preprocessing.stderr
    preprocessing_output = preprocessing.stdout + preprocessing.stderr
    coverage_data = runtime / "reports" / "preprocessing.coverage"
    coverage_json = runtime / "reports" / "preprocessing-coverage.json"
    assert f'COVERAGE_FILE="{coverage_data}"' in preprocessing_output
    assert (
        f'uv run python -m coverage json --data-file="{coverage_data}" -o "{coverage_json}"'
    ) in preprocessing_output
    assert "/var/folders/hostile" not in preprocessing_output

    crap = _run_just("--dry-run", "crap-report", env=child_env)
    assert crap.returncode == 0, crap.stderr
    crap_output = crap.stdout + crap.stderr
    assert "radon cc --show-closures -j preprocessing/src" in crap_output
    assert f'--coverage "{coverage_json}"' in crap_output


def test_crap_recipe_report_includes_a_nested_function(tmp_path: Path) -> None:
    source = tmp_path / "nested.py"
    source.write_text(
        "def outer(value):\n"
        "    def on_retry(item):\n"
        "        if item:\n"
        "            return value\n"
        "        return None\n",
        encoding="utf-8",
    )
    result = _run_just(
        "--dry-run",
        "crap-report",
        env={"QUALITY_REPORT_DIR": str(tmp_path / "reports")},
    )
    assert result.returncode == 0, result.stderr
    rendered = result.stdout + result.stderr
    line = next(line for line in rendered.splitlines() if "radon cc" in line)
    command_text = line[line.index("radon cc") : line.index(">")].strip()
    configured_command = shlex.split(command_text)
    radon_index = configured_command.index("radon")
    radon_executable = Path(sys.executable).with_name("radon")
    if radon_executable.is_file():
        command = [str(radon_executable), *configured_command[radon_index + 1 :]]
    else:
        command = ["uv", "run", *configured_command[radon_index:]]
    command[command.index("-j") + 1 :] = [str(source)]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert any(entry["name"] == "outer.on_retry" for entry in report[str(source)])
