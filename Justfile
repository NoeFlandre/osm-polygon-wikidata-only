set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

sync:
    uv sync --frozen

test:
    uv run pytest -q

coverage:
    uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q

lint:
    uv run ruff check .

format:
    uv run ruff format .

format-check:
    uv run ruff format --check .

typecheck:
    uv run ty check src scripts

build:
    uv build

check: sync coverage lint format-check typecheck build
    git diff --check
