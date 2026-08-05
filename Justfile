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
    uv run ruff check src tests scripts

format:
    uv run ruff format src tests scripts

format-check:
    uv run ruff format --check src tests scripts

typecheck:
    uv run ty check src scripts

build:
    uv build

docs:
    uv run mkdocs build --strict --site-dir /tmp/osm-polygon-wikidata-only-site

trackio:
    uv run osm-polygon-wikidata-only-trackio

check: sync coverage lint format-check typecheck build docs
    git diff --check
