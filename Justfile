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

# Build the minimal non-root runtime image; this never touches a data root.
docker-build:
    docker build --target runtime --tag osm-polygon-wikidata-only:local .

# Run the harmless default help command; this never touches a data root.
docker-help: docker-build
    docker run --rm osm-polygon-wikidata-only:local --help

# Build and run the development test target; this uses no production data.
docker-test:
    docker build --target development --tag osm-polygon-wikidata-only:dev .
    docker run --rm osm-polygon-wikidata-only:dev

# Run the development quality checks; this uses no production data.
docker-check:
    docker build --target development --tag osm-polygon-wikidata-only:dev .
    docker run --rm osm-polygon-wikidata-only:dev bash -lc \
        'uv run pytest -q && uv run ruff check src tests scripts && uv run ruff format --check src tests scripts && uv run ty check src scripts'

# Opt-in data/publish operation for a host root containing `raw/` and resumable state.
docker-run data_root: docker-build
    docker run --rm -it \
        --user "$(id -u):$(id -g)" \
        --env HOME=/tmp \
        --mount "type=bind,src={{data_root}},dst=/data" \
        --mount "type=bind,src={{data_root}}/raw,dst=/data/raw,readonly" \
        --env HF_TOKEN \
        --env WIKIMEDIA_BOT_USERNAME \
        --env WIKIMEDIA_BOT_PASSWORD \
        --env WIKIMEDIA_REQUESTS_PER_MINUTE \
        osm-polygon-wikidata-only:local \
        sync-dir /data/raw --data-root /data --skip-existing --push

check: sync coverage lint format-check typecheck build docs
    git diff --check
