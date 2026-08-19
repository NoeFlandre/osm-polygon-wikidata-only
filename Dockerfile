# syntax=docker/dockerfile:1.7
#
# The default uv image tag is version-pinned to a known uv/Python/Debian
# combination. Dependency versions are pinned separately by uv.lock. Operators
# may override UV_IMAGE with a verified registry digest for bit-for-bit image
# provenance without changing the Dockerfile.

ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.1-python3.12-trixie-slim
FROM ${UV_IMAGE} AS build

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# Install the locked dependency set before copying source files so rebuilds
# after a code-only change reuse the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY assets ./assets
RUN uv sync --frozen --no-dev --no-editable

# The development target is used by the Docker test/check recipes. It is
# deliberately separate from the runtime target so production images contain
# no tests, docs, source checkout, or development tools.
FROM build AS development
COPY . .
RUN uv sync --frozen

RUN groupadd --system app && useradd --system --gid app --create-home app \
    && chown -R app:app /app
USER app
ENV PATH=/app/.venv/bin:$PATH \
    OSM_POLYGON_DATA_ROOT=/data
WORKDIR /app
CMD ["uv", "run", "pytest", "-q"]

# Runtime images contain only the installed application and its runtime
# dependencies. No token, PBF, generated artifact, or local cache is copied
# into the image.
FROM ${UV_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    OSM_POLYGON_DATA_ROOT=/data \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --create-home app
COPY --from=build --chown=app:app /app/.venv /app/.venv

# Mount the external data root here. It holds raw PBFs, resumable state,
# caches, and generated artifacts; it is intentionally not part of the image.
VOLUME ["/data"]
USER app

# A plain `docker run IMAGE` is a harmless help command. Pass the explicit
# sync/process command to opt into any data work.
ENTRYPOINT ["osm-polygon-wikidata-only"]
CMD ["--help"]
