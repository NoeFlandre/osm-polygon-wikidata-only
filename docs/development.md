# Development guide

## Setup

The project supports Python 3.12 and uses `uv` for reproducible environments:

```bash
uv sync --frozen
```

Install [`just`](https://just.systems/) as the project command runner, then
install the repository's fast pre-commit hooks:

```bash
uv run pre-commit install
just --list
```

## Docker reproducibility

The checked-in `Dockerfile` provides a reproducible container path for the
runtime and for quality checks. It uses the version-pinned
`ghcr.io/astral-sh/uv:0.11.16-python3.12-bookworm-slim` base and
installs the Python dependency graph only from `uv.lock`. The image runs as an
unprivileged `app` user, uses a deterministic UTF-8/bytecode environment, and
contains no credentials or data files.

The versioned tag fixes the toolchain release, while a registry tag can still
be moved. For bit-for-bit image provenance, build with a digest resolved from
your trusted registry inventory:

```bash
docker build \
  --build-arg UV_IMAGE=ghcr.io/astral-sh/uv:<verified-tag>@sha256:<verified-digest> \
  --target runtime \
  --tag osm-polygon-wikidata-only:local .
```

The repository intentionally does not invent or hard-code a digest without a
verified registry manifest; the lockfile remains the authoritative dependency
resolution in every build.

The `just docker-build` recipe is a thin wrapper around `docker build` and
selects the runtime stage by default.

Build a minimal runtime image and verify its harmless default help command:

```bash
just docker-build
just docker-help
```

Run the development tests or the focused container quality checks:

```bash
just docker-test
just docker-check
```

The external data root is the only writable container volume. Mount it at
`/data`; it must contain the input PBFs below `/data/raw` and retains all
resumable state under the same tree. The checkout is never copied into the
runtime image, and raw/generated files are excluded from the Docker build
context by `.dockerignore`. The recipe uses Docker's explicit
`--mount type=bind,src=<host-data-root>,dst=/data` form and maps the host UID/GID
with `--user` so the non-root process can write resumable state. It mounts the
`raw/` subdirectory again as `readonly`, protecting source PBFs from accidental
container writes.

The full data workflow is deliberately opt-in. `docker-run` first builds (or
reuses) the cached runtime image, then requires a host data-root path explicitly:

```bash
just docker-run /path/to/osm-polygon-data
```

This invokes the same resumable `sync-dir --skip-existing --push` workflow as
the host command. The mounted state survives container removal, so `Ctrl-C`
and rerunning the identical command resumes completed work. `HF_TOKEN` and the
optional Wikimedia Bot Password variables are passed through from the host;
they are never written into the image or Dockerfile. Docker builds, help, and
tests never read a real PBF or contact Hugging Face.

Production data belongs under `OSM_POLYGON_DATA_ROOT`; tests use temporary
directories and in-memory clients. The automated suite must not access the
network or require a real PBF collection.

## Wikimedia authentication environment

Production operators can provide a Bot Password using the complete generated
username and secret:

```bash
export WIKIMEDIA_BOT_USERNAME='AccountName@osm-polygon-pipeline'
read -rs WIKIMEDIA_BOT_PASSWORD
export WIKIMEDIA_BOT_PASSWORD
```

The pair is optional but all-or-nothing. With neither variable, clients remain
anonymous at 180 requests per minute. With both, dependency construction shares
one authenticated session and adaptive scheduler across Wikidata and Wikipedia.
The authenticated ceiling defaults to 1,200 requests per minute and can be
overridden with a positive number:

```bash
export WIKIMEDIA_REQUESTS_PER_MINUTE=600
```

Never use live credentials in tests. Pass an explicit environment mapping and
an injected fake opener/session, as `tests/enrichment/test_wikimedia_auth.py` and
`tests/cli/test_dependencies.py` do. Tests must assert that exception messages and
representations do not contain passwords. Do not add network-dependent login
tests to the automated suite.

## Test-driven changes

Use red-green-refactor: add one focused failing test, confirm the expected
failure, implement the smallest behavior, rerun the focused test, then run the
full gate. For structural changes, first add characterization coverage around
the boundary, move one responsibility, and prove output equivalence.

## Quality gate

```bash
just check
```

The recipes use uv to run pytest with coverage, Ruff lint and format checks, ty
over `src` and maintained `scripts`, the strict MkDocs build, the package build,
and the whitespace gate. GitHub Actions invokes the same quality recipes and
publishes the documentation site from `main`. Pre-commit intentionally runs
only the fast Ruff and ty subset; `just check` remains the complete gate.
Concretely, the gate owns `uv run ruff check src tests scripts`,
`uv run ruff format --check src tests scripts`, `uv run ty check src scripts`, `uv build`, and
`git diff --check`; contributors do not need to maintain a separate command
sequence.

Build the documentation site locally without starting a server:

```bash
just docs
```

Strict typing applies to `src/`. Decoded third-party JSON may begin as `Any`,
but public and internal boundaries should narrow it immediately.

## Operator audit

`osm-polygon-wikidata-only-audit-remote` is a read-only maintenance command:

```bash
uv run osm-polygon-wikidata-only-audit-remote \
  --data-root "$OSM_POLYGON_DATA_ROOT"
```

Typer owns this command's option parsing, Rich renders its report, and tqdm
shows interactive progress while local augmentation state is checked. tqdm is
disabled automatically when stderr is not a terminal, keeping captured logs
clean. The command calculates a reconciliation plan but never uploads, deletes,
or rewrites data. These interface libraries are deliberately not used by the
stable argparse production CLI or its `sync-dir` logging.

## Release checklist

Verify the quality gate, inspect wheel contents for `py.typed` and the license,
confirm CLI help from the built artifact, review dataset compatibility, and
update the version intentionally. Publishing packages or datasets is a
maintainer action and is not part of ordinary test execution.
