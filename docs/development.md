# Development guide

## Setup

The project supports Python 3.12 and uses [`uv`](https://docs.astral.sh/uv/)
for the locked environment:

```bash
uv sync --frozen
uv run pre-commit install
just --list
```

The source checkout contains code, tests, and documentation. Keep PBFs, Parquet
files, credentials, generated cards, and other run output in an
operator-selected data root outside the checkout.

## Docker reproducibility

The checked-in `Dockerfile` has `runtime` and `development` targets. Both are
built from the locked `uv.lock` environment; the runtime image runs as a
non-root, unprivileged `app` user and contains no source data or credentials.
The `docker build` command selects either target without changing the host data
root.

Build and exercise the harmless default command:

```bash
just docker-build
just docker-help
```

Run development checks without production data:

```bash
just docker-test
just docker-check
```

To run the opt-in workflow, provide a host data root containing `raw/`:

```bash
just docker-run /path/to/osm-polygon-data
```

The recipe uses Docker's explicit `--mount` form, mounts the data root at
`/data`, and mounts `/data/raw` read-only. It passes `HF_TOKEN` and optional
Wikimedia credentials only at runtime. Removing the container does not remove
the host's resumable state; press `Ctrl-C` and rerun the same command to resume.
Docker builds, help, and tests do not read a real PBF or make Hugging
Face/Wikimedia requests.

## Wikimedia credentials

Wikimedia authentication is optional. The [README authentication section](https://github.com/NoeFlandre/osm-polygon-wikidata-only#wikimedia-bot-password-authentication)
describes how to create and revoke a least-privilege Bot Password. Keep the
password out of source files, logs, issues, and pull requests. The
`WIKIMEDIA_BOT_USERNAME`/`WIKIMEDIA_BOT_PASSWORD` pair is all-or-nothing:
supply the password securely, never log it, and do not commit it. The
`WIKIMEDIA_REQUESTS_PER_MINUTE` environment variable selects the client-side
request ceiling; that ceiling remains subject to Wikimedia's service limits.

The test suite never uses live credentials. Authentication tests pass explicit
environment mappings and fake transports (for example,
`tests/enrichment/test_wikimedia_auth.py` and
`tests/cli/test_dependencies.py`) and assert that errors do not echo secrets.

## Tests and quality checks

Use red-green-refactor for behavior or configuration changes: add one focused
failing test, confirm the expected failure, implement the smallest change, and
then refactor while the test remains green. Contract tests should check the
observable CLI, schema, workflow, or documentation behavior rather than private
implementation details.

The complete local gate is:

```bash
just check
```

It runs the frozen dependency setup, pytest with coverage, Ruff lint and format
checks, `ty` over `src` and maintained scripts, the package build, a strict
MkDocs build, and `git diff --check`. The individual commands are:

```bash
uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run ty check src scripts
uv run mkdocs build --strict --site-dir /tmp/osm-polygon-wikidata-only-site
uv build
git diff --check
```

### Mutation and complexity gates

The advanced gate covers the pure, deterministic Wikipedia and Wikidata parser
helpers plus the durable upload queue. It keeps reports in `/tmp`, uses two
mutation workers to limit peak Mac memory, and does not read production data.
HTML cleaning remains covered by the CRAP and branch-coverage gates; mutmut 3.7 cannot execute mutations inside
its `HTMLParser` subclass trampoline reliably, so those unsupported class
mutations are not counted as actionable results:

```bash
just crap
just mutation
just quality-advanced
```

`just crap`, `just crap-sync`, and `just crap-upload` join coverage.py and Radon
function reports and fail when any function reaches CRAP 6; every measured
score must therefore be below 6.
`just mutation` runs mutmut over the explicit two
module scope and fails unless every generated mutant is killed. These gates
are deliberately narrow: network clients, large data files, and external
publication are outside the mutation run.

Run `uv run pre-commit run --all-files` before opening a pull request. The
hooks intentionally run the fast Ruff and `ty` subset; `just check` remains the
complete gate used by GitHub Actions.

## Test strength checks

The normal gate measures regression coverage across the whole package. The
opt-in `just quality-strength` recipe adds focused checks for identity helpers,
parser/sync helpers, and durable upload-queue behavior:

```bash
just mutation
just crap
just crap-sync
just crap-upload
```

`mutmut` deliberately changes those helpers and requires every generated
mutant to be killed by the focused tests. `crap4py` reports the CRAP score,
which combines cyclomatic complexity and line coverage, and fails if any
targeted function reaches 6 or more. Keeping this gate focused makes the result
exhaustive and repeatable without pretending that a single mutation run can
meaningfully cover the entire I/O-heavy pipeline.

## Documentation and contribution

Build the site without starting a server:

```bash
just docs
```

Navigation targets must exist under `docs/`, links and images must resolve in a
clean checkout, and public examples must use current CLI options. The Pages
workflow builds with `--strict`, uploads only the generated site, and deploys
that artifact with least-privilege permissions.

Please read the repository's [contributing guide](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/CONTRIBUTING.md)
before proposing changes. Keep pull requests small, explain any compatibility
effect on schemas or CLI options, and never commit source data or credentials.

## Read-only operator audit

The `osm-polygon-wikidata-only-audit-remote` command reports local/remote
publication differences without uploading or deleting files:

```bash
uv run osm-polygon-wikidata-only-audit-remote \
  --data-root "$OSM_POLYGON_DATA_ROOT"
```

Typer parses this command, Rich renders the report, and tqdm shows progress only
when stderr is interactive. It is separate from the stable argparse processing
CLI and does not change dataset output.

## Release checklist

Before a release, run the complete gate, inspect the wheel for `py.typed` and
the license, verify CLI help from the built artifact, review dataset schemas and
attribution, and update the version intentionally. Publishing software or
datasets is a maintainer action; ordinary tests do not publish anything.
