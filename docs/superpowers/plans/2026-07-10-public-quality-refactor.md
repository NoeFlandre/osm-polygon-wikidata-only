# Public Quality Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the codebase into focused typed modules and make the repository public-ready without changing any observable pipeline behavior.

**Architecture:** Characterization tests freeze supported imports and outputs before responsibilities move. Existing facade modules re-export stable classes/functions while internal Wikipedia, Wikidata, pipeline, and CLI packages own one concern each. Documentation, metadata, CI, and coverage enforce the same local quality contract.

**Tech Stack:** Python 3.12, pytest/pytest-cov, Ruff, strict mypy, Hatchling, uv, GitHub Actions.

---

### Task 1: Freeze compatibility and behavioral contracts

**Files:**
- Create: `tests/test_public_api.py`
- Modify: `tests/test_enrichment.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_cli.py`

- [ ] Add a failing import-identity test that imports documented names from the current facades and the future focused modules, then asserts the objects are identical:

```python
def test_wikipedia_facade_preserves_public_types() -> None:
    from osm_polygon_wikidata_only.enrichment.wikipedia.models import FetchResult as Focused
    from osm_polygon_wikidata_only.enrichment.wikipedia_client import FetchResult as Facade
    assert Facade is Focused
```

- [ ] Add characterization tests for URL parameters, cache keys, exact-revision fallback, row ordering, manifest contents, parser defaults, and upload commit ordering using existing in-memory fixtures.
- [ ] Run `uv run pytest tests/test_public_api.py tests/test_enrichment.py tests/test_pipeline.py tests/test_cli.py -q`; expect failure because focused packages do not exist.
- [ ] Commit tests only after observing the expected import failure: `test: freeze public compatibility contracts`.

### Task 2: Extract Wikipedia models and pure parsing

**Files:**
- Create: `src/osm_polygon_wikidata_only/enrichment/wikipedia/__init__.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/wikipedia/models.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/wikipedia/parsing.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_enrichment.py`

- [ ] Move `WikipediaArticle`, `FetchResult`, `WikipediaClient`, and `BatchWikipediaClient` unchanged into `models.py`; give each contract-focused docstrings and `__all__`.
- [ ] Move response parsing, batch mapping, exact-revision extraction, parse-response text extraction, and query-copy helpers unchanged into `parsing.py`.
- [ ] Make the facade import and re-export the moved names. Do not duplicate class definitions:

```python
from .wikipedia.models import BatchWikipediaClient, FetchResult, WikipediaArticle, WikipediaClient
from .wikipedia.parsing import parse_wikipedia_response
```

- [ ] Run focused parser, fallback, and public-identity tests; expect all green with byte-equivalent articles.
- [ ] Run `uv run mypy src` and commit `refactor: extract Wikipedia models and parsing`.

### Task 3: Extract Wikipedia transport and cache adapter

**Files:**
- Create: `src/osm_polygon_wikidata_only/enrichment/wikipedia/transport.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/wikipedia/cache.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/wikipedia/client.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikipedia_client.py`
- Modify: `tests/test_enrichment.py`

- [ ] Add a failing transport test using a fake opener/scheduler that asserts headers, gzip decoding, 429/503 cooldown, and JSON validation without network access.
- [ ] Extract URL building and scheduled JSON GET into a typed `MediaWikiTransport`; keep all request parameters unchanged.
- [ ] Move article serialization, versioned key construction, and `CachedWikipediaClient` to `cache.py`.
- [ ] Move HTTP/in-memory client orchestration and exact-revision fallback to `client.py`, composing transport and pure parsers.
- [ ] Reduce `wikipedia_client.py` to documented compatibility imports and `__all__`.
- [ ] Run all enrichment tests, strict mypy, and Ruff; commit `refactor: separate Wikipedia transport and cache`.

### Task 4: Extract Wikidata responsibilities

**Files:**
- Create: `src/osm_polygon_wikidata_only/enrichment/wikidata/__init__.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/wikidata/models.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/wikidata/parsing.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/wikidata/transport.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/wikidata/cache.py`
- Create: `src/osm_polygon_wikidata_only/enrichment/wikidata/client.py`
- Modify: `src/osm_polygon_wikidata_only/enrichment/wikidata_client.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_enrichment.py`

- [ ] Add failing facade-identity and transport/cache characterization tests.
- [ ] Move entity/protocol types, pure QID/sitelink parsing, HTTP mechanics, cache serialization, and client composition into the matching focused modules without altering strings or status semantics.
- [ ] Make `wikidata_client.py` a compatibility facade with the established `__all__`.
- [ ] Run enrichment/public API tests, mypy, and Ruff; commit `refactor: separate Wikidata client responsibilities`.

### Task 5: Split single-PBF processing responsibilities

**Files:**
- Create: `src/osm_polygon_wikidata_only/pipeline/rows.py`
- Create: `src/osm_polygon_wikidata_only/pipeline/completeness.py`
- Create: `src/osm_polygon_wikidata_only/pipeline/publication.py`
- Modify: `src/osm_polygon_wikidata_only/pipeline/processor.py`
- Create: `tests/pipeline/test_rows.py`
- Create: `tests/pipeline/test_completeness.py`
- Create: `tests/pipeline/test_publication.py`
- Modify: `tests/test_public_api.py`

- [ ] Add focused failing tests for deterministic row construction, unresolved-site diagnostics, temporary-file cleanup, atomic promotion, and manifest-after-Parquet ordering.
- [ ] Move row copying/conversion into `rows.py`, `IncompleteEnrichmentError` and audit logic into `completeness.py`, and local publication/manifest mutation into `publication.py`.
- [ ] Keep `PbfStem`, `ProcessResult`, `process_pbf`, and compatibility re-exports in `processor.py`; reduce it to stage sequencing.
- [ ] Run pipeline, schema, parquet, manifest, and end-to-end tests; commit `refactor: split pipeline processing stages`.

### Task 6: Split CLI composition

**Files:**
- Create: `src/osm_polygon_wikidata_only/cli/parser.py`
- Create: `src/osm_polygon_wikidata_only/cli/dependencies.py`
- Create: `src/osm_polygon_wikidata_only/cli/publication.py`
- Modify: `src/osm_polygon_wikidata_only/cli/commands.py`
- Create: `tests/cli/test_parser.py`
- Create: `tests/cli/test_dependencies.py`
- Create: `tests/cli/test_publication.py`
- Modify: `tests/test_public_api.py`

- [ ] Add failing tests for parser identity/defaults, shared scheduler injection, cache paths, card snapshots, upload file order, queue resume, and final exit status.
- [ ] Extract argparse construction/language parsing, dependency composition, and upload snapshot/queue composition into their focused modules.
- [ ] Keep `build_parser`, `_build_settings`, and `main` compatible through `commands.py`; keep direct composition and avoid a container framework.
- [ ] Run CLI/HF/end-to-end tests and commit `refactor: separate CLI composition concerns`.

### Task 7: Public package metadata and documentation

**Files:**
- Modify: `pyproject.toml`
- Create: `src/osm_polygon_wikidata_only/py.typed`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Create: `docs/development.md`
- Create: `docs/api.md`
- Create: `CONTRIBUTING.md`
- Create: `SECURITY.md`
- Create: `tests/test_packaging.py`

- [ ] Add a failing metadata/wheel test that checks author `Noé Flandre`, license, URLs, classifiers, `py.typed`, and the console script.
- [ ] Complete PEP 621 metadata using the existing MIT `LICENSE`, repository/Hugging Face URLs, and Python 3.12 classifier; configure Hatchling to include `py.typed`.
- [ ] Rewrite and cross-link public docs around guarantees and actual supported workflows; document only stable API imports.
- [ ] Build sdist/wheel and inspect archives in the packaging test; commit `docs: make project public contributor ready`.

### Task 8: CI and coverage quality gate

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: tests for uncovered critical paths only.

- [ ] Add tests for CLI lifecycle, retry exhaustion, rate-limit parsing, PBF reader callbacks, and logging where they assert meaningful contracts.
- [ ] Raise `fail_under` from 60 to 80 only after `uv run pytest --cov` proves at least 80% branch-aware coverage.
- [ ] Add CI steps for `uv sync --frozen`, coverage tests, Ruff lint/format, strict mypy, and `uv build` on Python 3.12.
- [ ] Run the exact CI sequence locally and commit `ci: enforce public quality gate`.

### Task 9: Final compatibility and integration

**Files:**
- Verify all source, tests, docs, metadata, and workflow files.

- [ ] Run `uv run pytest --cov=osm_polygon_wikidata_only --cov-report=term-missing -q`; require 80% or higher and zero failures.
- [ ] Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`, and `uv build`; require success.
- [ ] Inspect wheel/sdist contents and run `uv run osm-polygon-wikidata-only --help` from the built installation environment.
- [ ] Compare supported facade imports, schema constants, CLI help snapshot, and deterministic fixture outputs with the baseline.
- [ ] Use the finishing-development-branch workflow, merge `codex/public-quality-refactor` into `main`, rerun the complete gate on `main`, then remove the worktree and feature branch.
