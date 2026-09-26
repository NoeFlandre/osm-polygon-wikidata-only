"""Contract tests for the public MkDocs site and Pages workflow."""

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def test_pages_workflow_builds_strictly_and_deploys_with_least_privilege() -> None:
    workflow = (REPOSITORY / ".github/workflows/docs.yml").read_text(encoding="utf-8")

    assert "branches: [main]" in workflow
    assert "workflow_dispatch:" in workflow
    assert "contents: read" in workflow
    assert "pages: read" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "uv sync --frozen" in workflow
    assert "uv run mkdocs build --strict" in workflow
    assert "actions/upload-pages-artifact@56afc609e74202658d3ffba0e8f6dda462b719fa # v3" in workflow
    assert "actions/deploy-pages@d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e # v4" in workflow
    assert "path: site" in workflow


def test_ci_runs_the_canonical_full_quality_gate() -> None:
    """CI must execute the same complete gate documented for contributors."""
    workflow = (REPOSITORY / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "run: just quality-gauntlet" in workflow
    assert "run: just check" not in workflow
    assert "run: just coverage" not in workflow


def test_pull_requests_build_docs_but_only_main_deploys() -> None:
    workflow = (REPOSITORY / ".github/workflows/docs.yml").read_text(encoding="utf-8")

    assert "  pull_request:\n" in workflow
    deploy = workflow.split("  deploy:\n", maxsplit=1)[1]
    assert "if: github.event_name != 'pull_request'" in deploy
    assert workflow.count("if: github.event_name != 'pull_request'") == 3


def test_ci_gates_prs_through_one_aggregate_check_without_duplicate_jobs() -> None:
    workflow = (REPOSITORY / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    justfile = (REPOSITORY / "Justfile").read_text(encoding="utf-8")

    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in workflow
    assert "  preprocessing:\n" not in workflow
    aggregate = workflow.split("  all-green:\n", maxsplit=1)[1]
    assert "if: always()" in aggregate
    assert "needs: [quality, security, container]" in aggregate
    architecture = justfile.split("architecture-checks:", maxsplit=1)[1].split("\n\n")[0]
    for recipe in ("just build", "just docs", "just package-smoke", "just preprocessing-check"):
        assert recipe in architecture


def test_ci_audits_dependencies_and_runs_codeql() -> None:
    workflow = (REPOSITORY / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    justfile = (REPOSITORY / "Justfile").read_text(encoding="utf-8")
    codeql = (REPOSITORY / ".github/workflows/codeql.yml").read_text(encoding="utf-8")

    security = workflow.split("  security:\n", maxsplit=1)[1].split("\n\n")[0]
    assert "run: just audit" in security
    audit = justfile.split("\naudit:", maxsplit=1)[1].split("\n\n")[0]
    assert audit.count("pip-audit") == 2
    assert "--strict" in audit
    assert "--directory preprocessing" in audit
    assert "security-events: write" in codeql.split("jobs:", maxsplit=1)[1]
    assert "schedule:" in codeql
    for line in codeql.splitlines():
        if "uses:" in line:
            assert "@" in line and len(line.split("@")[1].split()[0]) == 40, line
