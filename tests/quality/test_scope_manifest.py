from pathlib import Path

import pytest

from scripts.quality.scope_manifest import NER_SCOPE, validate_scope

ROOT = Path(__file__).resolve().parents[2]


def test_ner_scope_is_the_committed_pilot_boundary() -> None:
    assert NER_SCOPE.source_paths == (
        "src/osm_polygon_wikidata_only/ner/job.py",
        "src/osm_polygon_wikidata_only/ner/otter.py",
        "src/osm_polygon_wikidata_only/ner/pipeline.py",
        "src/osm_polygon_wikidata_only/ner/publication.py",
        "src/osm_polygon_wikidata_only/grid5000/ner_controller.py",
        "scripts/grid5000_geographic_ner.py",
        "scripts/prepare_geographic_ner_pilot.py",
    )
    assert NER_SCOPE.test_paths == (
        "tests/ner/test_job.py",
        "tests/ner/test_otter.py",
        "tests/ner/test_pipeline.py",
        "tests/ner/test_publication.py",
        "tests/grid5000/test_ner_controller.py",
        "tests/ner/test_pilot.py",
        "tests/ner/test_utf8_metadata.py",
    )


def test_validate_scope_rejects_duplicate_paths() -> None:
    duplicate = NER_SCOPE.__class__(
        name="duplicate",
        source_paths=("src/a.py", "src/a.py"),
        test_paths=("tests/a.py",),
    )

    with pytest.raises(ValueError, match="duplicate source path"):
        validate_scope(duplicate, ROOT)


def test_validate_scope_rejects_missing_paths() -> None:
    missing = NER_SCOPE.__class__(
        name="missing",
        source_paths=("src/missing.py",),
        test_paths=("tests/missing.py",),
    )

    with pytest.raises(ValueError, match="does not exist"):
        validate_scope(missing, ROOT)


def test_validate_scope_rejects_source_test_overlap() -> None:
    overlap = NER_SCOPE.__class__(
        name="overlap",
        source_paths=(NER_SCOPE.source_paths[0],),
        test_paths=(NER_SCOPE.source_paths[0],),
    )

    with pytest.raises(ValueError, match="both source and test"):
        validate_scope(overlap, ROOT)
