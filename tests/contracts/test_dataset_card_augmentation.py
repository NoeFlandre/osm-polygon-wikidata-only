"""Frozen augmentation schema + YAML front matter checks for the dataset card.

This module documents the schema of the augmentation tables and the
Hugging Face dataset-card configurations. It is a pure-extension
contract test; no Parquet reads, no network.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from osm_polygon_wikidata_only.hf.dataset_card import render_front_matter


def _render_front_matter_block() -> str:
    return render_front_matter(
        repo_id="NoeFlandre/osm-polygon-wikidata-only",
        license="odbl",
        primary_lang="en",
        polygon_count=1,
        article_count=1,
        unique_wikidata_count=1,
    )


# --- augmentation YAML front matter --------------------------------


def test_dataset_card_yaml_lists_all_canonical_configurations() -> None:
    md = _render_front_matter_block()
    for name in (
        "polygons",
        "polygon_articles",
        "wikipedia_documents",
        "wikipedia_sections",
        "wikivoyage_documents",
        "wikivoyage_sections",
        "wikidata_facts",
    ):
        assert f"config_name: {name}" in md, f"missing {name} in YAML front matter"


def test_dataset_card_yaml_paths_match_repo_layout() -> None:
    """Each config's ``path:`` glob must match the published sidecar path."""
    md = _render_front_matter_block()
    expected_globs = (
        "polygons/*.parquet",
        "polygon_articles/*.parquet",
        "wikipedia/documents/*.parquet",
        "wikipedia/sections/*.parquet",
        "wikivoyage/documents/*.parquet",
        "wikivoyage/sections/*.parquet",
        "wikidata/facts/*.parquet",
    )
    for glob in expected_globs:
        assert f"path: {glob}" in md, f"missing glob {glob} in YAML front matter"


def test_dataset_card_yaml_omits_retired_articles_configuration() -> None:
    md = _render_front_matter_block()
    assert "config_name: polygons" in md
    assert "config_name: polygon_articles" in md
    assert "config_name: articles" not in md
    assert "path: articles/*.parquet" not in md


def test_dataset_card_yaml_contains_wikivoyage_tag() -> None:
    md = _render_front_matter_block()
    assert "  - wikivoyage" in md


def test_dataset_card_yaml_starts_with_valid_front_matter() -> None:
    md = _render_front_matter_block()
    assert md.startswith("---\n"), "dataset card must start with YAML open delimiter"
    end_marker = "\n---\n"
    end_index = md.find(end_marker, 4)
    assert end_index > 0, "dataset card must close its YAML front matter"
    header_block = md[: end_index + len(end_marker)]
    lines = [line for line in header_block.splitlines() if line and not line.startswith("---")]
    for line in lines:
        # Accept `key: value` lines and bare `- item` list items.
        assert ":" in line or line.lstrip().startswith("- "), f"Malformed YAML line: {line!r}"


def test_dataset_card_front_matter_passes_structural_validator() -> None:
    """The YAML front matter must satisfy the structural validator.

    This is a stronger check than the line-shape assertion above:
    :func:`validate_front_matter` deserializes the block via PyYAML
    and walks the ``configs:`` list to confirm ``config_name``,
    ``data_files``, and at least one path glob per entry. Catches
    malformed globs, missing fields, or broken indentation that a
    line-shape check would miss.
    """
    from osm_polygon_wikidata_only.hf.dataset_card import validate_front_matter

    md = _render_front_matter_block()
    end_marker = "\n---\n"
    end_index = md.find(end_marker, 4)
    header_block = md[: end_index + len(end_marker)]
    validate_front_matter(header_block)


@pytest.mark.parametrize(
    ("front_matter", "message"),
    [
        ("", "must deserialize"),
        ("---\n- item\n---\n", "must deserialize to a mapping"),
        ("---\nconfigs: {}\n---\n", "non-empty"),
        ("---\nconfigs: []\n---\n", "non-empty"),
        ("---\nconfigs:\n  - item\n---\n", "entry must be a mapping"),
        ("---\nconfigs:\n  - data_files: []\n---\n", "contain `config_name`"),
        (
            "---\nconfigs:\n  - config_name: polygons\n---\n",
            "missing `data_files`",
        ),
        (
            "---\nconfigs:\n  - config_name: polygons\n    data_files: [item]\n---\n",
            "block must be a mapping",
        ),
        (
            "---\nconfigs:\n  - config_name: polygons\n    data_files:\n      - split: polygons\n---\n",
            "has no `path:`",
        ),
    ],
)
def test_dataset_card_front_matter_validator_rejects_malformed_shapes(
    front_matter: str, message: str
) -> None:
    from osm_polygon_wikidata_only.hf.dataset_card import validate_front_matter

    with pytest.raises(ValueError, match=message):
        validate_front_matter(front_matter)


# --- augmentation schema sections ---------------------------------


def _column_names(schema: pa.Schema) -> tuple[str, ...]:
    return tuple(field.name for field in schema)


# --- README sections ordering ---------------------------------------
