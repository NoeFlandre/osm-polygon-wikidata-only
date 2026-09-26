"""YAML front matter for the V1 dataset card.

The card body is rendered by
:mod:`osm_polygon_wikidata_only.hf.minimal_card`; this module owns only the
front matter, which declares the Dataset Viewer configurations and the
``dataset_info`` counters the Hub reads.

``validate_front_matter`` is a structural test seam. It is deliberately absent
from :data:`__all__` and is not re-exported by the package facade.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["render_front_matter"]


def render_front_matter(
    *,
    repo_id: str,  # noqa: ARG001 -- public keyword kept for API compatibility
    license: str,
    primary_lang: str,
    polygon_count: int,
    article_count: int,
    unique_wikidata_count: int,
) -> str:
    return (
        "---\n"
        "license: " + license + "\n"
        "language:\n"
        f"  - {primary_lang}\n"
        "tags:\n"
        "  - openstreetmap\n"
        "  - wikidata\n"
        "  - wikipedia\n"
        "  - wikivoyage\n"
        "  - polygons\n"
        "  - geospatial\n"
        "  - multilingual\n"
        "configs:\n"
        "  - config_name: polygons\n"
        "    data_files:\n"
        "      - split: polygons\n"
        "        path: polygons/*.parquet\n"
        "  - config_name: polygon_articles\n"
        "    data_files:\n"
        "      - split: polygon_articles\n"
        "        path: polygon_articles/*.parquet\n"
        "  - config_name: wikipedia_documents\n"
        "    data_files:\n"
        "      - split: wikipedia_documents\n"
        "        path: wikipedia/documents/*.parquet\n"
        "  - config_name: wikipedia_sections\n"
        "    data_files:\n"
        "      - split: wikipedia_sections\n"
        "        path: wikipedia/sections/*.parquet\n"
        "  - config_name: wikivoyage_documents\n"
        "    data_files:\n"
        "      - split: wikivoyage_documents\n"
        "        path: wikivoyage/documents/*.parquet\n"
        "  - config_name: wikivoyage_sections\n"
        "    data_files:\n"
        "      - split: wikivoyage_sections\n"
        "        path: wikivoyage/sections/*.parquet\n"
        "  - config_name: wikidata_facts\n"
        "    data_files:\n"
        "      - split: wikidata_facts\n"
        "        path: wikidata/facts/*.parquet\n"
        "dataset_info:\n"
        f"  polygon_count: {polygon_count}\n"
        f"  unique_wikidata_count: {unique_wikidata_count}\n"
        f"  article_count: {article_count}\n"
        "---\n"
    )


def validate_front_matter(front_matter: str) -> None:
    """Validate the structural shape of the dataset-card YAML front matter.

    The Hugging Face dataset card expects a top-level YAML mapping
    with a ``configs:`` sequence of well-formed objects. We check the
    shape concretely:

    * ``configs`` is a non-empty list.
    * Each entry contains the strings ``config_name``, ``data_files``.
    * Each entry has at least one path glob inside ``path:``.

    This is intentionally non-generic: we want to catch dangling
    entries, missing ``config_name`` fields, or glob typos before
    the card reaches the HF Hub.
    """
    parsed = _parse_front_matter(front_matter)
    configs = parsed.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("Front matter must declare a non-empty `configs:` list")
    for entry in configs:
        _validate_config_entry(entry)


def _parse_front_matter(front_matter: str) -> Mapping[str, Any]:
    """Deserialize the first non-empty YAML document as a mapping."""
    # PyYAML is a card-validation dependency; keep it out of core data imports.
    import yaml  # noqa: PLC0415

    # ``safe_load_all`` accepts the conventional ``---\n...\n---\n``
    # envelope produced by :func:`render_front_matter`. The first
    # yielded document is the canonical front-matter mapping; trailing
    # ``None`` entries (introduced by PyYAML's trailing whitespace
    # handling) are ignored.
    docs = [document for document in yaml.safe_load_all(front_matter) if document is not None]
    if not docs:
        raise ValueError("Front matter must deserialize to a YAML document")
    parsed = docs[0]
    if not isinstance(parsed, dict):
        raise ValueError("Front matter must deserialize to a mapping")
    return parsed


def _validate_config_entry(entry: Any) -> None:
    """Validate one HF config entry and its path declarations."""
    if not isinstance(entry, dict):
        raise ValueError("Each `configs:` entry must be a mapping")
    config_name, data_files = _required_config_fields(entry)
    if not any("path" in block for block in _data_file_blocks(data_files)):
        raise ValueError(f"configs entry {config_name!r} has no `path:` glob")


def _required_config_fields(entry: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return required config fields, raising the original diagnostics."""
    if "config_name" not in entry:
        raise ValueError("Each `configs:` entry must contain `config_name`")
    if "data_files" not in entry:
        raise ValueError(f"configs entry {entry.get('config_name')!r} is missing `data_files`")
    return entry["config_name"], entry["data_files"]


def _data_file_blocks(data_files: Any) -> list[Mapping[str, Any]]:
    """Normalize one config's data-file declaration and validate its blocks."""
    files_iter = data_files if isinstance(data_files, list) else [data_files]
    blocks: list[Mapping[str, Any]] = []
    for file_block in files_iter:
        if not isinstance(file_block, dict):
            raise ValueError("`data_files:` block must be a mapping")
        blocks.append(file_block)
    return blocks
