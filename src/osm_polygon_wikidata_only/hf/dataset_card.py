"""Multi-table dataset card for the HF Hub dataset.

Produces a single ``README.md``-style card with YAML front matter, a
schema section for every parquet table, OSM/Wikidata/Wikipedia
attribution, and license info. The augmentation schema descriptions
live in :mod:`osm_polygon_wikidata_only.augmentation.schema_descriptions`
which is the single source of truth referenced from here.

The ``Generated on`` line uses the current wall-clock date
(``datetime.now(UTC)``) at every invocation. Tests that need a
stable golden fixture MUST post-process the produced Markdown by
substituting the date with a stable placeholder
(``re.sub(r"Generated on \\d{4}-\\d{2}-\\d{2}\\.", "Generated on YYYY-MM-DD.", md)``).
No clock parameter is exposed on the public function: production
behaviour and the public function signature stay stable across
refactors.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from osm_polygon_wikidata_only.augmentation.schema import (
    DOCUMENT_COLUMNS,
    FACT_COLUMNS,
    SECTION_COLUMNS,
)
from osm_polygon_wikidata_only.augmentation.schema_descriptions import (
    DOCUMENT_DESCRIPTIONS,
    FACT_DESCRIPTIONS,
    SECTION_DESCRIPTIONS,
)
from osm_polygon_wikidata_only.augmentation.wikipedia_documents import (
    WIKIPEDIA_DOCUMENT_COLUMNS,
    WIKIPEDIA_DOCUMENT_DESCRIPTIONS,
)
from osm_polygon_wikidata_only.hf.repo_layout import (
    REMOTE_COVERAGE_MAP_FILE,
    REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE,
    REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE,
    REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE,
)


def render_dataset_card(
    *,
    repo_id: str,
    stats: Mapping[str, Any],
    polygon_columns: list[str],
    polygon_descriptions: Mapping[str, str],
    article_columns: list[str],
    article_descriptions: Mapping[str, str],
    link_columns: list[str],
    link_descriptions: Mapping[str, str],
    primary_lang: str = "en",
    maintainer: str = "Noé Flandre",
    stats_section: str | None = None,
) -> str:
    """Render the dataset card markdown.

    ``stats`` may include ``polygon_count``, ``article_count``,
    ``unique_wikidata_count``, etc.

    ``stats_section`` is an optional pre-rendered markdown block of
    factual dataset statistics (snapshot, funnel, language distribution).
    When provided, it is included verbatim after the coverage map.

    The YAML front matter declares the canonical dataset tables.

    The ``Generated on`` line uses the current UTC date at every
    invocation. Tests that compare against a golden fixture must
    post-process the rendered Markdown by substituting the date
    with a stable placeholder via regex; the production output
    itself always reflects the real publication date.
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    rc_lines = _render_front_matter(
        repo_id=repo_id,
        license="odbl",
        primary_lang=primary_lang,
        polygon_count=stats.get("polygon_count", 0),
        article_count=stats.get("article_count", 0),
        unique_wikidata_count=stats.get("unique_wikidata_count", 0),
    )

    schema_section = _render_schema(
        polygon_columns,
        polygon_descriptions,
        article_columns,
        article_descriptions,
        link_columns,
        link_descriptions,
    )

    stats_block = stats_section if stats_section is not None else ""

    body = (
        f"# {repo_id}\n\n"
        "OSM polygons tagged with a `wikidata=*` reference, "
        "enriched with Wikipedia and Wikivoyage text across all available languages. "
        "The published tables are:\n\n"
        "- `polygons/<stem>.parquet` — one row per polygon\n"
        "- `wikipedia/documents/<stem>.parquet` — one row per unique Wikipedia article\n"
        "- `polygon_articles/<stem>.parquet` — polygon-to-Wikipedia many-to-many links\n"
        "- `wikipedia/sections/<stem>.parquet`\n"
        "- `wikivoyage/documents/<stem>.parquet` and `wikivoyage/sections/<stem>.parquet`\n"
        "- `wikidata/facts/<stem>.parquet`\n\n"
        f"Generated on {today}.\n\n"
        f"Maintained by **{maintainer}**.\n\n"
        "## Coverage\n\n"
        "### Polygons with Wikipedia or Wikivoyage text\n\n"
        f"![Polygons with Wikipedia or Wikivoyage text]({REMOTE_GEOGRAPHIC_TEXT_PRESENCE_FILE})\n\n"
        "Each point is a dataset polygon with at least one non-empty Wikipedia document "
        "or a non-empty Wikivoyage document sharing its Wikidata entity. A polygon is "
        "shown once even when several documents qualify.\n\n"
        "### All dataset polygons\n\n"
        f"![Coverage Map]({REMOTE_COVERAGE_MAP_FILE})\n\n"
        "## Geographic coverage\n\n"
        "Both maps below aggregate dataset polygons into H3 cells at the "
        "same resolution. All denominators and counts are conditional on "
        "each polygon carrying an OSM `wikidata=*` tag.\n\n"
        "### Wikipedia text coverage\n\n"
        f"![Geographic Wikipedia Text Coverage]({REMOTE_GEOGRAPHIC_TEXT_COVERAGE_FILE})\n\n"
        "`coverage_rate(h) = covered_polygons(h) / all_dataset_polygons(h)`, "
        "where a covered polygon has at least one linked Wikipedia article "
        "with non-empty text. Cell colour encodes this fraction from 0% to "
        "100%; grey cells hold fewer than 20 polygons and are not "
        "statistically meaningful.\n\n"
        "### Polygon density\n\n"
        f"![Geographic Polygon Density]({REMOTE_GEOGRAPHIC_POLYGON_COUNT_FILE})\n\n"
        "`polygon_count(h) = number of dataset polygons whose centroid "
        "belongs to H3 cell h`. Colour encodes the raw count on a "
        "logarithmic scale because counts are highly skewed across the world. "
        "Low counts remain visible.\n\n"
        f"{stats_block}\n"
        f"{schema_section}\n"
        "## Data sources & licenses\n\n"
        "- **OpenStreetMap** polygons: (c) OpenStreetMap contributors, "
        "licensed under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/).\n"
        "- **Wikidata** entity data: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).\n"
        "- **Wikipedia** article text: licensed under "
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) "
        "by the respective Wikipedia editors; attributed inline per article.\n"
        "- **Wikivoyage** text: licensed under "
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) "
        "by the respective Wikivoyage editors; attributed inline per document.\n"
        "- **Natural Earth** Admin-0 geography: public-domain 1:110m reference data "
        "used only to assign centroid-based continent statistics and draw context maps.\n\n"
        "## How to load\n\n"
        "```python\n"
        "from datasets import load_dataset\n"
        'ds = load_dataset("parquet", data_files={\n'
        f'    "polygons": "hf://datasets/{repo_id}/polygons/*.parquet",\n'
        "})\n"
        "```\n"
    )

    return rc_lines + "\n" + body


def _render_front_matter(
    *,
    repo_id: str,
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


def _render_schema(
    poly_cols: list[str],
    poly_desc: Mapping[str, str],
    art_cols: list[str],
    art_desc: Mapping[str, str],
    link_cols: list[str],
    link_desc: Mapping[str, str],
) -> str:
    parts = ["## Schema\n"]
    parts.append(_render_table("polygons", poly_cols, poly_desc))
    parts.append(
        _render_table(
            "wikipedia/documents",
            list(WIKIPEDIA_DOCUMENT_COLUMNS),
            WIKIPEDIA_DOCUMENT_DESCRIPTIONS,
        )
    )
    parts.append(_render_table("polygon_articles", link_cols, link_desc))
    parts.append(
        _render_combined_table(
            "`wikivoyage/documents`",
            list(DOCUMENT_COLUMNS),
            DOCUMENT_DESCRIPTIONS,
        )
    )
    parts.append(
        _render_combined_table(
            "`wikipedia/sections` and `wikivoyage/sections`",
            list(SECTION_COLUMNS),
            SECTION_DESCRIPTIONS,
        )
    )
    parts.append(_render_combined_table("`wikidata/facts`", list(FACT_COLUMNS), FACT_DESCRIPTIONS))
    return "\n".join(parts) + "\n"


def _render_table(name: str, cols: list[str], descriptions: Mapping[str, str]) -> str:
    lines = [f"### `{name}`", "", "| Column | Description |", "| --- | --- |"]
    for c in cols:
        lines.append(f"| `{c}` | {descriptions.get(c, '')} |")
    lines.append("")
    return "\n".join(lines)


def _render_combined_table(heading: str, cols: list[str], descriptions: Mapping[str, str]) -> str:
    lines = [f"### {heading}", "", "| Column | Description |", "| --- | --- |"]
    for c in cols:
        lines.append(f"| `{c}` | {descriptions.get(c, '')} |")
    lines.append("")
    return "\n".join(lines)


__all__ = ["render_dataset_card"]


# ---------------------------------------------------------------------------
# Front-matter structural validation
# ---------------------------------------------------------------------------
#
# ``validate_front_matter`` is a TEST-only structural helper. It is
# imported directly by the dataset-card test suite via the module path;
# it deliberately does NOT appear in :data:`__all__` and is not
# re-exported by :mod:`osm_polygon_wikidata_only.hf` or by the
# dataset-card facade. The Phase 1 frozen public surface is exactly
# ``{"render_dataset_card"}``.


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
    import yaml

    # ``safe_load_all`` accepts the conventional ``---\n...\n---\n``
    # envelope produced by :func:`render_dataset_card`. The first
    # yielded document is the canonical front-matter mapping; trailing
    # ``None`` entries (introduced by PyYAML's trailing whitespace
    # handling) are ignored.
    docs = [d for d in yaml.safe_load_all(front_matter) if d is not None]
    if not docs:
        raise ValueError("Front matter must deserialize to a YAML document")
    parsed = docs[0]
    if not isinstance(parsed, dict):
        raise ValueError("Front matter must deserialize to a mapping")
    configs = parsed.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("Front matter must declare a non-empty `configs:` list")
    for entry in configs:
        if not isinstance(entry, dict):
            raise ValueError("Each `configs:` entry must be a mapping")
        if "config_name" not in entry:
            raise ValueError("Each `configs:` entry must contain `config_name`")
        if "data_files" not in entry:
            raise ValueError(f"configs entry {entry.get('config_name')!r} is missing `data_files`")
        data_files = entry["data_files"]
        files_iter = data_files if isinstance(data_files, list) else [data_files]
        seen_paths = False
        for file_block in files_iter:
            if not isinstance(file_block, dict):
                raise ValueError("`data_files:` block must be a mapping")
            if "path" in file_block:
                seen_paths = True
        if not seen_paths:
            raise ValueError(f"configs entry {entry['config_name']!r} has no `path:` glob")
