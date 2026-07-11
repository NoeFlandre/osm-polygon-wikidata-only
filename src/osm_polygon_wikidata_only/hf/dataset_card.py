"""Multi-table dataset card for the HF Hub dataset.

Produces a single ``README.md``-style card with YAML front matter, a
schema section for each parquet table, OSM/Wikidata/Wikipedia
attribution, and license info.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any


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
        "enriched with Wikidata descriptions and Wikipedia article "
        "text for every valid language-Wikipedia sitelink, with full text "
        "and no per-QID article cap. One PBF produces three parquet files "
        "in this Hub:\n\n"
        "- `polygons/<stem>.parquet` — one row per polygon\n"
        "- `articles/<stem>.parquet` — one row per unique Wikipedia article\n"
        "- `polygon_articles/<stem>.parquet` — many-to-many link table\n\n"
        "Optional additive text augmentation is published without replacing those tables:\n\n"
        "- `wikipedia/documents/<stem>.parquet` and `wikipedia/sections/<stem>.parquet`\n"
        "- `wikivoyage/documents/<stem>.parquet` and `wikivoyage/sections/<stem>.parquet`\n"
        "- `wikidata/facts/<stem>.parquet`\n\n"
        f"Generated on {today}.\n\n"
        f"Maintained by **{maintainer}**.\n\n"
        "## Coverage\n\n"
        "![Coverage Map](coverage_map.png)\n\n"
        f"{stats_block}\n"
        f"{schema_section}\n"
        "## Data sources & licenses\n\n"
        "- **OpenStreetMap** polygons: (c) OpenStreetMap contributors, "
        "licensed under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/).\n"
        "- **Wikidata** entity data: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).\n"
        "- **Wikipedia** article text: licensed under "
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) "
        "by the respective Wikipedia editors; attributed inline per article.\n\n"
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
        "  - polygons\n"
        "  - geospatial\n"
        "  - multilingual\n"
        "configs:\n"
        "  - config_name: polygons\n"
        "    data_files:\n"
        "      - split: polygons\n"
        "        path: polygons/*.parquet\n"
        "  - config_name: articles\n"
        "    data_files:\n"
        "      - split: articles\n"
        "        path: articles/*.parquet\n"
        "  - config_name: polygon_articles\n"
        "    data_files:\n"
        "      - split: polygon_articles\n"
        "        path: polygon_articles/*.parquet\n"
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
    parts.append(_render_table("articles", art_cols, art_desc))
    parts.append(_render_table("polygon_articles", link_cols, link_desc))
    return "\n".join(parts) + "\n"


def _render_table(name: str, cols: list[str], descriptions: Mapping[str, str]) -> str:
    lines = [f"### `{name}`", "", "| Column | Description |", "| --- | --- |"]
    for c in cols:
        lines.append(f"| `{c}` | {descriptions.get(c, '')} |")
    lines.append("")
    return "\n".join(lines)


__all__ = ["render_dataset_card"]
