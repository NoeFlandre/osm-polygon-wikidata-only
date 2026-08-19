"""Read the current local dataset snapshot for the presentation builder."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    """Small, public-safe view of the generated dataset card statistics."""

    generated_on: str
    active_regions: int
    raw_inputs: int
    retired_inputs: int
    polygons: str
    unique_wikidata: str
    wikipedia_documents: str
    wikipedia_sections: str
    wikivoyage_documents: str
    wikivoyage_sections: str
    wikidata_facts: str
    document_words: str
    total_parquet_size: str
    text_polygons: str
    text_polygon_rate: str
    languages: str
    top_five_language_share: str
    top_twenty_language_share: str
    top_language_rows: tuple[tuple[str, str, str], ...]
    continents: tuple[tuple[str, str, str, str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe values for the generated audit snapshot."""
        return asdict(self)


def _data_root() -> Path:
    value = os.environ.get("OSM_POLYGON_DATA_ROOT")
    if not value:
        raise RuntimeError("Set OSM_POLYGON_DATA_ROOT before building the decks")
    root = Path(value).expanduser()
    if not root.is_dir():
        raise RuntimeError(f"OSM_POLYGON_DATA_ROOT is not a directory: {root}")
    return root


def _section(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    try:
        start = lines.index(heading) + 1
    except ValueError as exc:
        raise RuntimeError(f"Missing generated-card section: {heading}") from exc
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## ") and lines[index] != heading:
            end = index
            break
    return "\n".join(lines[start:end])


def _table(markdown_section: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in markdown_section.splitlines():
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 2 and cells[0] != "Metric":
            values[cells[0]] = cells[1]
    return values


def _number(value: str) -> int:
    return int(value.replace(",", "").split()[0])


def _continent_rows(markdown: str) -> tuple[tuple[str, str, str, str, str], ...]:
    section = _section(markdown, "## Geographic distribution by continent")
    rows: list[tuple[str, str, str, str, str]] = []
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("| ---") or line.startswith("| Continent"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 7:
            rows.append((cells[0], cells[1], cells[4], cells[5], cells[6]))
    if len(rows) != 8:
        raise RuntimeError(f"Expected eight continent rows, found {len(rows)}")
    return tuple(rows)


def _language_rows(markdown: str) -> tuple[tuple[str, str, str], ...]:
    section = _section(markdown, "## Language distribution")
    rows: list[tuple[str, str, str]] = []
    in_table = False
    for line in section.splitlines():
        if line.startswith("| Language"):
            in_table = True
            continue
        if not in_table or line.startswith("| ---"):
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 4:
            rows.append((cells[0], cells[1], cells[2]))
    if len(rows) < 8:
        raise RuntimeError("Generated card did not contain the top language table")
    return tuple(rows[:8])


def _concentration_share(markdown: str, rank: int) -> str:
    pattern = rf"Top {rank} languages?:\s*([0-9]+(?:\.[0-9]+)?%)"
    match = re.search(pattern, markdown)
    if match is None:
        raise RuntimeError(f"Generated card has no top-{rank} language concentration")
    return match.group(1)


def read_snapshot() -> DatasetSnapshot:
    """Read and cross-check the latest generated card and manifest."""
    root = _data_root()
    manifest_path = root / "processed" / "manifests" / "processed_pbfs.json"
    card_path = root / "cache" / "metadata_upload_snapshots" / "README.md"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing processed manifest: {manifest_path}")
    if not card_path.is_file():
        raise RuntimeError(f"Missing generated dataset card snapshot: {card_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("Processed manifest must be a JSON object")
    card = card_path.read_text(encoding="utf-8")
    generated_match = re.search(r"Generated on (\d{4}-\d{2}-\d{2})\.", card)
    if generated_match is None:
        raise RuntimeError("Generated card has no snapshot date")

    snapshot = _table(_section(card, "## Dataset snapshot"))
    polygons = _number(snapshot["Polygons"])
    manifest_polygons = sum(int(entry.get("polygon_count", 0)) for entry in manifest.values())
    if polygons != manifest_polygons:
        raise RuntimeError(
            f"Card/manifest mismatch for polygons: card={polygons}, manifest={manifest_polygons}"
        )

    raw_inputs = len(list((root / "raw").glob("*.osm.pbf")))
    retired_path = root / "processed" / "manifests" / "containment_retirements.json"
    retired_inputs = 0
    if retired_path.is_file():
        retired = json.loads(retired_path.read_text(encoding="utf-8"))
        if isinstance(retired, dict):
            retired_entries = retired.get("retired", retired.get("retirements", []))
            retired_inputs = (
                len(retired_entries) if isinstance(retired_entries, (dict, list)) else 0
            )

    continent_rows = _continent_rows(card)
    text_polygon_count = sum(_number(row[3]) for row in continent_rows)
    text_polygons = f"{text_polygon_count:,}"
    text_polygon_rate = f"{100 * text_polygon_count / polygons:.1f}%"

    return DatasetSnapshot(
        generated_on=generated_match.group(1),
        active_regions=len(manifest),
        raw_inputs=raw_inputs,
        retired_inputs=retired_inputs,
        polygons=snapshot["Polygons"],
        unique_wikidata=snapshot["Unique Wikidata entities"],
        wikipedia_documents=snapshot["Wikipedia documents"],
        wikipedia_sections=snapshot["Wikipedia sections"],
        wikivoyage_documents=snapshot["Wikivoyage documents"],
        wikivoyage_sections=snapshot["Wikivoyage sections"],
        wikidata_facts=snapshot["Wikidata facts"],
        document_words=snapshot["Wikipedia + Wikivoyage document words"],
        total_parquet_size=snapshot["Total Parquet size"],
        text_polygons=text_polygons,
        text_polygon_rate=text_polygon_rate,
        languages=snapshot["Wikipedia + Wikivoyage languages"],
        top_five_language_share=_concentration_share(card, 5),
        top_twenty_language_share=_concentration_share(card, 20),
        top_language_rows=_language_rows(card),
        continents=continent_rows,
    )
