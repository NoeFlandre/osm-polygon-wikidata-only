"""Deterministic V2 card rendering primitives."""

from __future__ import annotations

from pathlib import Path

from osm_polygon_wikidata_only.hf._dataset_stats.rendering import demote_headings
from osm_polygon_wikidata_only.hf.polygon_geometry_stats import render_polygon_stats_section
from osm_polygon_wikidata_only.v2.card_metrics import compute_v2_card_stats
from osm_polygon_wikidata_only.v2.card_models import (
    SentenceCardStats as _SentenceCardStats,
)
from osm_polygon_wikidata_only.v2.card_models import (
    V2CardStats,
)
from osm_polygon_wikidata_only.v2.config import (
    V1_DATASET_URL,
    V2_ADDED_WIKIPEDIA_TAG_MAP_PATH,
    V2_CONTRACT_VERSION,
    V2_DATASET_CARD_VERSION,
    V2_GITHUB_URL,
    V2_REPO_ID,
    V2_TRACKIO_RUN_NAME,
    V2_TRACKIO_SPACE_ID,
)


def render_v2_card(
    processed_v2: Path,
    *,
    v1_processed: Path | None = None,
    stats: V2CardStats | None = None,
    generated_on: str | None = None,
) -> str:
    """Render a concise, viewer-compatible card from V2 files on disk."""
    snapshot = stats or compute_v2_card_stats(processed_v2, v1_processed=v1_processed)
    front_matter = _render_front_matter(snapshot, processed_v2=processed_v2)
    comparison = _render_comparison(snapshot)
    generated_block = [f"Generated on {generated_on}.", ""] if generated_on else []
    return (
        front_matter
        + "\n"
        + "\n".join(
            [
                "![NoeFlandre/osm-polygon-wikidata-and-wikipedia dataset overview](assets/dataset_hero.png)",
                "",
                "# OSM Polygon Wikidata + Wikipedia, V2",
                "",
                f"V2 builds on the [V1 Wikidata-only dataset]({V1_DATASET_URL}), which retained "
                f"OSM polygons carrying `wikidata=*` and enriched them with multilingual "
                f"Wikipedia and Wikivoyage text; V2 adds valid multilingual `wikipedia=*` "
                f"references, including polygons without a Wikidata QID. "
                f"The code is maintained in the [GitHub repository]({V2_GITHUB_URL}).",
                "",
                *generated_block,
                f"The public V2 Trackio snapshot is [`{V2_TRACKIO_RUN_NAME}`](https://huggingface.co/spaces/{V2_TRACKIO_SPACE_ID}).",
                "",
                "## Dataset snapshot",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Polygon rows across regional extracts | {snapshot.polygons:,} |",
                f"| Unique polygon identities (osm_type, osm_id) | {_unique_polygon_count(snapshot):,} |",
                f"| Polygons with successful non-empty text (unique OSM identities) | {_non_empty_text_polygon_count(snapshot):,} |",
                f"| Wikipedia documents | {snapshot.wikipedia_documents:,} |",
                f"| Wikipedia sections | {snapshot.wikipedia_sections:,} |",
                f"| Wikipedia + Wikivoyage languages | {snapshot.languages:,} |",
                f"| Geographic regions | {snapshot.regions:,} |",
                f"| Total Parquet size | {snapshot.total_parquet_storage_bytes / 1_000_000_000:.1f} GB |",
                "",
                "Polygon rows retain regional copies. Identity metrics use one deterministic representative per `(osm_type, osm_id)`; successful text additionally requires `fetch_status=ok` and trimmed non-empty `full_text`.",
                "",
                "<details>",
                "<summary>Full snapshot</summary>",
                "",
                f"- **Hugging Face dataset:** [{V2_REPO_ID}](https://huggingface.co/datasets/{V2_REPO_ID})",
                f"- **Unique Wikidata entities:** {snapshot.unique_wikidata_entities:,}",
                f"- **Wikivoyage documents:** {snapshot.wikivoyage_documents:,}",
                f"- **Wikipedia + Wikivoyage documents:** {snapshot.documents:,}",
                f"- **Wikivoyage sections:** {snapshot.wikivoyage_sections:,}",
                f"- **Wikidata facts:** {snapshot.wikidata_facts:,}",
                f"- **Polygon-document links:** {snapshot.polygon_document_links:,}",
                f"- **Wikipedia-tag-only polygons:** {snapshot.wikipedia_tag_only_polygons:,}",
                f"- **Document words:** {snapshot.document_words:,}",
                f"- **Polygon/link-table storage:** {snapshot.polygon_link_storage_bytes / 1_000_000_000:.1f} GB",
                "",
                "</details>",
                "",
                comparison,
                "<details>",
                "<summary>Polygon surface and geometry</summary>",
                "",
                demote_headings(render_polygon_stats_section(processed_v2).strip()),
                "",
                "</details>",
                "",
                "## Coverage maps",
                "",
                "### All V2 dataset polygons",
                "",
                "![All V2 dataset polygons](assets/coverage_map.png)",
                "",
                "Every point is one globally unique retained V2 `(osm_type, osm_id)` identity, including identities without text. Regional polygon rows remain separate source/provenance records.",
                "",
                "### Polygons with Wikipedia or Wikivoyage text",
                "",
                "![V2 polygons with text](assets/geographic_text_presence.png)",
                "",
                "Each point is one globally unique `(osm_type, osm_id)` identity linked to at least one successfully extracted (`fetch_status=ok`) Wikipedia or Wikivoyage document with trimmed non-empty `full_text`. Overlapping regional rows and multiple qualifying documents count once.",
                "",
                "### H3 density of polygons with text",
                "",
                "![V2 geographic text density](assets/geographic_text_density.png)",
                "",
                "Each H3 cell shows the raw count of unique V2 `(osm_type, osm_id)` identities with successfully extracted (`fetch_status=ok`) non-empty Wikipedia or Wikivoyage text. Colour uses a logarithmic scale; it is not a proportion.",
                "",
                "## Deduplication and provenance",
                "",
                "V2 deduplicates documents by `document_id` and polygon-document links by `(polygon_id, project, document_id)` within each region. Byte-identical repeats collapse deterministically; conflicting rows fail closed. `discovery_sources` explains how a polygon was included: `wikidata` means the polygon came from an OSM `wikidata=*` tag, while `wikipedia_tag` means it came from an OSM `wikipedia=*` tag. `link_sources` explains each polygon-document relationship: `wikidata_sitelink` means the relationship came from a Wikidata sitelink, while `osm_wikipedia_tag` means it came directly from an OSM `wikipedia=*` tag. A relationship can list both when both routes agree.",
                "",
                "Regional extracts can overlap, so the same OSM object or document may appear in more than one regional file. We keep those copies to preserve regional membership and provenance. Row-based snapshot, document, link, and storage metrics retain those regional copies; map points, text funnels, and text-covered card metrics use one deterministic representative per global `(osm_type, osm_id)` identity.",
                "",
                "## Sentence-level text",
                "",
                *_sentence_section_lines(snapshot.sentence_stats),
                "",
                "## Repository layout",
                "",
                "- `polygons/<stem>.parquet` — one row per retained OSM polygon.",
                "- `wikipedia/documents/<stem>.parquet` and `wikipedia/sections/<stem>.parquet` — multilingual Wikipedia documents and their sections.",
                "  Wikipedia sections retain the exact V1 22-column section schema for lossless reuse.",
                "- `wikivoyage/documents/<stem>.parquet` and `wikivoyage/sections/<stem>.parquet` — Wikivoyage documents and sections reused from V1 where available.",
                "- `wikipedia/sentences/<stem>.parquet` and `wikivoyage/sentences/<stem>.parquet` — optional sentence rows with explicit split/unsplit provenance.",
                "- `polygon_document_links/<stem>.parquet` — unified Wikipedia and Wikivoyage polygon links.",
                "- `wikidata/facts/<stem>.parquet` — structured Wikidata facts.",
                "- `stats.json` — the complete machine-readable polygon surface and geometry report.",
                "",
                "## Reproducibility",
                "",
                "Run V2 explicitly with `sync-dir --dataset-version v2`. V1 remains a separate contract and is not modified by V2 processing.",
                "",
                "## Citation",
                "",
                "If you use this dataset, please cite it. Download the dataset citation metadata from [`CITATION.cff`](CITATION.cff).",
                "",
            ]
        )
    )


def _render_front_matter(snapshot: V2CardStats, *, processed_v2: Path) -> str:
    configs = [
        ("polygons", "polygons", "polygons/*.parquet"),
        ("polygon_document_links", "polygon_document_links", "polygon_document_links/*.parquet"),
        ("wikipedia_documents", "wikipedia_documents", "wikipedia/documents/*.parquet"),
        ("wikipedia_sections", "wikipedia_sections", "wikipedia/sections/*.parquet"),
        ("wikivoyage_documents", "wikivoyage_documents", "wikivoyage/documents/*.parquet"),
        ("wikivoyage_sections", "wikivoyage_sections", "wikivoyage/sections/*.parquet"),
        ("wikidata_facts", "wikidata_facts", "wikidata/facts/*.parquet"),
    ]
    if _has_parquet(processed_v2 / "wikipedia/sentences"):
        configs.append(
            ("wikipedia_sentences", "wikipedia_sentences", "wikipedia/sentences/*.parquet")
        )
    if _has_parquet(processed_v2 / "wikivoyage/sentences"):
        configs.append(
            ("wikivoyage_sentences", "wikivoyage_sentences", "wikivoyage/sentences/*.parquet")
        )
    lines = [
        "---",
        "license: odbl",
        "language:",
        "  - en",
        "tags:",
        "  - openstreetmap",
        "  - wikidata",
        "  - wikipedia",
        "  - wikivoyage",
        "  - geospatial",
        "configs:",
    ]
    for config_name, split, path in configs:
        lines.extend(
            [
                f"  - config_name: {config_name}",
                "    data_files:",
                f"      - split: {split}",
                f"        path: {path}",
            ]
        )
    lines.extend(
        [
            "dataset_info:",
            f"  version: {V2_DATASET_CARD_VERSION}",
            f"  regions: {snapshot.regions}",
            f"  polygons: {snapshot.polygons}",
            f"  documents: {snapshot.documents}",
            f"dataset_contract: {V2_CONTRACT_VERSION}",
            "---",
        ]
    )
    return "\n".join(lines) + "\n"


def _has_parquet(directory: Path) -> bool:
    return any(directory.glob("*.parquet"))


def _unique_polygon_count(snapshot: V2CardStats) -> int:
    if snapshot.unique_polygon_identities is not None:
        return snapshot.unique_polygon_identities
    return snapshot.polygons


def _non_empty_text_polygon_count(snapshot: V2CardStats) -> int:
    if snapshot.non_empty_text_polygons is not None:
        return snapshot.non_empty_text_polygons
    return dict(snapshot.text_coverage_funnel).get("With non-empty text", 0)


def _sentence_section_lines(stats: _SentenceCardStats | None) -> tuple[str, ...]:
    lines = (
        "Sentence sidecars are opt-in and use `sat-3l-sm` only for the exact ISO codes listed in `docs/sentence-splitting.md` in the [source repository](https://github.com/NoeFlandre/osm-polygon-wikidata-only/blob/main/docs/sentence-splitting.md). Any other language code remains one unsplit row with `segmentation_status=unsupported_language`; it is never passed to SaT.",
    )
    if stats is not None:
        lines += (
            f"Data-derived totals: {stats.split_rows:,} split sentence rows plus {stats.unsupported_rows:,} unsupported-language rows retained unsplit = {stats.total_rows:,} total rows; {stats.supported_language_count:,} supported language codes; {stats.polygon_count:,} polygons linked to sentence-sidecar documents across {stats.wikipedia_sidecars:,} Wikipedia and {stats.wikivoyage_sidecars:,} Wikivoyage sidecars.",
        )
    return (
        *lines,
        "When generated, sentence rows are stored in `wikipedia/sentences/<stem>.parquet` and `wikivoyage/sentences/<stem>.parquet`; `manifests/sentence_splitting.json` records the model, revision, and observed routing.",
    )


def _render_comparison(snapshot: V2CardStats) -> str:
    lines = ["## V2 compared with V1", ""]
    if any(
        value is None
        for value in (
            snapshot.new_polygons_vs_v1,
            snapshot.new_wikipedia_documents_vs_v1,
            snapshot.additional_document_words_vs_v1,
            snapshot.additional_sections_vs_v1,
            snapshot.new_polygons_wikipedia_tag_vs_v1,
            snapshot.new_polygons_wikidata_only_vs_v1,
            snapshot.new_wikipedia_tag_polygons_without_document,
            snapshot.new_wikipedia_document_identity_words_vs_v1,
            snapshot.new_wikipedia_documents_sharing_v1_content,
            snapshot.additional_unique_sections_vs_v1,
            snapshot.new_wikipedia_tag_document_polygons_vs_v1,
        )
    ):
        lines.append(
            "The local V1 artifact root was not supplied for this card render, so the delta is not estimated."
        )
    else:
        lines.extend(
            [
                "The following deltas are computed from the local V1 and V2 Parquet snapshots:",
                "",
                f"- **Additional polygon identities:** {snapshot.new_polygons_vs_v1:,}",
                f"- **Of those, polygons with a Wikipedia tag:** {snapshot.new_polygons_wikipedia_tag_vs_v1:,}",
                f"- **Of those, Wikidata-only polygons:** {snapshot.new_polygons_wikidata_only_vs_v1:,}",
                f"- **Additional Wikipedia document identities:** {snapshot.new_wikipedia_documents_vs_v1:,}",
                f"- **Additional document-row words in V2 (Wikipedia + Wikivoyage):** {snapshot.additional_document_words_vs_v1:,}",
                f"- **Words in newly added Wikipedia document identities:** {snapshot.new_wikipedia_document_identity_words_vs_v1:,}",
                f"- **New Wikipedia document identities sharing content with V1:** {snapshot.new_wikipedia_documents_sharing_v1_content:,}",
                f"- **Additional section rows in V2 (Wikipedia + Wikivoyage):** {snapshot.additional_sections_vs_v1:,}",
                f"- **Additional unique section identities:** {snapshot.additional_unique_sections_vs_v1:,}",
                f"- **New Wikipedia-tag polygons without a matching page at the snapshot:** {snapshot.new_wikipedia_tag_polygons_without_document:,}",
                f"- **V2-added polygons with a new Wikipedia-tag document and no Wikidata discovery:** {snapshot.new_wikipedia_tag_document_polygons_vs_v1:,}",
                "",
                "### V2-added polygons with Wikipedia-tag documents",
                "",
                f"![V2-added polygons with Wikipedia-tag documents]({V2_ADDED_WIKIPEDIA_TAG_MAP_PATH})",
                "",
                "This map shows only polygon identities absent from V1 whose discovery provenance is exactly `wikipedia_tag` and which link to at least one Wikipedia document identity new in V2 through `osm_wikipedia_tag`.",
                "",
                "The polygon and document figures are set differences of stable identities, while row-word and row-section figures include regional copies. V2 keeps regional copies to preserve source membership and provenance; a direct Wikipedia reference without a matching page remains represented in the polygon table and is not counted as a document.",
                "",
            ]
        )
    return "\n".join(lines)


# Public collaborator spellings used by release preparation and the facade.
render_front_matter = _render_front_matter
has_parquet = _has_parquet
unique_polygon_count = _unique_polygon_count
non_empty_text_polygon_count = _non_empty_text_polygon_count
sentence_section_lines = _sentence_section_lines
render_comparison = _render_comparison
