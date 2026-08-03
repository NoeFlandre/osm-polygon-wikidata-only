"""Immutable values for the published ``final-dataset-snapshot`` run.

The run is deliberately a frozen publication artifact. It is not a training
run and therefore has no epochs, progress counters, request metrics, or
pipeline timing fields.
"""

from __future__ import annotations

from dataclasses import dataclass

TRACKIO_PROJECT = "osm-polygon-wikidata-only"
TRACKIO_RUN_NAME = "final-dataset-snapshot"
TRACKIO_SPACE_ID = "NoeFlandre/osm-polygon-wikidata-only-trackio"
TRACKIO_DATASET_ID = "NoeFlandre/osm-polygon-wikidata-only-trackio-data"
TRACKIO_SPACE_URL = f"https://huggingface.co/spaces/{TRACKIO_SPACE_ID}"
DATASET_PRESENTATION_URL = (
    "https://noeflandre.github.io/osm-polygon-wikidata-only/presentations/dataset.html"
)


@dataclass(frozen=True, slots=True)
class FinalDatasetSnapshot:
    """The public metrics and plot data for the finished dataset."""

    polygons: int = 1_184_110
    unique_wikidata_entities: int = 1_119_223
    documents: int = 2_288_170
    document_words: int = 801_528_334
    languages: int = 351
    geographic_regions: int = 375

    wikipedia_documents: int = 2_273_750
    wikipedia_sections: int = 11_997_165
    wikivoyage_documents: int = 14_420
    wikivoyage_sections: int = 302_200
    wikidata_facts: int = 3_901_092

    wikipedia_polygon_document_links: int = 2_468_604
    polygon_link_storage_gb: float = 9.9
    total_parquet_storage_gb: float = 19.2

    # The funnel uses the canonical Wikipedia polygon fields. Those fields
    # define English and multi-language coverage; Wikivoyage is represented
    # in the combined corpus and composition metrics above.
    text_coverage_funnel: tuple[tuple[str, int], ...] = (
        ("All polygons", 1_184_110),
        ("With non-empty text", 666_226),
        ("English coverage", 238_917),
        ("Non-English-only coverage", 427_311),
        ("2+ languages", 289_113),
        ("5+ languages", 108_007),
        ("10+ languages", 47_976),
    )
    top_wikipedia_languages: tuple[tuple[str, int], ...] = (
        ("en", 224_462),
        ("de", 149_091),
        ("fr", 112_223),
        ("ceb", 110_289),
        ("sv", 76_085),
        ("ru", 75_557),
        ("es", 72_538),
        ("it", 69_103),
        ("pl", 66_276),
        ("zh", 60_408),
    )

    @property
    def other_wikipedia_languages(self) -> int:
        """Return Wikipedia documents outside the displayed top ten."""
        return self.wikipedia_documents - sum(value for _, value in self.top_wikipedia_languages)

    def metrics(self) -> dict[str, int]:
        """Return only the approved static Trackio scalar metrics."""
        return {
            "scale/polygons": self.polygons,
            "scale/unique_wikidata_entities": self.unique_wikidata_entities,
            "corpus/documents": self.documents,
            "corpus/document_words": self.document_words,
            "coverage/languages": self.languages,
            "coverage/geographic_regions": self.geographic_regions,
        }

    def table_rows(self) -> tuple[tuple[str, str], ...]:
        """Return the small public snapshot table in display order."""
        return (
            ("Wikipedia polygon-document links", f"{self.wikipedia_polygon_document_links:,}"),
            ("Polygon/link-table storage", f"{self.polygon_link_storage_gb:.1f} GB"),
            ("Total Parquet storage", f"{self.total_parquet_storage_gb:.1f} GB"),
        )


FINAL_DATASET_SNAPSHOT = FinalDatasetSnapshot()


__all__ = [
    "DATASET_PRESENTATION_URL",
    "FINAL_DATASET_SNAPSHOT",
    "TRACKIO_DATASET_ID",
    "TRACKIO_PROJECT",
    "TRACKIO_RUN_NAME",
    "TRACKIO_SPACE_ID",
    "TRACKIO_SPACE_URL",
    "FinalDatasetSnapshot",
]
