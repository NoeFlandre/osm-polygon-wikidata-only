---
license: odbl
language:
  - en
tags:
  - openstreetmap
  - wikidata
  - wikipedia
  - wikivoyage
  - polygons
  - geospatial
  - multilingual
configs:
  - config_name: polygons
    data_files:
      - split: polygons
        path: polygons/*.parquet
  - config_name: articles
    data_files:
      - split: articles
        path: articles/*.parquet
  - config_name: polygon_articles
    data_files:
      - split: polygon_articles
        path: polygon_articles/*.parquet
  - config_name: wikipedia_documents
    data_files:
      - split: wikipedia_documents
        path: wikipedia/documents/*.parquet
  - config_name: wikipedia_sections
    data_files:
      - split: wikipedia_sections
        path: wikipedia/sections/*.parquet
  - config_name: wikivoyage_documents
    data_files:
      - split: wikivoyage_documents
        path: wikivoyage/documents/*.parquet
  - config_name: wikivoyage_sections
    data_files:
      - split: wikivoyage_sections
        path: wikivoyage/sections/*.parquet
  - config_name: wikidata_facts
    data_files:
      - split: wikidata_facts
        path: wikidata/facts/*.parquet
dataset_info:
  polygon_count: 1
  unique_wikidata_count: 1
  article_count: 1
---

# NoeFlandre/osm-polygon-wikidata-only

OSM polygons tagged with a `wikidata=*` reference, enriched with Wikidata descriptions and Wikipedia article text for every valid language-Wikipedia sitelink, with full text and no per-QID article cap. One PBF produces three parquet files in this Hub:

- `polygons/<stem>.parquet` — one row per polygon
- `articles/<stem>.parquet` — one row per unique Wikipedia article
- `polygon_articles/<stem>.parquet` — many-to-many link table

Optional additive text augmentation is published without replacing those tables:

- `wikipedia/documents/<stem>.parquet` and `wikipedia/sections/<stem>.parquet`
- `wikivoyage/documents/<stem>.parquet` and `wikivoyage/sections/<stem>.parquet`
- `wikidata/facts/<stem>.parquet`

Generated on YYYY-MM-DD.

Maintained by **Noé Flandre**.

## Coverage

![Coverage Map](coverage_map.png)

## Geographic coverage

Both maps below aggregate dataset polygons into H3 cells at the same resolution. All denominators and counts are conditional on each polygon carrying an OSM `wikidata=*` tag.

### Wikipedia text coverage

![Geographic Wikipedia Text Coverage](assets/geographic_wikipedia_text_coverage.png)

`coverage_rate(h) = covered_polygons(h) / all_dataset_polygons(h)`, where a covered polygon has at least one linked Wikipedia article with non-empty text. Cell colour encodes this fraction from 0% to 100%; grey cells hold fewer than 20 polygons and are not statistically meaningful.

### Polygon density

![Geographic Polygon Density](assets/geographic_polygon_count.png)

`polygon_count(h) = number of dataset polygons whose centroid belongs to H3 cell h`. Colour encodes the raw count on a logarithmic scale because counts are highly skewed across the world. Low counts remain visible.


## Schema

### `polygons`

| Column | Description |
| --- | --- |
| `polygon_id` | Deterministic ID: `<source_pbf_stem>:<osm_type>:<osm_id>`. |
| `region` | Geofabrik region slug parsed from the source PBF filename (e.g. `monaco`). |
| `source_pbf` | Source PBF filename (e.g. `monaco-latest.osm.pbf`). |
| `osm_type` | OSM element type: `way` or `relation`. |
| `osm_id` | OpenStreetMap numeric identifier of the element. |
| `wikidata` | Wikidata Q-id from the OSM `wikidata=*` tag. |
| `name` | Convenience: `tags.name` if present, empty string otherwise. |
| `tags` | Deterministic JSON object of all OSM tags except `wikidata`. |
| `tag_keys` | Deterministic JSON list of sorted OSM tag keys. |
| `tag_count` | Number of OSM tags (excluding `wikidata`). |
| `osm_primary_tag` | Best single tag for coarse analysis, e.g. `landuse=forest`. |
| `centroid` | Polygon centroid as a GeoJSON Point string (`[lon, lat]`). |
| `lat` | Centroid latitude in decimal degrees (WGS84). |
| `lon` | Centroid longitude in decimal degrees (WGS84). |
| `bbox` | Bounding box as JSON list `[min_lon, min_lat, max_lon, max_lat]`. |
| `geometry` | Full polygon geometry as deterministic GeoJSON string in WGS84 lon/lat coordinates. |
| `area_m2` | Polygon area in square meters (WGS84 equirectangular approximation). |
| `area_km2` | Polygon area in square kilometers. |
| `area_bucket` | Human-readable size bucket (e.g. `1-10km2`). |
| `has_name` | True if the polygon has a `name` tag. |
| `has_wikidata` | True if the polygon has a `wikidata` tag (always true by filter). |
| `has_wikipedia` | True if at least one linked Wikipedia article was fetched. |
| `wikipedia_language_count` | Number of Wikipedia languages for which articles were linked. |
| `wikipedia_languages` | Deterministic JSON list of language codes (e.g. `["en","fr"]`). |
| `wikipedia_article_count` | Number of unique article revisions linked to this polygon. |
| `has_english_wikipedia` | True if `en` is among the available languages. |
| `has_french_wikipedia` | True if `fr` is among the available languages. |
| `text_available` | True if at least one linked article has non-empty full text. |
| `best_language` | Deterministic preferred language code (e.g. `en`). |
| `extraction_version` | Package version that produced the row. |
| `extracted_at` | ISO-8601 UTC timestamp at the moment the row was extracted. |

### `articles`

| Column | Description |
| --- | --- |
| `article_id` | Deterministic ID: `<wikidata>:<language>:<page_id>:<revision_id>`. |
| `wikidata` | Wikidata Q-id this article is linked to. |
| `language` | Wikipedia language code, e.g. `en`. |
| `site` | Wikidata sitelink site, e.g. `enwiki`. |
| `title` | Article title as returned by the Wikipedia API. |
| `url` | Canonical Wikipedia article URL. |
| `page_id` | MediaWiki page ID. |
| `revision_id` | Exact revision ID used for text extraction. |
| `revision_timestamp` | ISO-8601 timestamp of the revision. |
| `retrieved_at` | ISO-8601 UTC timestamp when this pipeline fetched the article. |
| `wikidata_label` | Best Wikidata label for the article's language, fallback English. |
| `wikidata_description` | Best Wikidata description for the article's language. |
| `wikidata_aliases` | Deterministic JSON list of Wikidata aliases. |
| `lead_text` | Lead section of the article, plain text. |
| `extract` | Short summary/extract if the API provided one. |
| `full_text` | Full cleaned article text, plain text. |
| `full_text_format` | Encoding of `full_text`; always `plain_text`. |
| `article_length_chars` | Length of `full_text` in characters. |
| `article_length_words` | Approximate whitespace-token count of `full_text`. |
| `article_length_tokens_estimate` | Rough token count: `chars / 4`. |
| `thumbnail_url` | Thumbnail URL only (no image bytes stored). |
| `thumbnail_width` | Thumbnail width in pixels, if known. |
| `thumbnail_height` | Thumbnail height in pixels, if known. |
| `categories` | Deterministic JSON list of category titles, or `[]`. |
| `license` | License string, e.g. `CC BY-SA`. |
| `attribution` | Attribution string for the article. |
| `source_api` | Which API was queried: `mediawiki_action_api` / `wikipedia_rest_api`. |
| `fetch_status` | One of: `ok`, `article_not_found`, `http_error`, `rate_limited`, `parse_error`, `empty_text`. |
| `fetch_error` | Short diagnostic on failure, empty string on success. |
| `content_hash` | Stable SHA-256 of `full_text` for change tracking. |

### `polygon_articles`

| Column | Description |
| --- | --- |
| `polygon_id` | FK to `polygons.polygon_id`. |
| `article_id` | FK to `articles.article_id`. |
| `wikidata` | Wikidata Q-id (denormalized for fast filtering). |
| `language` | Wikipedia language code. |
| `source_pbf` | Source PBF filename (denormalized). |
| `region` | Geofabrik region slug. |
| `osm_type` | OSM element type. |
| `osm_id` | OSM numeric identifier. |
| `page_id` | MediaWiki page ID of the linked article. |
| `revision_id` | MediaWiki revision ID of the linked article. |
| `is_best_language` | True if this row's language matches the polygon's `best_language`. |

### `wikipedia/documents` and `wikivoyage/documents`

| Column | Description |
| --- | --- |
| `document_id` | Deterministic document identifier (`<wikidata>:<project>:<language>:<page_id>:<revision_id>`). |
| `article_id` | Stable article identifier that pairs this document with its `articles/<stem>.parquet` row. |
| `wikidata` | Wikidata QID this document is linked to. |
| `project` | Wiki project name: `wikipedia` or `wikivoyage`. |
| `language` | Wikipedia or Wikivoyage language code (e.g. `en`). |
| `site` | Wikidata sitelink host, e.g. `enwiki` or `enwikivoyage`. |
| `title` | Page title as returned by the MediaWiki API. |
| `url` | Canonical URL of the page. |
| `page_id` | MediaWiki page ID (integer). |
| `revision_id` | MediaWiki revision ID used to fetch the page text (integer). |
| `revision_timestamp` | ISO-8601 timestamp of the revision. |
| `retrieved_at` | ISO-8601 UTC timestamp when the pipeline fetched the page. |
| `full_text` | Cleaned plain-text document body (Wikipedia articles or Wikivoyage pages). |
| `full_text_format` | Encoding of `full_text`; always `plain_text`. |
| `article_length_chars` | Length of `full_text` in characters. |
| `article_length_words` | Approximate whitespace-token count of `full_text`. |
| `article_length_tokens_estimate` | Rough token estimate: `chars / 4`. |
| `license` | License string (`CC BY-SA` for Wikipedia/Wikivoyage text). |
| `attribution` | Attribution string for the page. |
| `source_api` | Which API was queried: `mediawiki_action_api` or `wikivoyage_rest_api`. |
| `fetch_status` | One of: `ok`, `page_not_found`, `http_error`, `rate_limited`, `parse_error`, `empty_text`. |
| `fetch_error` | Short diagnostic on failure, empty string on success. |
| `content_hash` | Stable SHA-256 of `full_text` for change tracking. |

### `wikipedia/sections` and `wikivoyage/sections`

| Column | Description |
| --- | --- |
| `section_id` | Deterministic SHA-256 over `(document_id, section_index, anchor)`. |
| `document_id` | FK back to `documents` (Wikipedia or Wikivoyage). |
| `article_id` | FK back to the corresponding `articles` row. |
| `wikidata` | Wikidata QID (denormalized for fast filtering). |
| `project` | Wiki project name: `wikipedia` or `wikivoyage`. |
| `language` | Wikipedia or Wikivoyage language code. |
| `site` | Wikidata sitelink host. |
| `page_id` | MediaWiki page ID (integer). |
| `revision_id` | MediaWiki revision ID (integer). |
| `section_index` | Sequential position of the section inside the document (integer). |
| `heading` | Section heading, or empty string when the section is the lead. |
| `anchor` | Section anchor after MediaWiki parsing. |
| `level` | Heading level (1..6), or 0 for the lead section (integer). |
| `parent_section_id` | Section ID of the enclosing section, or empty string. |
| `section_path` | JSON array of ancestor section IDs, in order. |
| `text` | Plain-text section body. |
| `text_length_chars` | Length of `text` in characters (integer). |
| `text_length_words` | Approximate whitespace-token count of `text` (integer). |
| `text_length_tokens_estimate` | Rough token estimate: `chars / 4` (integer). |
| `content_hash` | Stable SHA-256 of `section.text`. |
| `license` | License string for this section. |
| `attribution` | Attribution string for this section. |

### `wikidata/facts`

| Column | Description |
| --- | --- |
| `fact_id` | Deterministic SHA-256 over `(subject, property, value, ordinal)`. |
| `wikidata` | Wikidata QID the fact belongs to (the subject). |
| `property_id` | Property P-id (e.g. `P17`). |
| `property_label_en` | English label for the property, when available. |
| `property_labels` | Deterministic JSON object of property labels per language. |
| `value_type` | Wikidata value datatype: `wikibase-entityid`, `string`, `quantity`, `time`, ... |
| `value_entity_id` | Entity-valued object QID, when the value is a Wikidata entity. |
| `value_label_en` | English label for the value entity, when available. |
| `value_labels` | Deterministic JSON object of value labels per language. |
| `value_text` | Rendered text representation of the value. |
| `numeric_value` | Numeric amount for `quantity`-typed values, otherwise null. |
| `unit_entity_id` | Wikidata QID of the unit (e.g. `Q11573` for metre). |
| `rank` | Wikidata rank: `preferred`, `normal`, or `deprecated`. |
| `qualifiers` | Deterministic JSON object of qualifier snaks, or `{}` when absent. |
| `references` | Deterministic JSON array of reference groups, or `[]` when absent. |
| `retrieved_at` | ISO-8601 UTC timestamp when the pipeline fetched the entity. |
| `source_api` | Which API was queried (always `wikidata_action_api`). |


## Data sources & licenses

- **OpenStreetMap** polygons: (c) OpenStreetMap contributors, licensed under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/).
- **Wikidata** entity data: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
- **Wikipedia** article text: licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) by the respective Wikipedia editors; attributed inline per article.

## How to load

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files={
    "polygons": "hf://datasets/NoeFlandre/osm-polygon-wikidata-only/polygons/*.parquet",
})
```
