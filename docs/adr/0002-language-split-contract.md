# ADR 0002: Row-level language-split contract

- Status: Accepted
- Date: 2026-09-16

## Decision

The `language` column of a text row or a document row is the only key of the language partitions. The polygon tables are not partitioned by `best_language`. That field is the preference of a polygon. It is not the language of a specific text row. A polygon can have several link rows and document rows, one in each applicable language partition.

The shared language policy applies independently to the two dataset contracts:

- V1 (`NoeFlandre/osm-polygon-wikidata-only`) has an inventory of these tables: the canonical `polygon_articles`, the Wikipedia documents and sections, and the Wikivoyage documents and sections.
- V2 (`NoeFlandre/osm-polygon-wikidata-and-wikipedia`) has an inventory of these tables: the canonical `polygon_document_links`, the Wikipedia documents, and the Wikipedia sections. V2 has its own schemas and manifest under `processed_v2/`. Never mix V2 with the V1 artifacts.

The normalizer follows the existing language rule of the repository:

1. Trim the value.
2. Convert it to lower case.
3. Replace `_` with `-`.
4. Accept only values that match `[a-z]{2,3}(?:-[a-z0-9]{2,8})*`.

The usable legacy alias `be_x_old` becomes `be-tarask`. The explicit `lang-unknown` bucket keeps these values: null, only whitespace, not a string, malformed, and legacy and not usable. The V1 project labels `simple` and `abstract` are in this group. The pipeline never drops these values. It never changes them to a guessed language.

Each table that has a language gets additive Hugging Face language selections:

- V1 keeps one `<table>_by_language` configuration. Its splits have the name `lang-<canonical-language>`. This includes `lang-unknown`.
- V2 uses one Viewer configuration for each pair of table and language. The name is `<table>_by_language__lang_<canonical-language>`, with underscores in place of dashes. The configuration has one `train` split. The V2 storage directories keep the canonical form `lang-<language>`.

This design puts the language dimension in the Viewer configurations. It does not make hundreds of splits in one configuration. The split names stay unambiguous. They do not use the reserved generic names `train`, `test`, and `validation`. The V1 default configurations and paths do not change. The V2 configurations are separate because V2 has its own repository, root, schemas, and link table.

The pipeline computes the inventory from the Parquet artifacts that pass schema validation and from the manifest of the contract. The inventory records these items:

- The dataset contract.
- The digest of the source manifest.
- The fingerprint of the artifacts.
- The source files.
- The row counts.
- The counts of canonical, legacy-alias, missing, blank, malformed, and legacy-unusable values for each table and partition.

The pipeline finds the languages in the rows when it computes the inventory. The repository does not contain a list of languages.

The rows keep their schema, source fields, and identity. In V1 and V2, the document identity is `document_id` and the section identity is `section_id`. The link identity is `(polygon_id, project, document_id)`. The inventory and the later partitioning keep the physical rows in the deterministic sorted order of the source files. They do not remove duplicates by `(osm_type, osm_id)`.

## User loading examples

The default V1 dataset loads as before. A table that has only a language uses the additive configuration and split:

```python
from datasets import load_dataset

french_documents = load_dataset(
    "NoeFlandre/osm-polygon-wikidata-only",
    name="wikipedia_documents_by_language",
    split="lang-fr",
)
```

The V2 call names the separate repository and contract:

```python
french_v2_links = load_dataset(
    "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
    name="polygon_document_links_by_language__lang_fr",
    split="train",
)
```

The Hub CLI selects one configuration and one split:

```bash
hf download NoeFlandre/osm-polygon-wikidata-only \
  --repo-type dataset \
  --include 'wikipedia_documents_by_language/lang-fr/**'
```

This ADR defines only the contract and the inventory. Other issues cover the next work: #7 (partition generation), #8 (property and integration coverage for the generator), and #9 (Hub publication).

## Rationale

A row-level language keeps the multilingual coverage. It does not merge different documents that belong to the same polygon. Separate configurations keep the different table schemas. They let `datasets` select one language split. The user does not download the default table with all languages.
