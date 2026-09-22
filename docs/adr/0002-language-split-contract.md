# ADR 0002: Row-level language-split contract

- Status: Accepted
- Date: 2026-09-16

## Decision

Language partitions are keyed only by the `language` column on a textual or
document row. The polygon tables are not partitioned by `best_language`: that
field is a polygon-level preference, not the language of a particular text
row. A polygon can therefore be represented by several link and document rows,
one in each relevant language partition.

The shared language policy is applied independently to the two dataset
contracts:

- V1 (`NoeFlandre/osm-polygon-wikidata-only`) inventories canonical
  `polygon_articles`, Wikipedia documents and sections, and Wikivoyage
  documents and sections.
- V2 (`NoeFlandre/osm-polygon-wikidata-and-wikipedia`) inventories canonical
  `polygon_document_links`, Wikipedia documents, and Wikipedia sections. V2
  uses its own schemas and manifest under `processed_v2/`; it is never mixed
  with V1 artifacts.

The normalizer follows the repository's existing language rule: trim, lower
case, replace `_` with `-`, and accept only
`[a-z]{2,3}(?:-[a-z0-9]{2,8})*`. The existing usable legacy alias
`be_x_old` becomes `be-tarask`. Null, whitespace-only, non-string, malformed,
and legacy-unusable values (including the observed V1 project labels
`simple` and `abstract`) are retained in an explicit `lang-unknown` bucket;
they are never dropped or converted to a guessed language.

Each language-bearing table gets an additive Hugging Face configuration named
`<table>_by_language`. V1 splits are `lang-<canonical-language>`, including
`lang-unknown`. V2 Viewer splits are `lang_<canonical-language>` with dashes in
the language code replaced by underscores, including `lang_unknown`; V2 storage
directories retain the canonical `lang-<language>` form. These prefixes keep
split names unambiguous and avoid reserved generic names such as `train`,
`test`, and `validation`. The V1 default configurations and paths remain
unchanged. The V2 configurations are separate because its repository, root,
schemas, and link table are separate.

The inventory is computed from schema-validated Parquet artifacts and the
corresponding contract manifest. It records the dataset contract, source
manifest digest, artifact fingerprint, source files, row counts, and counts
for canonical, legacy-alias, missing, blank, malformed, and
legacy-unusable values per table and partition. Languages are discovered from
the rows at inventory time; no language list is checked in.

Rows retain their existing schema, source fields, and identity. V1 and V2
document identity is `document_id`, section identity is `section_id`, and link
identity is `(polygon_id, project, document_id)`. Inventory and later
partitioning preserve physical rows in deterministic sorted source-file order
and do not deduplicate by `(osm_type, osm_id)`.

## User loading examples

The default V1 dataset remains loadable as before. A language-only table uses
the additive configuration and split:

```python
from datasets import load_dataset

french_documents = load_dataset(
    "NoeFlandre/osm-polygon-wikidata-only",
    name="wikipedia_documents_by_language",
    split="lang-fr",
)
```

The corresponding V2 call is explicit about the separate repository and
contract:

```python
french_v2_links = load_dataset(
    "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
    name="polygon_document_links_by_language",
    split="lang_fr",
)
```

The equivalent Hub CLI selection is scoped to one configuration and split:

```bash
hf download NoeFlandre/osm-polygon-wikidata-only \
  --repo-type dataset \
  --include 'wikipedia_documents_by_language/lang-fr/**'
```

This ADR defines the contract and inventory only. Partition generation,
property/integration coverage for that generator, and Hub publication remain
the separate downstream issues #7, #8, and #9.

## Rationale

Using a row-level language avoids losing multilingual coverage or collapsing
different documents associated with the same polygon. Separate configurations
preserve the existing heterogeneous table schemas and let `datasets` select a
single language split without downloading the default all-language table.
