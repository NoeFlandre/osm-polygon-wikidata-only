# Text Augmentation Sidecars Design

## Goal

Add structured Wikipedia sections, Wikivoyage documents and sections, and
selected Wikidata geographic facts without removing, rewriting, or changing any
column in the existing polygon, article, polygon-article, or manifest artifacts.
Validate the design on Andorra and publish only its augmentation artifacts to
the existing Hugging Face dataset.

## Compatibility contract

The current three Parquet tables remain byte-for-byte untouched during an
augmentation run. Their schemas, paths, IDs, row order, manifests, skip logic,
cache formats, and CLI behavior remain supported. Augmentation files are
strictly additive and join through existing `wikidata` and `article_id` values.

An augmentation failure cannot invalidate or delete a completed core region.
No PBF is read during backfill. Existing regions are inputs, not work to redo.

## Architecture

Create an `augmentation` package with focused document discovery, MediaWiki
section parsing, Wikidata fact normalization, sidecar row construction,
publication, and orchestration modules. A new CLI command augments one existing
region stem from its published Parquet files. The Andorra pilot is explicit;
normal `process-pbf` and `process-dir` behavior does not change during the pilot.

After the pilot is accepted, the same orchestrator can be called after normal
core publication and can backfill all existing manifest entries. Both paths use
the same cache contracts and output schemas.

## Sidecar tables

### Documents

Path: `augmentations/documents/<stem>.parquet`

One row represents an exact Wikimedia document revision. Columns:

- `document_id`: deterministic
  `<wikidata>:<project>:<language>:<page_id>:<revision_id>`;
- `article_id`: existing Wikipedia article ID when applicable, otherwise empty;
- `wikidata`, `project`, `language`, `site`, `title`, `url`;
- `page_id`, `revision_id`, `revision_timestamp`, `retrieved_at`;
- `full_text`, `full_text_format`, character/word/token counts;
- `license`, `attribution`, `source_api`, `fetch_status`, `fetch_error`;
- `content_hash`.

Existing Wikipedia article rows are converted locally into document rows. They
are not refetched merely to duplicate full text. Wikivoyage sitelinks and
documents are fetched through the shared Wikimedia scheduler and a separate,
versioned augmentation cache.

### Sections

Path: `augmentations/sections/<stem>.parquet`

One row represents an ordered section from an exact document revision. Columns:

- `section_id`, `document_id`, `article_id`, `wikidata`;
- `project`, `language`, `site`, `page_id`, `revision_id`;
- `section_index`, `heading`, `anchor`, `level`;
- `parent_section_id`, `section_path` as deterministic JSON;
- `text`, character/word/token counts, `content_hash`;
- `license`, `attribution`.

The lead is section index zero with an empty heading and level zero. Section
text is clean plain text. Navigation, references, edit controls, scripts, and
empty sections are excluded. Parsing uses the exact revision ID so documents
and sections cannot drift.

### Wikidata facts

Path: `augmentations/wikidata_facts/<stem>.parquet`

One row represents a normalized selected claim. Columns:

- `fact_id`, `wikidata`, `property_id`, `property_label`;
- `value_type`, `value_entity_id`, `value_label`, `value_text`;
- `numeric_value`, `unit_entity_id`, `rank`;
- `qualifiers`, `references` as deterministic JSON;
- `retrieved_at`, `source_api`.

The initial allow-list is instance of, subclass of, administrative parent,
country, part of, elevation, inception, heritage designation, and protected
classification where present. Entity-valued claims resolve labels in the
document language when available and fall back to English. Unsupported value
types are skipped deterministically rather than stringified ambiguously.

## Incremental processing

An augmentation manifest lives at
`augmentations/manifests/augmentation_manifest.json`. Each region entry records:

- augmentation contract version;
- source core manifest entry and input file fingerprints;
- document, section, and fact paths and row counts;
- source revision IDs and cache contract versions;
- completion timestamp.

The orchestrator skips a region only when the contract version and all input
fingerprints match and every sidecar exists. If the core region or augmentation
contract changes, only that region is rebuilt. Atomic temporary files prevent
partial sidecars from appearing complete.

## Andorra pilot and upload

The pilot reads the existing Andorra core Parquet files, produces all three
sidecars, validates joins and deterministic ordering, and leaves the core files
unchanged. The upload is one atomic Hugging Face commit containing:

- the three Andorra augmentation Parquet files;
- the augmentation manifest snapshot; and
- a dataset-card update documenting schemas and joins.

No other region or core artifact is uploaded by the pilot.

## Failure handling

Missing Wikivoyage pages and empty sections are non-fatal data outcomes.
Authentication fallback follows the existing Wikimedia policy. Rate limits,
HTTP failures, malformed responses, schema violations, unresolved required
joins, and partial publication fail the augmentation run without modifying core
artifacts or marking the augmentation complete.

## Testing

Development follows red-green-refactor. Required tests cover:

- deterministic schemas, IDs, ordering, and hashes;
- existing Wikipedia article-to-document conversion without network access;
- multilingual Wikivoyage discovery and exact-revision retrieval;
- lead/section hierarchy and plain-text parsing;
- selected Wikidata claim/value normalization;
- cache hits, stale contracts, and interrupted resume;
- atomic sidecar publication and augmentation-manifest ordering;
- zero changes to existing Andorra core files;
- offline end-to-end Andorra fixtures; and
- live Andorra smoke validation before the explicitly requested upload.

The full existing test, coverage, Ruff, mypy, and package-build gates must pass
before the pilot is published.

## Completion criteria

The design is complete when Andorra has three joinable, validated augmentation
sidecars on the remote dataset, all existing artifacts remain unchanged, the
augmentation is resumable and versioned, and the same code can backfill other
completed regions without rereading their PBFs.
