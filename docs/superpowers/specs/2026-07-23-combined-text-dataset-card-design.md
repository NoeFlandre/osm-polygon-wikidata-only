# Combined Text Dataset Card Design

## Objective

Make the Hugging Face dataset card describe and measure the complete published
Wikipedia plus Wikivoyage text corpus. Replace the two lower-value H3 maps with
one deterministic raw-density map of polygons having non-empty text from either
project.

## Public table descriptions

The card lists every canonical table with one concise description:

- `polygons`: one row per OSM polygon carrying `wikidata=*`.
- `wikipedia/documents`: canonical full Wikipedia documents.
- `polygon_articles`: Wikipedia-only polygon-to-document links.
- `wikipedia/sections`: Wikipedia documents partitioned into section rows.
- `wikivoyage/documents`: full Wikivoyage documents associated through QID.
- `wikivoyage/sections`: Wikivoyage documents partitioned into section rows.
- `wikidata/facts`: structured Wikidata claims for polygon QIDs.

The card explicitly states that Wikivoyage has no `polygon_articles` table:
its documents associate with every matching polygon through the shared
Wikidata QID.

## Maps

The public card contains exactly three maps in this order:

1. Individual polygons with non-empty Wikipedia or Wikivoyage text.
2. All dataset polygons.
3. H3 density of polygons with non-empty Wikipedia or Wikivoyage text.

The third map counts each qualifying polygon once, even when it has documents
from both projects or several languages. Its H3 cell value is:

`text_polygon_count(h) = number of qualifying polygon centroids in h`

It uses a logarithmic purple-to-yellow colour scale. It is not normalized by
all polygons. The two superseded assets
`assets/geographic_wikipedia_text_coverage.png` and
`assets/geographic_polygon_count.png` are deleted atomically when the new
canonical `assets/geographic_text_density.png` is added.

All map inputs come from finalized Parquet tables and are sorted before
aggregation and rendering. Existing antimeridian-safe H3 geometry and
deterministic PNG writing are reused.

## Combined language statistics

All public language-distribution, concentration and long-tail metrics use the
union of canonical Wikipedia and Wikivoyage documents:

- Document counts are grouped by `(project, document_id)` and language.
- Wikipedia polygon-language coverage follows `polygon_articles.article_id`
  to non-empty Wikipedia documents.
- Wikivoyage polygon-language coverage joins non-empty documents to polygons
  through Wikidata QID.
- A polygon is counted once per language even if both projects or multiple
  documents provide that language.
- Concentration percentages use the total combined document count.

Project-specific corpus sections remain because they describe distinct source
corpora. The older Wikipedia-only funnel remains clearly labelled
Wikipedia-only unless a metric is recomputed from both sources; no existing
field is relabelled to imply broader coverage than its data supports.

## Automatic publication

The existing publication assembly remains the only workflow. Whenever
canonical text or core data changes, it generates the combined density map,
the other current maps, factual statistics, and README before upload. README is
the final add operation. Publication adds the new canonical asset and safely,
idempotently deletes both legacy H3 assets in the same commit.

A one-time live publication uses current finalized local data to update the
Hugging Face card immediately. Future `sync-dir --push` runs reproduce it.

## Failure and performance behavior

Missing required directories or columns fail with actionable errors before
submission. Expensive I/O is column-pruned. Combined language summaries use a
content/fingerprint cache consistent with the existing dataset-statistics cache;
unchanged Parquet inputs are not rescanned. No Wikimedia requests are needed
for card generation.

## TDD and verification

Tests first cover:

- Wikipedia-only link-table wording and Wikivoyage QID association wording.
- Descriptions for all canonical tables.
- Exactly three maps in the intended order.
- H3 raw-count aggregation, cross-project deduplication and logarithmic scale.
- Combined document-language counts and combined polygon-language counts.
- Deterministic output and stable canonical paths.
- Automatic publication and atomic deletion of both superseded assets.
- README-last ordering and no submission on snapshot failure.
- Small synthetic end-to-end generation without network access.

Run the complete test suite, coverage, Ruff, formatting, mypy and
`git diff --check`; then generate the real asset, visually inspect it, publish
the card/assets, and verify live Hub paths and README references.
