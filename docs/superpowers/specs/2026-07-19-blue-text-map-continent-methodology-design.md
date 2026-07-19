# Blue Text Map and Continent Methodology Design

## Scope

Make the first two dataset-card maps visually distinct and explain the automatically generated continent table without changing dataset schemas, counts, or pipeline semantics.

## Visual design

The combined Wikipedia-or-Wikivoyage text-presence map uses publication blue (`#2563EB`) with a darker blue edge. The all-polygons map retains its existing orange palette. The combined renderer passes its palette explicitly so the shared coverage-map defaults remain unchanged.

## Geographic methodology documentation

The generated continent section explains that each polygon is assigned from its WGS84 centroid through the bundled Natural Earth 1:110m Admin-0 country boundaries and their continent property. It defines every column, the text-coverage formula, distinct-document counting, the polygon-to-Wikipedia link relationship, the Wikidata-based Wikivoyage relationship, multi-continent document behavior, and the meaning of `Unassigned`.

All prose and values are rendered by `render_continent_stats`. Values continue to be recomputed from finalized Parquet tables before each README publication; no statistics are hardcoded and no secondary workflow is added.

## Verification

Tests inspect the combined renderer's explicit blue palette while confirming the all-polygons defaults remain orange. Markdown tests freeze the methodology, metric definitions, formula, and automatic publication path. The real map and README are regenerated and published atomically after the full quality gates pass.
