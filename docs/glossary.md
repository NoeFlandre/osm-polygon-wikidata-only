# Glossary

This page defines the project terms. Each term has one meaning in all the documents.

| Term | Meaning |
| --- | --- |
| Region | One geographic area. It comes from one Geofabrik extract. |
| PBF | An `.osm.pbf` file. It holds OpenStreetMap data for one region. |
| Polygon | A closed OSM way or a multipolygon relation. |
| Stem | The file name of a PBF, without the extension. Tables use it in their file names. |
| Data root | The directory that holds the raw input, the processed tables, and the restart state. It is always outside the source checkout. |
| Source checkout | The local copy of this Git repository. |
| Contract | The set of rules for the selected tables, schemas, and manifests of a dataset version. |
| V1 | The default dataset contract. It selects polygons that have a `wikidata=*` tag. |
| V2 | The optional dataset contract. It also selects polygons that have a valid `wikipedia=*` tag. A V2 row can have no Wikidata QID. |
| QID | The identifier of a Wikidata entity. |
| Enrichment | The step that adds Wikidata, Wikipedia, and Wikivoyage data to the polygons. |
| Augmentation | The step that adds section tables and Wikidata facts to a completed region. |
| Document | One Wikipedia or Wikivoyage page revision in one language. |
| Section | One part of a document, with its own text. |
| Link table | The table that links polygons to documents. |
| Language split | A partition of a table by the `language` column of each row. |
| Unknown bucket | The `lang-unknown` partition. It holds rows whose language value is not usable. |
| Sentence sidecar | A Parquet table of sentences. The sentence stage creates it from a section table. |
| SaT | The sentence segmentation model `segment-any-text/sat-3l-sm`. |
| Manifest | A JSON file that records counts and provenance. |
| Hub | The Hugging Face Hub. |
| Publish | Upload files to the Hub. All publish options are opt-in. |
| Dry run | A run that validates the input and prints a plan. It does not write output. |
| ADR | Architecture decision record. |
| CRAP | The score that combines the complexity and the coverage of a function. |
| Quality gauntlet | The fail-fast sequence of checks that you run before a review. |
| Grid5000 | The French research compute platform. The project uses it for GPU jobs. |
| Controller | The local program that submits and monitors the Grid5000 jobs. |
| Ledger | The file `grid5000_sentence_run.json`. It records the state of a Grid5000 run. |
