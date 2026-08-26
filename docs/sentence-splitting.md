# V2 sentence splitting

V2 sentence sidecars are an opt-in post-processing stage over finalized V2
section tables. The stage uses only
[`segment-any-text/sat-3l-sm`](https://huggingface.co/segment-any-text/sat-3l-sm),
with model revision `137da05` and ONNX Runtime's CPU provider. It is a separate
command, so ordinary V1 and V2 synchronization does not download a sentence
model or change existing section files.

## Exact language scope

Only these exact 85 language codes are sent to SaT:

```text
af am ar az be bg bn ca ceb cs cy da de el en eo es et eu fa fi fr fy ga gd gl gu ha he hi hu hy id ig is it ja jv ka kk km kn ko ku ky la lt lv mg mk ml mn mr ms mt my ne nl no pa pl ps pt ro ru si sk sl sq sr sv ta te tg th tr uk ur uz vi xh yi yo zh zu
```

Language matching is exact. For example, `en` and `zh` are supported, while
`xx` and `zh-hans` are not. Only these exact language codes are sent to SaT.
Every other language code remains in the output as one unsplit row; it is never
sent to SaT and is marked with
`segmentation_status=unsupported_language`. The complete observed list of
unsupported languages and the model provenance are recorded in
`manifests/sentence_splitting.json`.

This policy is intentional because the V2 snapshot contains language codes
outside the SaT model's supported set. Unsupported text is preserved rather
than dropped or silently routed through another model. Empty supported sections
produce no sentence rows; non-empty supported sections produce lossless rows.

## Install and run

Install the optional sentence runtime:

```bash
uv sync --extra sentence-splitting
```

With a finalized V2 data root selected, run:

```bash
uv run osm-polygon-wikidata-only split-v2-sentences \
  --data-root "$OSM_POLYGON_DATA_ROOT" \
  --batch-size 256 \
  --inference-batch-size 16
```

Add `--push` only after reviewing the local sidecars and manifest. The command
uses the V2 dataset repository selected by the CLI default; `--repo-id` remains
available for an intentional override. `--inference-batch-size 16` is a
conservative starting point for an 8 GB MacBook Air M2 and can be lowered if
the model cache or other applications need more memory.

## Outputs and resumability

For each finalized region, the command reads the existing section table in
bounded Parquet batches and writes:

- `wikipedia/sentences/<stem>.parquet`;
- `wikivoyage/sentences/<stem>.parquet` when a Wikivoyage section table exists;
- `manifests/sentence_splitting.json` with model, revision, supported routing,
  observed unsupported languages, and per-table counts.

Sentence rows retain the source section context and add `sentence_index`,
`start_char`, `end_char`, sentence text and length metrics, content hashes,
`segmenter`, `segmenter_version`, `model_id`, and `segmentation_status`.
Supported rows have `segmentation_status=split`; unsupported rows have
`segmentation_status=unsupported_language` and cover the complete source
section from character 0 to its length. Sentence pieces are required to
reconstruct the source section exactly, including whitespace.

Each source batch is written atomically to restart state under the external
data root. The restart state records the source file hash, batch size, model
identifier, and model revision, so changing any of those inputs starts a fresh
contract. A stopped run reuses valid completed batches and never rewrites the
source section tables. The final sentence Parquet file and routing manifest
are published only after every source batch has completed.

The generated V2 dataset card repeats this policy so a published snapshot makes
the supported-language boundary and the explicit unsplit treatment visible.
