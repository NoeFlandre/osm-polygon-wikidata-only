# V2 sentence splitting

The V2 sentence sidecars are an opt-in post-processing stage. The stage uses the finalized V2 section tables. It uses only the model [`segment-any-text/sat-3l-sm`](https://huggingface.co/segment-any-text/sat-3l-sm), with model revision `137da05`. On Apple Silicon, ONNX Runtime prefers its CoreML provider. If CoreML is not available, it uses the CPU provider. Other platforms use the CPU.

The stage is a separate command. Ordinary V1 and V2 synchronization does not download a sentence model. It does not change the existing section files.

## Exact language scope

The command sends only these 85 language codes to SaT:

```text
af am ar az be bg bn ca ceb cs cy da de el en eo es et eu fa fi fr fy ga gd gl gu ha he hi hu hy id ig is it ja jv ka kk km kn ko ku ky la lt lv mg mk ml mn mr ms mt my ne nl no pa pl ps pt ro ru si sk sl sq sr sv ta te tg th tr uk ur uz vi xh yi yo zh zu
```

The language match is exact. For example, `en` and `zh` are supported. `xx` and `zh-hans` are not supported. Only these exact language codes are sent to SaT. Each other language code stays in the output as one unsplit row. The command never sends it to SaT. The command marks it with `segmentation_status=unsupported_language`. The file `manifests/sentence_splitting.json` records the complete list of the unsupported languages that the command observed. It also records the model provenance.

This policy is intentional. The V2 snapshot has language codes that the SaT model does not support. The command keeps the unsupported text. It does not drop the text. It does not send the text through another model. A supported section that is empty produces no sentence rows. A supported section that is not empty produces lossless rows.

## Install and run

Install the optional sentence runtime:

```bash
uv sync --extra sentence-splitting
```

Select a finalized V2 data root. Then run:

```bash
uv run osm-polygon-wikidata-only split-v2-sentences \
  --data-root "$OSM_POLYGON_DATA_ROOT" \
  --batch-size 256 \
  --inference-batch-size 16
```

Add `--push` only after you review the local sidecars and the manifest. The command uses the V2 dataset repository that is the CLI default. You can use `--repo-id` for an intentional override. The value `--inference-batch-size 16` is a safe first value for an 8 GB MacBook Air M2. Decrease it if the model cache or other applications need more memory.

## Outputs and resumability

For each finalized region, the command reads the existing section table in bounded Parquet batches. It writes these files:

- `wikipedia/sentences/<stem>.parquet`.
- `wikivoyage/sentences/<stem>.parquet`, when a Wikivoyage section table exists.
- `manifests/sentence_splitting.json`. It has the model, the revision, the supported routing, the observed unsupported languages, and the counts for each table.

The sentence rows keep the context of the source section. They add these fields: `sentence_index`, `start_char`, `end_char`, the sentence text, the length metrics, the content hashes, `segmenter`, `segmenter_version`, `model_id`, and `segmentation_status`.

A supported row has `segmentation_status=split`. An unsupported row has `segmentation_status=unsupported_language`. It covers the complete source section, from character 0 to the length of the section. The sentence pieces must rebuild the source section exactly, with the whitespace.

The command writes each source batch atomically to a restart state. The restart state is under the external data root. It records the hash of the source file, the batch size, the model identifier, and the model revision. If you change one of these inputs, a new contract starts. After a stop, the command reuses the valid completed batches. It never rewrites the source section tables. The command publishes the final sentence Parquet file and the routing manifest only after it completes all the source batches.

The generated V2 dataset card repeats this policy. A published snapshot thus shows the boundary of the supported languages and the explicit unsplit treatment.
