# Local Colloquium decks

This directory has two generated slide decks. They are local:

- `output/dataset.html` and `output/dataset.pdf` explain the published data.
- `output/codebase.html` and `output/codebase.pdf` explain the software.

The script generates the Markdown sources from the current local dataset snapshot:

```bash
OSM_POLYGON_DATA_ROOT=/path/to/data-root \
  .venv/bin/python -m presentations.build_decks
```

The script reads the generated snapshot of the dataset card and the manifest of the processed regions. It copies only the three map images that the decks need. It does not write a private storage path or a credential into the slides.

These are the Colloquium 0.2.2 commands that make the rendered outputs:

```bash
colloquium build presentations/dataset.md -o presentations/output
colloquium build presentations/codebase.md -o presentations/output
colloquium export presentations/dataset.md -o presentations/output/dataset.pdf
colloquium export presentations/codebase.md -o presentations/output/codebase.pdf
colloquium capture presentations/dataset.md -o presentations/captures/dataset
colloquium capture presentations/codebase.md -o presentations/captures/codebase
```

The generated files are local on purpose. Do not commit them. Do not push them.
