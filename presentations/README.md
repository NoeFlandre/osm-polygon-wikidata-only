# Local Colloquium decks

This directory contains two local, generated slide decks:

- `output/dataset.html` and `output/dataset.pdf` explain the published data.
- `output/codebase.html` and `output/codebase.pdf` explain the software.

The Markdown sources are generated from the current local dataset snapshot:

```bash
OSM_POLYGON_DATA_ROOT=/path/to/data-root \
  .venv/bin/python -m presentations.build_decks
```

The script reads the generated dataset card snapshot and the processed-region
manifest. It copies only the three map images needed by the decks. No private
storage path or credential is written into the slides.

Colloquium 0.2.2 commands used for the rendered outputs are:

```bash
colloquium build presentations/dataset.md -o presentations/output
colloquium build presentations/codebase.md -o presentations/output
colloquium export presentations/dataset.md -o presentations/output/dataset.pdf
colloquium export presentations/codebase.md -o presentations/output/codebase.pdf
colloquium capture presentations/dataset.md -o presentations/captures/dataset
colloquium capture presentations/codebase.md -o presentations/captures/codebase
```

The generated files are intentionally local and are not committed or pushed.
