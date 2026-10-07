# Colloquium decks

This directory holds two slide decks about the project. The rendered decks and their images are tracked in git and published with the documentation site:

- `dataset.html` explains the published data.
- `codebase.html` explains the software.
- `assets/coverage_map.png`, `assets/text_density.png` and `assets/text_presence.png` are the three map images that the decks use.

The Markdown sources `dataset.md` and `codebase.md` are tracked too. The docs site assembly (`scripts/assemble_docs_site.py`) copies the two HTML files and the three PNGs into the site under `presentations/`. The dataset page links to the published `dataset.html`.

## Regenerate the decks

The script generates the Markdown sources from the current local dataset snapshot:

```bash
OSM_POLYGON_DATA_ROOT=/path/to/data-root \
  .venv/bin/python -m presentations.build_decks
```

The script reads the generated snapshot of the dataset card and the manifest of the processed regions. It copies only the three map images that the decks need. It does not write a private storage path or a credential into the slides.

Then render the HTML with Colloquium 0.2.2. These commands write the HTML next to the sources, so the tracked files are updated:

```bash
colloquium build presentations/dataset.md -o presentations
colloquium build presentations/codebase.md -o presentations
```

Commit the updated Markdown, HTML and PNG files together.

## Local only

Only these two directories are local. They are ignored by git. Do not commit them. Do not push them.

- `output/` holds the PDF exports and other build products:

  ```bash
  colloquium export presentations/dataset.md -o presentations/output/dataset.pdf
  colloquium export presentations/codebase.md -o presentations/output/codebase.pdf
  ```

- `captures/` holds the slide captures:

  ```bash
  colloquium capture presentations/dataset.md -o presentations/captures/dataset
  colloquium capture presentations/codebase.md -o presentations/captures/codebase
  ```
