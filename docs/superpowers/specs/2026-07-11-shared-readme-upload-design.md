# Shared README Upload Design

Augmentation uploads must update the dataset README in the same atomic commit as their sidecars. Both the main pipeline and augmentation pipeline will use one helper that renders a README snapshot from the current local manifest and processed Parquet inventory. This avoids divergent README implementations and lets either concurrently running pipeline publish a complete current snapshot.

The existing main-pipeline behavior remains unchanged except that its inline rendering moves into the shared helper. Augmentation uploads add the generated `README.md` to their existing atomic file list. No remote README read/merge is performed; last-writer ordering is safe because each snapshot is regenerated from the shared additive local state immediately before submission.

The repository README receives only concise documentation of the new augmentation layout and commands. Tests verify that the shared helper renders the canonical card and that augmentation upload lists include `README.md`.
