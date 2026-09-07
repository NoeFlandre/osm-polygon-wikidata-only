"""Exercise Linux filename spelling on a case-insensitive development volume."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest


@pytest.fixture
def case_sensitive_paths(tmp_path, monkeypatch):
    """Keep real files/Parquet, but require exact spelling for Path reads."""

    def exact(path):
        try:
            relative = path.relative_to(tmp_path)
        except ValueError:
            return True
        current = tmp_path
        for part in relative.parts:
            try:
                names = {entry.name for entry in current.iterdir()}
            except OSError:
                return False
            if part not in names:
                return False
            current /= part
        return True

    def checked(original):
        def query(path, *args, **kwargs):
            return exact(path) and original(path, *args, **kwargs)

        return query

    for name in ("exists", "is_file", "is_dir"):
        monkeypatch.setattr(Path, name, checked(getattr(Path, name)))
    original_open = Path.open

    def open_exact(path, mode="r", *args, **kwargs):
        if "r" in mode and not exact(path):
            raise FileNotFoundError(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_exact)
    original_parquet_file = pq.ParquetFile

    def parquet_file_exact(source, *args, **kwargs):
        if isinstance(source, (str, Path)) and not exact(Path(source)):
            raise FileNotFoundError(source)
        return original_parquet_file(source, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", parquet_file_exact)
