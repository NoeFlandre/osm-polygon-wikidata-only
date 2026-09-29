"""Small helpers shared by the offline and orchestration tests."""

from __future__ import annotations

import ipaddress
import struct
import zlib
from pathlib import Path
from typing import Any


def _socket_address_is_allowed(address: Any, *, allow_local_network: bool) -> bool:
    """Return whether a numeric loopback address is allowed for this test."""
    if not allow_local_network:
        return False
    host = address[0] if isinstance(address, (tuple, list)) and address else address
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    if not isinstance(host, str):
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def write_tiny_png(path: Path) -> Path:
    """Write a valid, deterministic 1x1 RGB PNG for orchestration tests."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
    png += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def _map_output_path(args: tuple[Any, ...], kwargs: dict[str, Any], position: int) -> Path:
    output_path = kwargs.get("output_path")
    return Path(output_path if output_path is not None else args[position])


def write_coverage_map_placeholder(*args: Any, **kwargs: Any) -> Path:
    """Match the coverage renderer's output contract without plotting."""
    return write_tiny_png(_map_output_path(args, kwargs, 2))


def write_count_map_placeholder(*args: Any, **kwargs: Any) -> Any:
    """Match the count renderer's result contract without plotting."""
    from osm_polygon_wikidata_only.hf._geographic.models import RenderResult

    output_path = write_tiny_png(_map_output_path(args, kwargs, 1))
    return RenderResult(output_path=output_path, caption=kwargs.get("caption", ""))


def write_publication_map_placeholders(
    *,
    data_root: Any,
    snapshot_stem: str,
    snapshots_dir: Path,
    world_land_warning: Any,
    text_snapshot: Any = None,
) -> tuple[Path, Path, Path]:
    """Write publication map placeholders without scanning dataset tables."""
    del data_root, world_land_warning, text_snapshot
    return (
        write_tiny_png(snapshots_dir / f"{snapshot_stem}-coverage_map.png"),
        write_tiny_png(snapshots_dir / f"{snapshot_stem}-geographic_text_presence.png"),
        write_tiny_png(snapshots_dir / f"{snapshot_stem}-geographic_text_density.png"),
    )


def skip_landmass_drawing(*_args: Any, **_kwargs: Any) -> None:
    """Avoid polygon work in tests whose contract does not inspect pixels."""
