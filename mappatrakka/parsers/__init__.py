"""GPS file parsing.

Each format registers a parser (bytes -> Track) and a sniffer that recognises
the format from the first few bytes. Adding a format is one `register()` call.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mappatrakka.parsers.model import ParseError, Segment, SegmentBuilder, Track

log = logging.getLogger(__name__)

ParserFn = Callable[[bytes], Track]
SniffFn = Callable[[bytes], bool]


@dataclass(frozen=True)
class _Format:
    name: str
    extensions: tuple[str, ...]
    parse: ParserFn
    sniff: SniffFn


_FORMATS: dict[str, _Format] = {}


def register(name: str, extensions: tuple[str, ...], parse: ParserFn, sniff: SniffFn) -> None:
    _FORMATS[name] = _Format(name, extensions, parse, sniff)


def supported_formats() -> list[str]:
    return sorted(_FORMATS)


def _gunzip(data: bytes, warnings: list[str]) -> bytes:
    """Decompress gzip data, keeping whatever we can from a truncated file."""
    d = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
    try:
        out = d.decompress(data)
        out += d.flush()
    except zlib.error as exc:
        # Decompress whatever comes before the damage.
        d = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
        out = b""
        step = 64 * 1024
        for i in range(0, len(data), step):
            try:
                out += d.decompress(data[i : i + step])
            except zlib.error:
                break
        if not out:
            raise ParseError(f"corrupt gzip data ({exc})") from None
        warnings.append(f"gzip stream is damaged, using the first {len(out)} bytes")
        return out
    if not d.eof:
        warnings.append("gzip stream ends early, file is probably truncated")
    return out


def detect_format(data: bytes, filename: str = "") -> str | None:
    """Work out the format from the content, falling back to the extension."""
    head = data[:4096]
    for fmt in _FORMATS.values():
        if fmt.sniff(head):
            return fmt.name
    name = filename.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    for fmt in _FORMATS.values():
        if name.endswith(fmt.extensions):
            return fmt.name
    return None


def parse_bytes(data: bytes, filename: str = "") -> Track:
    warnings: list[str] = []
    if data[:2] == b"\x1f\x8b":
        data = _gunzip(data, warnings)
        if filename.lower().endswith(".gz"):
            filename = filename[:-3]
    if not data.strip():
        raise ParseError("file is empty")

    fmt = detect_format(data, filename)
    if fmt is None:
        raise ParseError("unrecognised file format")
    track = _FORMATS[fmt].parse(data)
    track.warnings[:0] = warnings
    return track


def read_track(path: Path) -> Track:
    """Read a GPS file of any supported format, gzipped or not."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ParseError(f"can't read file: {exc}") from None
    return parse_bytes(data, Path(path).name)


# Registration happens on import of each module.
from mappatrakka.parsers import fit, gpx, tcx  # noqa: E402,F401

__all__ = [
    "ParseError",
    "Segment",
    "SegmentBuilder",
    "Track",
    "detect_format",
    "parse_bytes",
    "read_track",
    "register",
    "supported_formats",
]
