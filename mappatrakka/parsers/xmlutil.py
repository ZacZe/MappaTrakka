"""Small helpers shared by the GPX and TCX parsers."""

from __future__ import annotations

import io
import math
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Iterator


def local(tag: str) -> str:
    """Strip the namespace: '{http://...}trkpt' -> 'trkpt'."""
    return tag.rsplit("}", 1)[-1]


def clean_xml(data: bytes) -> bytes:
    # Strava's TCX files famously start with a run of spaces before <?xml,
    # which ElementTree refuses to parse. Drop that and any BOM.
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.lstrip()


def root_tag(head: bytes) -> str | None:
    """Name of the first element in an XML snippet, without parsing it fully."""
    text = clean_xml(head)
    i = 0
    while True:
        i = text.find(b"<", i)
        if i < 0 or i + 1 >= len(text):
            return None
        nxt = text[i + 1 : i + 2]
        if nxt in (b"?", b"!"):
            i += 1
            continue
        end = i + 1
        while end < len(text) and text[end : end + 1] not in (b" ", b">", b"/", b"\n", b"\r", b"\t"):
            end += 1
        return local(text[i + 1 : end].decode("utf-8", "replace")).split(":")[-1]


def iterparse(data: bytes) -> Iterator[tuple[str, ET.Element]]:
    return ET.iterparse(io.BytesIO(clean_xml(data)), events=("start", "end"))


def parse_time(text: str | None) -> float | None:
    if not text:
        return None
    text = text.strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def to_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None
