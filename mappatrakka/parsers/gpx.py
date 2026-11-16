"""GPX 1.0 / 1.1.

Each <trkseg> becomes its own segment - in GPX a new trkseg means the device
lost its fix or was paused, so the gap between them must not be drawn.
Routes (<rte>) are only used when the file has no track at all.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from mappatrakka.parsers import register
from mappatrakka.parsers.model import ParseError, SegmentBuilder, Track
from mappatrakka.parsers.xmlutil import iterparse, local, parse_time, root_tag, to_float


def parse_gpx(data: bytes) -> Track:
    track = Track()
    trk = SegmentBuilder()
    rte = SegmentBuilder()
    seen_root = False

    try:
        for event, elem in iterparse(data):
            tag = local(elem.tag)
            if event == "start":
                if not seen_root:
                    if tag != "gpx":
                        raise ParseError(f"not a GPX file (root element is <{tag}>)")
                    seen_root = True
                continue

            if tag in ("trkpt", "rtept"):
                lat = to_float(elem.get("lat"))
                lon = to_float(elem.get("lon"))
                when = None
                for child in elem:
                    if local(child.tag) == "time":
                        when = parse_time(child.text)
                        break
                (trk if tag == "trkpt" else rte).point(lat, lon, when)
                elem.clear()
            elif tag in ("trkseg", "trk"):
                trk.split()
                elem.clear()
            elif tag == "rte":
                rte.split()
                elem.clear()
            elif tag in ("wpt", "metadata"):
                elem.clear()
    except ET.ParseError as exc:
        if not seen_root:
            raise ParseError(f"invalid XML: {exc}") from None
        track.warnings.append(f"XML is broken part way through ({exc}); kept the points before it")

    segments = trk.finish()
    if not segments:
        segments = rte.finish()
    track.segments = segments
    return track


def _sniff(head: bytes) -> bool:
    return root_tag(head) == "gpx"


register("gpx", (".gpx",), parse_gpx, _sniff)
