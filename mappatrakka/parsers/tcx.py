"""Garmin Training Center XML.

Laps follow each other directly in time, so trackpoints from consecutive laps
(and the multiple <Track> elements a device writes after a pause) are joined
into one segment. The gap check in processing still splits it if the device
moved a long way while paused. Trackpoints without a <Position> (heart rate
only, indoor bits) are skipped.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from mappatrakka.parsers import register
from mappatrakka.parsers.model import ParseError, SegmentBuilder, Track
from mappatrakka.parsers.xmlutil import iterparse, local, parse_time, root_tag, to_float


def parse_tcx(data: bytes) -> Track:
    track = Track()
    builder = SegmentBuilder()
    seen_root = False

    try:
        for event, elem in iterparse(data):
            tag = local(elem.tag)
            if event == "start":
                if not seen_root:
                    if tag != "TrainingCenterDatabase":
                        raise ParseError(f"not a TCX file (root element is <{tag}>)")
                    seen_root = True
                elif tag == "Activity" and track.sport is None:
                    sport = elem.get("Sport")
                    track.sport = sport.lower() if sport else None
                continue

            if tag == "Trackpoint":
                lat = lon = when = None
                for child in elem:
                    name = local(child.tag)
                    if name == "Time":
                        when = parse_time(child.text)
                    elif name == "Position":
                        for coord in child:
                            cname = local(coord.tag)
                            if cname == "LatitudeDegrees":
                                lat = to_float(coord.text)
                            elif cname == "LongitudeDegrees":
                                lon = to_float(coord.text)
                builder.point(lat, lon, when)
                elem.clear()
            elif tag in ("Activity", "Course"):
                builder.split()
    except ET.ParseError as exc:
        if not seen_root:
            raise ParseError(f"invalid XML: {exc}") from None
        track.warnings.append(f"XML is broken part way through ({exc}); kept the points before it")

    track.segments = builder.finish()
    return track


def _sniff(head: bytes) -> bool:
    return root_tag(head) == "TrainingCenterDatabase"


register("tcx", (".tcx",), parse_tcx, _sniff)
