"""Garmin FIT files, decoded with fitdecode.

Positions are stored as 32-bit "semicircles": degrees = value * 180 / 2**31.
We ask fitdecode for the raw integers and convert them ourselves so there's no
doubt about which unit we're holding.

A timer stop event ends the current segment. Records keep coming after a
start event, and if the rider moved while stopped (on a train, say) that
movement must not be drawn.
"""

from __future__ import annotations

import io
import warnings
from datetime import datetime

import fitdecode

from mappatrakka.parsers import register
from mappatrakka.parsers.model import ParseError, SegmentBuilder, Track

SEMICIRCLE_TO_DEG = 180.0 / 2**31
_INVALID_SINT32 = 0x7FFFFFFF
_STOP_EVENTS = {"stop", "stop_all", "stop_disable", "stop_disable_all"}


def semicircles_to_degrees(value: int | None) -> float | None:
    if value is None or value == _INVALID_SINT32 or value == -_INVALID_SINT32 - 1:
        return None
    return value * SEMICIRCLE_TO_DEG


def _timestamp(value: object) -> float | None:
    if isinstance(value, datetime):
        return value.timestamp()
    return None


def _value(frame: fitdecode.FitDataMessage, name: str, raw: bool = False):
    try:
        return frame.get_value(name, fallback=None, raw_value=raw)
    except (KeyError, ValueError, TypeError):
        return None


def _enum(value: object) -> str | None:
    if value is None:
        return None
    return str(value).lower()


def parse_fit(data: bytes) -> Track:
    track = Track()
    builder = SegmentBuilder()
    got_header = False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            reader = fitdecode.FitReader(
                io.BytesIO(data),
                check_crc=fitdecode.CrcCheck.WARN,
                error_handling=fitdecode.ErrorHandling.WARN,
            )
            for frame in reader:
                if isinstance(frame, fitdecode.FitHeader):
                    got_header = True
                    continue
                if not isinstance(frame, fitdecode.FitDataMessage):
                    continue

                if frame.name == "record":
                    lat = semicircles_to_degrees(_value(frame, "position_lat", raw=True))
                    lon = semicircles_to_degrees(_value(frame, "position_long", raw=True))
                    builder.point(lat, lon, _timestamp(_value(frame, "timestamp")))
                elif frame.name == "event":
                    if (
                        _enum(_value(frame, "event")) == "timer"
                        and _enum(_value(frame, "event_type")) in _STOP_EVENTS
                    ):
                        builder.split()
                elif frame.name in ("session", "sport"):
                    track.sport = track.sport or _enum(_value(frame, "sport"))
                    track.sub_sport = track.sub_sport or _enum(_value(frame, "sub_sport"))
        except fitdecode.FitError as exc:
            if not got_header:
                raise ParseError(f"not a readable FIT file: {exc}") from None
            track.warnings.append(f"FIT data is damaged ({exc}); kept the records before it")
        except (ValueError, IndexError, KeyError, TypeError, EOFError) as exc:
            # fitdecode can trip over really mangled data in odd ways.
            if not got_header:
                raise ParseError(f"not a readable FIT file: {exc}") from None
            track.warnings.append(f"FIT decoder failed part way ({type(exc).__name__}: {exc})")

    for w in caught:
        msg = str(w.message)
        if msg not in track.warnings:
            track.warnings.append(msg)

    track.segments = builder.finish()
    if track.sub_sport in ("generic", "all"):
        track.sub_sport = None
    return track


def _sniff(head: bytes) -> bool:
    # 12 or 14 byte header with ".FIT" at offset 8.
    return len(head) >= 12 and head[0] in (12, 14) and head[8:12] == b".FIT"


register("fit", (".fit",), parse_fit, _sniff)
