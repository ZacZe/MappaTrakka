from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


class ParseError(Exception):
    """The file couldn't be read at all."""


@dataclass
class Segment:
    """A run of GPS points recorded without a break.

    `time` is seconds since the epoch, NaN where the file had no timestamp.
    """

    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray

    def __len__(self) -> int:
        return len(self.lat)


@dataclass
class Track:
    segments: list[Segment] = field(default_factory=list)
    # Anything that went wrong but didn't stop us getting points out.
    warnings: list[str] = field(default_factory=list)
    sport: str | None = None
    sub_sport: str | None = None

    @property
    def point_count(self) -> int:
        return sum(len(s) for s in self.segments)


class SegmentBuilder:
    """Collects points and turns them into Segments.

    Call `point()` for each fix and `split()` wherever the source format says
    a new segment begins.
    """

    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self._lat: list[float] = []
        self._lon: list[float] = []
        self._time: list[float] = []

    def point(self, lat: float | None, lon: float | None, t: float | None = None) -> None:
        if lat is None or lon is None:
            return
        self._lat.append(lat)
        self._lon.append(lon)
        self._time.append(math.nan if t is None else t)

    def split(self) -> None:
        if self._lat:
            self.segments.append(
                Segment(
                    np.asarray(self._lat, dtype=np.float64),
                    np.asarray(self._lon, dtype=np.float64),
                    np.asarray(self._time, dtype=np.float64),
                )
            )
            self._lat, self._lon, self._time = [], [], []

    def finish(self) -> list[Segment]:
        self.split()
        return self.segments
