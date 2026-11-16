"""Web Mercator projection and map bounds.

Mercator is conformal: at any spot, north-south and east-west distances get
the same scale, so roads keep their real shape and angles and a city doesn't
look squashed. The scale itself grows towards the poles (x1.6 at 52 deg N,
x2 at 60 deg). Over a city-sized map that change is tiny, but it means very
large maps exaggerate the northern part, and points past ~85 deg can't be
shown at all.

It's also the projection web maps use, so the image lines up with standard
map tiles if you ever want to lay it over one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_WEB = 6_378_137.0
MAX_LAT = 85.05112878


def to_mercator(lat, lon):
    lat = np.clip(np.asarray(lat, dtype=np.float64), -MAX_LAT, MAX_LAT)
    lon = np.asarray(lon, dtype=np.float64)
    x = EARTH_RADIUS_WEB * np.radians(lon)
    y = EARTH_RADIUS_WEB * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def from_mercator(x, y):
    lon = np.degrees(np.asarray(x, dtype=np.float64) / EARTH_RADIUS_WEB)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y, dtype=np.float64) / EARTH_RADIUS_WEB)) - np.pi / 2)
    return lat, lon


def scale_factor(lat: float) -> float:
    """How many Mercator metres one ground metre becomes at this latitude."""
    return 1.0 / math.cos(math.radians(lat))


@dataclass(frozen=True)
class Bounds:
    """A rectangle in Web Mercator metres."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> tuple[float, float]:
        return (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2

    @classmethod
    def from_latlon(cls, south: float, west: float, north: float, east: float) -> Bounds:
        x0, y0 = to_mercator(south, west)
        x1, y1 = to_mercator(north, east)
        return cls(float(x0), float(y0), float(x1), float(y1))

    def to_latlon(self) -> tuple[float, float, float, float]:
        """(south, west, north, east) in degrees."""
        s, w = from_mercator(self.xmin, self.ymin)
        n, e = from_mercator(self.xmax, self.ymax)
        return float(s), float(w), float(n), float(e)

    def center_latlon(self) -> tuple[float, float]:
        lat, lon = from_mercator(*self.center)
        return float(lat), float(lon)

    def padded(self, fraction: float) -> Bounds:
        dx, dy = self.width * fraction, self.height * fraction
        return Bounds(self.xmin - dx, self.ymin - dy, self.xmax + dx, self.ymax + dy)

    def at_least(self, min_size: float) -> Bounds:
        """Grow a degenerate box (a single point, a straight N-S line...)."""
        cx, cy = self.center
        w = max(self.width, min_size) / 2
        h = max(self.height, min_size) / 2
        return Bounds(cx - w, cy - h, cx + w, cy + h)

    def with_aspect(self, aspect: float) -> Bounds:
        """Grow the shorter side so width / height == aspect. Never crops."""
        cx, cy = self.center
        w, h = self.width, self.height
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        return Bounds(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def intersects(self, other: Bounds) -> bool:
        return not (
            other.xmax < self.xmin
            or other.xmin > self.xmax
            or other.ymax < self.ymin
            or other.ymin > self.ymax
        )

    def ground_size_km(self) -> tuple[float, float]:
        """Approximate real-world width and height at the centre latitude."""
        lat, _ = self.center_latlon()
        k = scale_factor(lat)
        return self.width / k / 1000, self.height / k / 1000


@dataclass(frozen=True)
class PixelTransform:
    """Maps Mercator metres to image pixels (origin top-left, y down)."""

    bounds: Bounds
    width: int
    height: int

    @property
    def scale(self) -> float:
        return self.width / self.bounds.width

    def __call__(self, x, y):
        s = self.scale
        px = (np.asarray(x) - self.bounds.xmin) * s
        py = (self.bounds.ymax - np.asarray(y)) * self.height / self.bounds.height
        return px, py

    def metres_per_pixel(self) -> float:
        lat, _ = self.bounds.center_latlon()
        return self.bounds.width / self.width / scale_factor(lat)
