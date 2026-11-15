"""Picking which part of the world to draw.

A plain bounding box of every point gets ruined by one holiday abroad. Instead
each activity is reduced to an anchor (its median point), anchors are grouped
by single-linkage clustering, and the group with the most activities wins. The
map is sized to fit that group's activities, minus the odd one that reaches
much further than the rest, so a single epic ride doesn't zoom everything out.

Routes from *any* activity that cross the chosen area are still drawn.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np

from mappatrakka.config import RegionConfig
from mappatrakka.projection import Bounds, scale_factor, to_mercator

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088
# Smallest map we'll make, in ground metres. Stops a single short walk turning
# into a map of one street corner.
MIN_EXTENT_M = 1000.0


@dataclass
class ActivitySummary:
    """What region selection needs to know about one activity."""

    index: int
    anchor_lat: float
    anchor_lon: float
    length_m: float
    # Mercator bounding box of the cleaned route.
    bounds: Bounds


@dataclass
class Cluster:
    members: list[int]
    count: int
    length_m: float
    center_lat: float
    center_lon: float


@dataclass
class RegionChoice:
    bounds: Bounds
    clusters: list[Cluster] = field(default_factory=list)
    chosen: Cluster | None = None
    description: str = ""


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Smaller root wins, so labels don't depend on processing order.
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _unit_xyz_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return EARTH_RADIUS_KM * np.column_stack(
        (np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la))
    )


def _any_within(a: np.ndarray, b: np.ndarray, r2: float, block: int = 1024) -> bool:
    """Is any point of `a` within sqrt(r2) of any point of `b`?"""
    for i in range(0, len(a), block):
        chunk = a[i : i + block]
        d2 = ((chunk[:, None, :] - b[None, :block, :]) ** 2).sum(axis=2)
        if d2.min() <= r2:
            return True
        for j in range(block, len(b), block):
            d2 = ((chunk[:, None, :] - b[None, j : j + block, :]) ** 2).sum(axis=2)
            if d2.min() <= r2:
                return True
    return False


def cluster_points(lat: np.ndarray, lon: np.ndarray, radius_km: float) -> np.ndarray:
    """Single-linkage clustering: points closer than radius_km (directly or
    through a chain of other points) get the same label.

    Works in 3D so the poles and the antimeridian need no special handling.
    Points are bucketed into cubes small enough that everything inside one
    cube is already within range, so only neighbouring cubes need comparing.
    Labels are 0..k-1, numbered in order of first appearance.
    """
    n = len(lat)
    if n == 0:
        return np.zeros(0, dtype=int)
    xyz = _unit_xyz_km(np.asarray(lat, float), np.asarray(lon, float))
    # Chord distance slightly underestimates arc distance; at these scales the
    # difference is nanometres.
    cell = radius_km / np.sqrt(3)
    keys = np.floor(xyz / cell).astype(np.int64)

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    cell_keys = sorted(buckets)
    cell_index = {k: i for i, k in enumerate(cell_keys)}
    uf = _UnionFind(len(cell_keys))
    pts = {k: xyz[v] for k, v in buckets.items()}

    # A point within radius can be up to ceil(sqrt(3)) = 2 cells away per axis.
    offsets = [o for o in itertools.product(range(-2, 3), repeat=3) if o > (0, 0, 0)]
    r2 = radius_km * radius_km
    for k in cell_keys:
        a = pts[k]
        for o in offsets:
            nk = (k[0] + o[0], k[1] + o[1], k[2] + o[2])
            b = pts.get(nk)
            if b is None:
                continue
            ci, cj = cell_index[k], cell_index[nk]
            if uf.find(ci) == uf.find(cj):
                continue
            if _any_within(a, b, r2):
                uf.union(ci, cj)

    roots = np.array([uf.find(cell_index[tuple(k)]) for k in keys])
    _, first = np.unique(roots, return_index=True)
    order = {roots[i]: label for label, i in enumerate(sorted(first))}
    return np.array([order[r] for r in roots])


def find_clusters(summaries: list[ActivitySummary], radius_km: float) -> list[Cluster]:
    """Group activities by area, busiest first."""
    if not summaries:
        return []
    lat = np.array([s.anchor_lat for s in summaries])
    lon = np.array([s.anchor_lon for s in summaries])
    labels = cluster_points(lat, lon, radius_km)

    clusters = []
    for label in range(labels.max() + 1):
        idx = np.flatnonzero(labels == label)
        clusters.append(
            Cluster(
                members=[summaries[i].index for i in idx],
                count=len(idx),
                length_m=float(sum(summaries[i].length_m for i in idx)),
                center_lat=float(np.median(lat[idx])),
                center_lon=float(np.median(lon[idx])),
            )
        )
    # Most activities first; total distance breaks ties, then the earliest
    # activity so the result is stable.
    clusters.sort(key=lambda c: (-c.count, -c.length_m, min(c.members)))
    return clusters


def bounds_of(summaries: list[ActivitySummary], trim_percent: float = 0.0) -> Bounds:
    """Box around a set of activities, ignoring the few that reach furthest.

    On each side the `trim_percent` of activities that stick out furthest are
    ignored. Trimming whole activities rather than a share of the route
    distance means a road you've used many times is never cut off - it takes
    one or two lone long rides to push an edge out, and those are what get
    trimmed. Small exports (fewer than 100 / trim_percent activities) aren't
    trimmed at all.
    """
    xmin = np.sort([s.bounds.xmin for s in summaries])
    ymin = np.sort([s.bounds.ymin for s in summaries])
    xmax = np.sort([s.bounds.xmax for s in summaries])[::-1]
    ymax = np.sort([s.bounds.ymax for s in summaries])[::-1]
    k = int(len(summaries) * trim_percent / 100)
    return Bounds(float(xmin[k]), float(ymin[k]), float(xmax[k]), float(ymax[k]))


def _finish(bounds: Bounds, aspect: float, padding: float) -> Bounds:
    lat, _ = bounds.center_latlon()
    min_size = MIN_EXTENT_M * scale_factor(lat)
    return bounds.at_least(min_size).padded(padding).with_aspect(aspect)


def choose_region(
    summaries: list[ActivitySummary], cfg: RegionConfig, aspect: float
) -> RegionChoice:
    """Work out the map bounds (Mercator) according to the region settings."""
    if cfg.mode == "bbox":
        min_lon, min_lat, max_lon, max_lat = cfg.bbox  # type: ignore[misc]
        b = Bounds.from_latlon(min_lat, min_lon, max_lat, max_lon)
        out = b.with_aspect(aspect)
        if abs(b.width / b.height - aspect) > 0.01:
            log.info("Widened the bbox a little to match the %.2f image aspect ratio", aspect)
        return RegionChoice(out, description="fixed bounding box")

    if cfg.mode == "center":
        lat, lon = cfg.center  # type: ignore[misc]
        x, y = to_mercator(lat, lon)
        r = cfg.radius_km * 1000 * scale_factor(lat)
        b = Bounds(float(x - r), float(y - r), float(x + r), float(y + r)).with_aspect(aspect)
        return RegionChoice(b, description=f"{cfg.radius_km:g} km around {lat:.4f}, {lon:.4f}")

    if not summaries:
        raise ValueError("no activities with GPS data to build a map from")

    if cfg.mode == "all":
        b = _finish(bounds_of(summaries), aspect, cfg.padding)
        return RegionChoice(b, description="every activity")

    clusters = find_clusters(summaries, cfg.cluster_radius_km)
    rank = cfg.cluster_rank
    if rank > len(clusters):
        log.warning("Asked for area #%d but only found %d; using #1", rank, len(clusters))
        rank = 1
    chosen = clusters[rank - 1]
    members = set(chosen.members)
    b = _finish(bounds_of([s for s in summaries if s.index in members], cfg.trim_percent), aspect, cfg.padding)

    others = [c for c in clusters if c is not chosen]
    desc = (
        f"area #{rank} of {len(clusters)}: {chosen.count} activities around "
        f"{chosen.center_lat:.3f}, {chosen.center_lon:.3f}"
    )
    if others:
        desc += f" (left out {len(others)} other area(s) with {sum(c.count for c in others)} activities)"
    return RegionChoice(b, clusters=clusters, chosen=chosen, description=desc)
