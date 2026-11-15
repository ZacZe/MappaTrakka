"""Turning raw parsed tracks into clean polylines ready for drawing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mappatrakka.config import CleaningConfig
from mappatrakka.parsers.model import Segment, Track
from mappatrakka.projection import MAX_LAT

EARTH_RADIUS_M = 6_371_008.8


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres. Works element-wise on arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class PrivacyZone:
    lat: float
    lon: float
    radius_m: float


def _runs(keep: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs for each run of True values."""
    if not keep.any():
        return []
    padded = np.concatenate(([False], keep, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def _split_at(breaks: np.ndarray, n: int) -> list[tuple[int, int]]:
    """Split range(n) before every index i where breaks[i - 1] is True.

    `breaks` has n - 1 entries, one per step between consecutive points.
    """
    cut = np.flatnonzero(breaks) + 1
    starts = np.concatenate(([0], cut))
    stops = np.concatenate((cut, [n]))
    return list(zip(starts, stops))


def _remove_spikes(lat, lon, t, cfg: CleaningConfig) -> np.ndarray:
    """Mask of points to keep after removing single-point GPS spikes.

    A spike is a point that jumps away from its neighbours and straight back,
    i.e. both legs are implausibly long but skipping it is short.
    """
    n = len(lat)
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep
    d_prev = haversine(lat[:-2], lon[:-2], lat[1:-1], lon[1:-1])
    d_next = haversine(lat[1:-1], lon[1:-1], lat[2:], lon[2:])
    d_skip = haversine(lat[:-2], lon[:-2], lat[2:], lon[2:])

    dt_prev = t[1:-1] - t[:-2]
    dt_next = t[2:] - t[1:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        timed = np.isfinite(dt_prev) & np.isfinite(dt_next) & (dt_prev > 0) & (dt_next > 0)
        fast = (d_prev / dt_prev > cfg.max_speed_mps) & (d_next / dt_next > cfg.max_speed_mps)
    far = (d_prev > cfg.spike_distance_m) & (d_next > cfg.spike_distance_m)
    # Moving more than a few metres either way (rules out noise on a stationary
    # rider being labelled a spike).
    moved = (d_prev > 20) & (d_next > 20)
    bounces_back = d_skip < 0.5 * np.minimum(d_prev, d_next)

    spike = moved & bounces_back & np.where(timed, fast, far)
    keep[1:-1] = ~spike
    return keep


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    half = window // 2
    padded = np.pad(values, (half, window - 1 - half), mode="edge")
    kernel = np.full(window, 1.0 / window)
    out = np.convolve(padded, kernel, mode="valid")
    # Keep the real endpoints so routes still start and finish where they did.
    out[0], out[-1] = values[0], values[-1]
    return out


def _in_zones(lat, lon, zones: list[PrivacyZone]) -> np.ndarray:
    hidden = np.zeros(len(lat), dtype=bool)
    for z in zones:
        hidden |= haversine(lat, lon, z.lat, z.lon) <= z.radius_m
    return hidden


def clean_segment(
    seg: Segment, cfg: CleaningConfig, zones: list[PrivacyZone] | None = None
) -> list[np.ndarray]:
    """Clean one raw segment. Returns zero or more (N, 2) [lat, lon] arrays."""
    lat, lon, t = seg.lat, seg.lon, seg.time
    if len(lat) == 0:
        return []

    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (np.abs(lat) <= MAX_LAT)
        & (np.abs(lon) <= 180)
        # (0, 0) is what a lot of devices write before they get a fix.
        & ~((lat == 0) & (lon == 0))
    )
    pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if zones:
        hidden = _in_zones(np.where(valid, lat, 0), np.where(valid, lon, 0), zones)
        # A privacy zone is a hard break: never join the line across it.
        for a, b in _runs(~hidden):
            v = valid[a:b]
            pieces.append((lat[a:b][v], lon[a:b][v], t[a:b][v]))
    else:
        pieces.append((lat[valid], lon[valid], t[valid]))

    out: list[np.ndarray] = []
    for plat, plon, pt in pieces:
        out.extend(_clean_piece(plat, plon, pt, cfg))
    return out


def _clean_piece(lat, lon, t, cfg: CleaningConfig) -> list[np.ndarray]:
    if len(lat) < 2:
        return []

    # Drop consecutive duplicate fixes.
    same = np.concatenate(([False], (lat[1:] == lat[:-1]) & (lon[1:] == lon[:-1])))
    if same.any():
        lat, lon, t = lat[~same], lon[~same], t[~same]

    keep = _remove_spikes(lat, lon, t, cfg)
    if not keep.all():
        lat, lon, t = lat[keep], lon[keep], t[keep]
    if len(lat) < 2:
        return []

    step = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = np.diff(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        too_fast = np.isfinite(dt) & (dt > 0) & (step / dt > cfg.max_speed_mps) & (step > 50)
    breaks = (
        (step > cfg.max_gap_m)
        | too_fast
        # Crossing the antimeridian - drawing it would wrap around the world.
        | (np.abs(np.diff(lon)) > 180)
    )

    out = []
    for a, b in _split_at(breaks, len(lat)):
        if b - a < cfg.min_segment_points:
            continue
        slat = _smooth(lat[a:b], cfg.smoothing_window)
        slon = _smooth(lon[a:b], cfg.smoothing_window)
        out.append(np.column_stack((slat, slon)))
    return out


def clean_track(
    track: Track, cfg: CleaningConfig, zones: list[PrivacyZone] | None = None
) -> list[np.ndarray]:
    lines: list[np.ndarray] = []
    for seg in track.segments:
        lines.extend(clean_segment(seg, cfg, zones))
    return lines


def line_length_m(line: np.ndarray) -> float:
    if len(line) < 2:
        return 0.0
    return float(haversine(line[:-1, 0], line[:-1, 1], line[1:, 0], line[1:, 1]).sum())
