"""Make a fake Strava export to try things out with.

    python -m mappatrakka.demo demo_export

It builds a jittered street grid around a made-up home, then generates
activities that mostly reuse a few favourite loops, plus some one-offs, a
couple of holiday rides far away, virtual rides, a manual entry, a corrupt
file and a missing file - the usual mess of a real export. Files come in all
three formats, some gzipped. The same seed always gives the same export.

The tests use the writers in here too.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import random
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# FIT writer (just enough of the format for activity files)

FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)
_CRC_TABLE = [
    0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
    0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400,
]

SPORTS = {"running": 1, "cycling": 2, "walking": 11, "hiking": 17, "generic": 0}
SUB_SPORTS = {"generic": 0, "road": 7, "indoor_cycling": 6, "virtual_activity": 58, "treadmill": 1}


def fit_crc(data: bytes, crc: int = 0) -> int:
    for byte in data:
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[byte & 0xF]
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[(byte >> 4) & 0xF]
    return crc


def degrees_to_semicircles(deg: float) -> int:
    return int(round(deg * 2**31 / 180.0))


def _fit_time(t: datetime) -> int:
    return int((t - FIT_EPOCH).total_seconds())


class FitWriter:
    """Writes FIT activity files: file_id, records, timer events, session."""

    # local message number -> (global number, [(field number, size, base type, struct fmt)])
    _DEFS = {
        0: (0, [(0, 1, 0x00, "B"), (1, 2, 0x84, "H"), (4, 4, 0x86, "I")]),  # file_id
        1: (20, [(253, 4, 0x86, "I"), (0, 4, 0x85, "i"), (1, 4, 0x85, "i")]),  # record
        2: (21, [(253, 4, 0x86, "I"), (0, 1, 0x00, "B"), (1, 1, 0x00, "B")]),  # event
        3: (18, [(253, 4, 0x86, "I"), (5, 1, 0x00, "B"), (6, 1, 0x00, "B")]),  # session
    }

    def __init__(self) -> None:
        self.body = bytearray()
        self._defined: set[int] = set()

    def _define(self, local: int) -> None:
        if local in self._defined:
            return
        glob, fields = self._DEFS[local]
        self.body += struct.pack("<BBBHB", 0x40 | local, 0, 0, glob, len(fields))
        for num, size, base, _ in fields:
            self.body += struct.pack("<BBB", num, size, base)
        self._defined.add(local)

    def _data(self, local: int, *values: int) -> None:
        self._define(local)
        fmt = "<B" + "".join(f[3] for f in self._DEFS[local][1])
        self.body += struct.pack(fmt, local, *values)

    def file_id(self, created: datetime) -> None:
        self._data(0, 4, 255, _fit_time(created))

    def record(self, t: datetime, lat: float | None, lon: float | None) -> None:
        invalid = 0x7FFFFFFF
        slat = invalid if lat is None else degrees_to_semicircles(lat)
        slon = invalid if lon is None else degrees_to_semicircles(lon)
        self._data(1, _fit_time(t), slat, slon)

    def timer(self, t: datetime, start: bool) -> None:
        self._data(2, _fit_time(t), 0, 0 if start else 4)  # event=timer, start / stop_all

    def session(self, t: datetime, sport: str = "cycling", sub_sport: str = "generic") -> None:
        self._data(3, _fit_time(t), SPORTS[sport], SUB_SPORTS[sub_sport])

    def to_bytes(self) -> bytes:
        header = struct.pack("<BBHI4s", 14, 0x20, 2132, len(self.body), b".FIT")
        header += struct.pack("<H", fit_crc(header))
        data = header + bytes(self.body)
        return data + struct.pack("<H", fit_crc(data))


# ---------------------------------------------------------------------------
# GPX / TCX writers

Point = tuple[float, float, datetime | None]


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_gpx(segments: list[list[Point]], name: str = "Activity") -> bytes:
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx creator="mappatrakka demo" version="1.1" xmlns="http://www.topografix.com/GPX/1/1">',
        f"  <trk><name>{name}</name>",
    ]
    for seg in segments:
        out.append("    <trkseg>")
        for lat, lon, t in seg:
            when = f"<time>{_iso(t)}</time>" if t else ""
            out.append(f'      <trkpt lat="{lat:.7f}" lon="{lon:.7f}">{when}</trkpt>')
        out.append("    </trkseg>")
    out += ["  </trk>", "</gpx>"]
    return "\n".join(out).encode("utf-8")


def write_tcx(laps: list[list[Point]], sport: str = "Biking") -> bytes:
    # Strava's own TCX exports start with indentation before the declaration,
    # so do the same here to keep the parser honest.
    out = [
        '          <?xml version="1.0" encoding="UTF-8"?>',
        '<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">',
        f'  <Activities><Activity Sport="{sport}">',
    ]
    for lap in laps:
        start = next((t for _, _, t in lap if t), None)
        out.append(f'    <Lap StartTime="{_iso(start) if start else ""}"><Track>')
        for lat, lon, t in lap:
            when = f"<Time>{_iso(t)}</Time>" if t else ""
            out.append(
                f"      <Trackpoint>{when}<Position><LatitudeDegrees>{lat:.7f}</LatitudeDegrees>"
                f"<LongitudeDegrees>{lon:.7f}</LongitudeDegrees></Position></Trackpoint>"
            )
        out.append("    </Track></Lap>")
    out += ["  </Activity></Activities>", "</TrainingCenterDatabase>"]
    return "\n".join(out).encode("utf-8")


def write_fit(
    segments: list[list[Point]], sport: str = "cycling", sub_sport: str = "generic"
) -> bytes:
    w = FitWriter()
    first = next((t for seg in segments for _, _, t in seg if t), datetime(2024, 1, 1, tzinfo=timezone.utc))
    w.file_id(first)
    last = first
    for seg in segments:
        if not seg:
            continue
        w.timer(seg[0][2] or last, start=True)
        for lat, lon, t in seg:
            last = t or last
            w.record(last, lat, lon)
        w.timer(last, start=False)
    w.session(last, sport, sub_sport)
    return w.to_bytes()


# ---------------------------------------------------------------------------
# Fake town


@dataclass
class Town:
    lat: float
    lon: float
    nodes: dict[tuple[int, int], tuple[float, float]]
    size: int


def make_town(lat: float, lon: float, size: int, spacing_m: float, rng: random.Random) -> Town:
    """A square street grid with wobbly junctions, centred on (lat, lon)."""
    dlat = spacing_m / 111_320
    dlon = spacing_m / (111_320 * math.cos(math.radians(lat)))
    nodes = {}
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            nodes[(i, j)] = (
                lat + (i + rng.uniform(-0.25, 0.25)) * dlat,
                lon + (j + rng.uniform(-0.25, 0.25)) * dlon,
            )
    return Town(lat, lon, nodes, size)


def _neighbours(town: Town, node: tuple[int, int]) -> list[tuple[int, int]]:
    i, j = node
    cand = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
    return [c for c in cand if c in town.nodes]


def random_loop(town: Town, steps: int, rng: random.Random) -> list[tuple[int, int]]:
    """Wander out from home, then head back the most direct way."""
    path = [(0, 0)]
    for _ in range(steps):
        nxt = _neighbours(town, path[-1])
        if len(path) > 1 and len(nxt) > 1:
            nxt = [n for n in nxt if n != path[-2]]
        path.append(rng.choice(nxt))
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i and (not j or rng.random() < 0.5):
            path.append((i - (1 if i > 0 else -1), j))
        else:
            path.append((i, j - (1 if j > 0 else -1)))
    return path


def path_points(
    town: Town, path: list[tuple[int, int]], start: datetime, speed_mps: float,
    rng: random.Random, every_s: float = 2.0, noise_m: float = 3.0,
) -> list[Point]:
    pts: list[Point] = []
    t = start
    noise_lat = noise_m / 111_320
    noise_lon = noise_m / (111_320 * math.cos(math.radians(town.lat)))
    for a, b in zip(path[:-1], path[1:]):
        (la1, lo1), (la2, lo2) = town.nodes[a], town.nodes[b]
        dist = math.hypot((la2 - la1) * 111_320, (lo2 - lo1) * 111_320 * math.cos(math.radians(la1)))
        n = max(1, int(dist / (speed_mps * every_s)))
        for k in range(n):
            f = k / n
            pts.append(
                (
                    la1 + (la2 - la1) * f + rng.gauss(0, noise_lat),
                    lo1 + (lo2 - lo1) * f + rng.gauss(0, noise_lon),
                    t,
                )
            )
            t += timedelta(seconds=every_s)
    last = town.nodes[path[-1]]
    pts.append((last[0], last[1], t))
    return pts


# ---------------------------------------------------------------------------
# The export itself

CSV_HEADER = [
    "Activity ID", "Activity Date", "Activity Name", "Activity Type", "Activity Description",
    "Elapsed Time", "Distance", "Max Heart Rate", "Relative Effort", "Commute",
    "Activity Private Note", "Activity Gear", "Filename", "Athlete Weight", "Bike Weight",
    "Elapsed Time", "Moving Time", "Distance", "Max Speed", "Commute",
]


def _row(aid: int, when: datetime, name: str, kind: str, filename: str, commute: bool = False) -> list[str]:
    date = when.strftime("%b %d, %Y, %I:%M:%S %p")
    c = "true" if commute else "false"
    return [str(aid), date, name, kind, "", "3600", "20.0", "", "", c, "", "", filename,
            "", "", "3600", "3500", "20000.0", "10.0", c]


def build_demo_export(
    out_dir: str | Path,
    activities: int = 120,
    seed: int = 7,
    home: tuple[float, float] = (52.3676, 4.9041),
) -> Path:
    rng = random.Random(seed)
    out = Path(out_dir)
    (out / "activities").mkdir(parents=True, exist_ok=True)

    town = make_town(home[0], home[1], size=18, spacing_m=450, rng=rng)
    favourites = [random_loop(town, rng.randint(25, 60), rng) for _ in range(6)]
    weights = [6, 4, 3, 2, 1, 1]

    rows = []
    t0 = datetime(2023, 3, 1, 7, 30, tzinfo=timezone.utc)
    aid = 1000

    def save(name: str, data: bytes, gz: bool) -> str:
        fname = f"activities/{name}" + (".gz" if gz else "")
        (out / fname).write_bytes(gzip.compress(data, mtime=0) if gz else data)
        return fname

    for n in range(activities):
        aid += 1
        start = t0 + timedelta(days=n * 2, hours=rng.randint(0, 10))
        kind = rng.choices(["Ride", "Run", "Walk"], weights=[6, 3, 1])[0]
        speed = {"Ride": 7.0, "Run": 3.2, "Walk": 1.4}[kind]
        if rng.random() < 0.7:
            path = rng.choices(favourites, weights=weights)[0]
        else:
            path = random_loop(town, rng.randint(15, 70), rng)
        if kind != "Ride":
            path = path[: max(8, len(path) // 3)]
            path = path + path[-2::-1]  # out and back
        pts = path_points(town, path, start, speed, rng)

        segments = [pts]
        if rng.random() < 0.15 and len(pts) > 40:
            # Paused, and then "teleported" (car, train) before resuming.
            cut = len(pts) // 2
            jump = 0.02
            second = [(la + jump, lo + jump, t) for la, lo, t in pts[cut:]]
            segments = [pts[:cut], second]

        fmt = ("fit", "gpx", "tcx")[n % 3]
        gz = n % 2 == 0
        if fmt == "fit":
            sport = {"Ride": "cycling", "Run": "running", "Walk": "walking"}[kind]
            data = write_fit(segments, sport)
        elif fmt == "gpx":
            data = write_gpx(segments, f"{kind} {n}")
        else:
            data = write_tcx(segments, "Biking" if kind == "Ride" else "Running")
        fname = save(f"{aid}.{fmt}", data, gz)
        rows.append(_row(aid, start, f"Morning {kind}", kind, fname, commute=rng.random() < 0.2))

    # Holiday rides ~1300 km away.
    away = make_town(39.57, 2.65, size=8, spacing_m=600, rng=rng)
    for k in range(3):
        aid += 1
        start = t0 + timedelta(days=200 + k)
        pts = path_points(away, random_loop(away, 40, rng), start, 7.0, rng)
        fname = save(f"{aid}.gpx", write_gpx([pts]), gz=True)
        rows.append(_row(aid, start, "Holiday ride", "Ride", fname))

    # Virtual rides: Zwift writes real-looking coordinates on a Pacific island.
    zwift = make_town(-11.64, 166.95, size=6, spacing_m=500, rng=rng)
    for k in range(4):
        aid += 1
        start = t0 + timedelta(days=30 + k * 7, hours=19)
        pts = path_points(zwift, random_loop(zwift, 30, rng), start, 9.0, rng)
        data = write_fit([pts], "cycling", "virtual_activity")
        fname = save(f"{aid}.fit", data, gz=True)
        rows.append(_row(aid, start, "Zwift - Watopia", "Virtual Ride", fname))

    aid += 1
    rows.append(_row(aid, t0 + timedelta(days=5), "Gym", "Weight Training", ""))
    aid += 1
    rows.append(_row(aid, t0 + timedelta(days=6), "Lost file", "Ride", "activities/99999.fit.gz"))
    aid += 1
    (out / "activities" / f"{aid}.fit").write_bytes(b"\x0e\x20garbage that is not a fit file")
    rows.append(_row(aid, t0 + timedelta(days=7), "Corrupt", "Ride", f"activities/{aid}.fit"))
    rows.append(list(rows[0]))  # duplicate row

    with (out / "activities.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate a fake Strava export for testing.")
    p.add_argument("out_dir", nargs="?", default="demo_export")
    p.add_argument("-n", "--activities", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    path = build_demo_export(args.out_dir, args.activities, args.seed)
    print(f"Wrote a demo export to {path}")


if __name__ == "__main__":
    main()
