"""The whole run, start to finish.

1. read activities.csv and filter activities by type/name
2. parse every file (in parallel), clean it and summarise it
3. pick the map area from the summaries
4. draw every activity that crosses the area
5. colour it in and save
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from mappatrakka.archive import open_export
from mappatrakka.cache import TrackCache
from mappatrakka.colormaps import get_colormap
from mappatrakka.config import CleaningConfig, Config
from mappatrakka.export import Activity, read_activities, select_activities
from mappatrakka.parsers import ParseError, read_track
from mappatrakka.parsers.model import Track
from mappatrakka.processing import PrivacyZone, clean_track, line_length_m
from mappatrakka.projection import Bounds, PixelTransform, to_mercator
from mappatrakka.region import ActivitySummary, RegionChoice, choose_region
from mappatrakka.render import HeatmapCanvas, colourise, normalise

log = logging.getLogger(__name__)

# Called as progress(stage, done, total), e.g. ("Reading", 12, 340).
ProgressFn = Callable[[str, int, int], None]


class Progress:
    """Progress line on stderr, or forwarded to a callback (the web page)."""

    def __init__(self, total: int, label: str, enabled: bool = True, callback: ProgressFn | None = None):
        self.total = max(total, 1)
        self.label = label
        self.enabled = enabled and total > 0
        self.callback = callback
        self.tty = sys.stderr.isatty()
        self.done = 0
        self._last_pct = -1
        self._start = time.monotonic()
        if callback:
            callback(label, 0, total)

    def step(self, n: int = 1) -> None:
        self.done += n
        pct = int(100 * self.done / self.total)
        if self.callback and (pct != self._last_pct or self.done == self.total):
            self.callback(self.label, self.done, self.total)
        if self.enabled:
            if self.tty:
                if pct != self._last_pct or self.done == self.total:
                    sys.stderr.write(f"\r{self.label}: {self.done}/{self.total} ({pct}%)")
                    sys.stderr.flush()
            elif pct // 10 != self._last_pct // 10:
                sys.stderr.write(f"{self.label}: {self.done}/{self.total} ({pct}%)\n")
        self._last_pct = pct

    def close(self) -> None:
        if self.enabled and self.tty:
            elapsed = time.monotonic() - self._start
            sys.stderr.write(f"\r{self.label}: {self.done}/{self.total} done in {elapsed:.1f}s\n")


@dataclass
class _Job:
    index: int
    path: str
    cache_dir: str
    cleaning: CleaningConfig
    zones: list[PrivacyZone]
    excluded_sub_sports: frozenset[str]


@dataclass
class _Parsed:
    index: int
    summary: ActivitySummary | None = None
    skip_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    first_time: float | None = None
    last_time: float | None = None


def load_track(path: Path, cache: TrackCache) -> Track:
    track = cache.get(path)
    if track is None:
        track = read_track(path)
        cache.put(path, track)
    return track


def summarise(index: int, lines: list[np.ndarray]) -> ActivitySummary:
    pts = np.concatenate(lines)
    length = sum(line_length_m(line) for line in lines)
    x, y = to_mercator(pts[:, 0], pts[:, 1])
    return ActivitySummary(
        index=index,
        anchor_lat=float(np.median(pts[:, 0])),
        anchor_lon=float(np.median(pts[:, 1])),
        length_m=float(length),
        bounds=Bounds(float(x.min()), float(y.min()), float(x.max()), float(y.max())),
    )


def _process(job: _Job) -> _Parsed:
    """Parse, clean and summarise one file. Runs in a worker process."""
    out = _Parsed(job.index)
    try:
        track = load_track(Path(job.path), TrackCache(job.cache_dir))
    except ParseError as exc:
        out.skip_reason = "unreadable"
        out.warnings.append(str(exc))
        return out
    except Exception as exc:  # never let one bad file kill the run
        out.skip_reason = "unreadable"
        out.warnings.append(f"unexpected {type(exc).__name__}: {exc}")
        return out

    out.warnings.extend(track.warnings)
    if track.sub_sport and track.sub_sport in job.excluded_sub_sports:
        out.skip_reason = f"indoor ({track.sub_sport})"
        return out

    lines = clean_track(track, job.cleaning, job.zones)
    if not lines:
        out.skip_reason = "no usable GPS points"
        return out
    out.summary = summarise(job.index, lines)

    times = [s.time for s in track.segments if len(s)]
    if times:
        allt = np.concatenate(times)
        allt = allt[np.isfinite(allt)]
        if allt.size:
            out.first_time, out.last_time = float(allt.min()), float(allt.max())
    return out


def _worker_count(requested: int, jobs: int) -> int:
    if requested:
        return max(1, min(requested, jobs))
    if jobs < 16:
        return 1
    return max(1, min(8, (os.cpu_count() or 2) - 1, jobs))


def _run_jobs(jobs: list[_Job], workers: int) -> Iterator[_Parsed]:
    if workers <= 1:
        yield from map(_process, jobs)
        return
    # Imported here because multiprocessing isn't available everywhere (the
    # in-browser build runs single-process).
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # map() yields in submission order, which keeps everything downstream
        # deterministic no matter which worker finishes first.
        yield from pool.map(_process, jobs, chunksize=4)


@dataclass
class RunResult:
    image_path: Path
    region: RegionChoice
    transform: PixelTransform
    stats: dict
    # Why activities were left out, e.g. {"excluded type Virtual Ride": 4}.
    skipped: dict[str, int] = field(default_factory=dict)


def run(
    export: str | Path,
    cfg: Config,
    show_progress: bool = True,
    progress: ProgressFn | None = None,
) -> RunResult:
    """Make the heatmap. `export` is the export folder or the .zip itself."""
    with open_export(export) as export_dir:
        return _run(export_dir, cfg, show_progress, progress)


def _run(export_dir: Path, cfg: Config, show_progress: bool, progress: ProgressFn | None) -> RunResult:
    activities = read_activities(export_dir)
    log.info("Found %d activities in the export", len(activities))
    selected, skipped = select_activities(activities, cfg.filters)
    for reason, n in skipped.most_common():
        log.info("  skipped %d: %s", n, reason)
    if not selected:
        raise ValueError("no activities left after filtering - check the export and the filters")

    zones = [PrivacyZone(float(z["lat"]), float(z["lon"]), float(z["radius_m"])) for z in cfg.privacy.zones]
    excluded_sub = frozenset(s.lower() for s in cfg.filters.exclude_sub_sports)

    with ExitStack() as stack:
        if cfg.output.use_cache:
            cache_dir = Path(cfg.output.cache_dir)
        else:
            cache_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="mappatrakka-")))
        cache_dir.mkdir(parents=True, exist_ok=True)

        jobs = [
            _Job(i, str(a.path), str(cache_dir), cfg.cleaning, zones, excluded_sub)
            for i, a in enumerate(selected)
        ]
        workers = _worker_count(cfg.output.workers, len(jobs))
        log.info("Reading %d GPS files with %d worker(s)", len(jobs), workers)

        summaries: list[ActivitySummary] = []
        reasons: Counter[str] = Counter()
        first_times, last_times = [], []
        bar = Progress(len(jobs), "Reading", show_progress, progress)
        for res in _run_jobs(jobs, workers):
            act = selected[res.index]
            for w in res.warnings:
                log.warning("%s: %s", act.path.name if act.path else act.activity_id, w)
            if res.summary is None:
                reasons[res.skip_reason or "unknown"] += 1
            else:
                summaries.append(res.summary)
                if res.first_time is not None:
                    first_times.append(res.first_time)
                    last_times.append(res.last_time)
            bar.step()
        bar.close()
        for reason, n in reasons.most_common():
            log.info("  dropped %d: %s", n, reason)
        if not summaries:
            raise ValueError("none of the selected activities contain usable GPS data")

        r = cfg.render
        region = choose_region(summaries, cfg.region, r.width / r.height)
        log.info("Map area: %s", region.description)
        transform = PixelTransform(region.bounds, r.width, r.height)
        w_km, h_km = region.bounds.ground_size_km()
        log.info("Covers about %.1f x %.1f km (%.1f m per pixel)", w_km, h_km, transform.metres_per_pixel())

        canvas = HeatmapCanvas(transform, r.line_width, r.supersample)
        cache = TrackCache(cache_dir)
        visible = [s for s in summaries if s.bounds.intersects(region.bounds)]
        drawn: list[Activity] = []
        bar = Progress(len(visible), "Drawing", show_progress, progress)
        for s in visible:
            act = selected[s.index]
            try:
                track = load_track(act.path, cache)  # type: ignore[arg-type]
            except ParseError:
                bar.step()
                continue
            lines = clean_track(track, cfg.cleaning, zones)
            if canvas.add_activity([to_mercator(line[:, 0], line[:, 1]) for line in lines]):
                drawn.append(act)
            bar.step()
        bar.close()

    if progress:
        progress("Colouring", 0, 1)
    values, reference = normalise(canvas.counts, r)
    stats = _stats(selected, summaries, drawn, canvas, reference, region, first_times, last_times)
    del canvas
    cmap = get_colormap(r.colormap, r.colormap_stops)
    poster = colourise(values, cmap, r)
    del values

    image_path = Path(cfg.output.path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    poster.save(image_path)
    if progress:
        progress("Colouring", 1, 1)
    log.info("Saved %s (%dx%d)", image_path, poster.width, poster.height)

    all_skipped = Counter(skipped)
    all_skipped.update(reasons)
    return RunResult(image_path, region, transform, stats, dict(all_skipped.most_common()))


def _stats(
    selected: list[Activity],
    summaries: list[ActivitySummary],
    drawn: list[Activity],
    canvas: HeatmapCanvas,
    reference: float,
    region: RegionChoice,
    first_times: Iterable[float],
    last_times: Iterable[float],
) -> dict:
    by_index = {s.index: s for s in summaries}
    drawn_ids = {a.activity_id for a in drawn}
    drawn_km = sum(
        by_index[i].length_m for i, a in enumerate(selected) if a.activity_id in drawn_ids and i in by_index
    ) / 1000
    types = Counter((a.type or "Unknown") for a in drawn)
    first = min(first_times, default=None)
    last = max(last_times, default=None)
    w_km, h_km = region.bounds.ground_size_km()
    return {
        "activities_with_gps": len(summaries),
        "activities_drawn": len(drawn),
        "distance_km": round(drawn_km, 1),
        "types": dict(types.most_common()),
        "first_activity": time.strftime("%Y-%m-%d", time.gmtime(first)) if first else None,
        "last_activity": time.strftime("%Y-%m-%d", time.gmtime(last)) if last else None,
        "max_overlap": int(round(float(canvas.counts.max()))) if canvas.counts.size else 0,
        "brightness_reference": round(reference, 2),
        "area_km": [round(w_km, 1), round(h_km, 1)],
        "areas_found": len(region.clusters),
        "region": region.description,
    }
