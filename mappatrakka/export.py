"""Reading the Strava bulk export: activities.csv plus the activities/ folder."""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mappatrakka.config import FilterConfig

log = logging.getLogger(__name__)

GPS_SUFFIXES = (".fit", ".gpx", ".tcx")


@dataclass(frozen=True)
class Activity:
    activity_id: str
    name: str
    type: str
    date: str
    path: Path | None
    commute: bool = False
    # Filename as listed in activities.csv, even if the file wasn't found.
    filename: str = ""


def normalise_type(value: str) -> str:
    """'Virtual Ride' -> 'virtualride', 'E-Bike Ride' -> 'ebikeride'."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def is_gps_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return name.endswith(GPS_SUFFIXES)


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes")


class _Columns:
    """Column lookup by header name.

    Strava's CSV repeats some headers (Distance, Elapsed Time, Commute...), so
    csv.DictReader would quietly keep the *last* one. We want the first.
    """

    def __init__(self, header: list[str]):
        self.index: dict[str, int] = {}
        for i, name in enumerate(header):
            self.index.setdefault(name.strip().lower(), i)

    def get(self, row: list[str], name: str, default: str = "") -> str:
        i = self.index.get(name)
        if i is None or i >= len(row):
            return default
        return row[i].strip()

    def has(self, name: str) -> bool:
        return name in self.index


def _resolve(export_dir: Path, filename: str) -> Path | None:
    if not filename:
        return None
    # Strava writes forward slashes; be forgiving about backslashes too.
    candidate = export_dir / Path(filename.replace("\\", "/"))
    if candidate.is_file():
        return candidate
    # Some exports get unzipped with the files sitting next to the csv.
    alt = export_dir / Path(filename).name
    if alt.is_file():
        return alt
    alt = export_dir / "activities" / Path(filename).name
    if alt.is_file():
        return alt
    return None


def read_activities(export_dir: Path) -> list[Activity]:
    """Read every activity listed in the export.

    Falls back to scanning activities/ if there's no activities.csv. In that
    case nothing is known about activity types, so type filters can't apply.
    """
    export_dir = Path(export_dir)
    csv_path = export_dir / "activities.csv"
    if not csv_path.is_file():
        log.warning("No activities.csv in %s - using every GPS file in activities/ instead", export_dir)
        return _scan_folder(export_dir)

    activities: list[Activity] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    missing_name = missing_file = duplicates = 0

    with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            log.warning("activities.csv is empty")
            return []
        cols = _Columns(header)
        if not cols.has("filename"):
            log.warning("activities.csv has no Filename column - falling back to a folder scan")
            return _scan_folder(export_dir)

        for line_no, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue
            activity_id = cols.get(row, "activity id") or f"row{line_no}"
            if activity_id in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(activity_id)

            filename = cols.get(row, "filename")
            if not filename:
                missing_name += 1
                path = None
            else:
                path = _resolve(export_dir, filename)
                if path is None:
                    missing_file += 1
                    log.debug("activity %s: file %s not found", activity_id, filename)
                elif path.resolve() in seen_paths:
                    duplicates += 1
                    continue
                else:
                    seen_paths.add(path.resolve())

            activities.append(
                Activity(
                    activity_id=activity_id,
                    name=cols.get(row, "activity name"),
                    type=cols.get(row, "activity type"),
                    date=cols.get(row, "activity date"),
                    path=path,
                    commute=_truthy(cols.get(row, "commute", "false")),
                    filename=filename,
                )
            )

    if missing_name:
        log.info("%d activities have no GPS file (manual entries etc.)", missing_name)
    if missing_file:
        log.warning("%d activities point to files that aren't in the export", missing_file)
    if duplicates:
        log.info("Ignored %d duplicate rows", duplicates)
    return activities


def _scan_folder(export_dir: Path) -> list[Activity]:
    folder = export_dir / "activities"
    if not folder.is_dir():
        folder = export_dir
    files = sorted(p for p in folder.iterdir() if p.is_file() and is_gps_file(p))
    return [
        Activity(activity_id=p.name.split(".")[0], name=p.name, type="", date="", path=p)
        for p in files
    ]


def select_activities(
    activities: list[Activity], filters: FilterConfig
) -> tuple[list[Activity], Counter[str]]:
    """Apply the type/name filters. Returns (kept, reasons-for-skipping)."""
    include = {normalise_type(t) for t in filters.include_types}
    exclude = {normalise_type(t) for t in filters.exclude_types}
    patterns = [p.lower() for p in filters.exclude_name_patterns if p]

    kept: list[Activity] = []
    skipped: Counter[str] = Counter()
    for act in activities:
        kind = normalise_type(act.type)
        if act.path is None:
            skipped["GPS file missing from export" if act.filename else "no GPS file (manual entry)"] += 1
        # No type known (loose files without activities.csv): the type filters
        # can't judge these, so they're kept rather than silently dropped.
        elif include and kind and kind not in include:
            skipped[f"type {act.type or 'unknown'} not included"] += 1
        elif kind and kind in exclude:
            skipped[f"excluded type {act.type}"] += 1
        elif patterns and any(p in act.name.lower() for p in patterns):
            skipped["excluded by name"] += 1
        elif filters.exclude_commutes and act.commute:
            skipped["commute"] += 1
        elif not is_gps_file(act.path):
            skipped["unsupported file type"] += 1
        else:
            kept.append(act)
    return kept, skipped
