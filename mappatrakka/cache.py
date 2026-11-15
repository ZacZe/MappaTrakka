"""On-disk cache of parsed tracks.

Parsing FIT files in pure Python is the slowest part of a run, and the map is
drawn in a second pass over the same files. Caching the decoded points means
that second pass (and every later run) skips the parsing, and we never need to
hold the whole export in memory.

Raw parser output is cached, before any cleaning, so changing cleaning or
render settings never needs a re-parse. Entries are keyed on path, size and
modification time, plus a version number bumped whenever parser output
changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from mappatrakka.parsers.model import Segment, Track

CACHE_VERSION = 1


class TrackCache:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def _key(self, path: Path) -> str | None:
        try:
            st = path.stat()
        except OSError:
            return None
        raw = f"{CACHE_VERSION}|{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _file(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.npz"

    def get(self, path: Path) -> Track | None:
        key = self._key(path)
        if key is None:
            return None
        f = self._file(key)
        if not f.is_file():
            return None
        try:
            with np.load(f, allow_pickle=False) as z:
                lat, lon, t, offsets = z["lat"], z["lon"], z["time"], z["offsets"]
                meta = json.loads(str(z["meta"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None  # corrupt entry, just re-parse
        segments = [
            Segment(lat[a:b], lon[a:b], t[a:b]) for a, b in zip(offsets[:-1], offsets[1:])
        ]
        return Track(
            segments=segments,
            warnings=meta.get("warnings", []),
            sport=meta.get("sport"),
            sub_sport=meta.get("sub_sport"),
        )

    def put(self, path: Path, track: Track) -> None:
        key = self._key(path)
        if key is None:
            return
        f = self._file(key)
        f.parent.mkdir(parents=True, exist_ok=True)
        if track.segments:
            lat = np.concatenate([s.lat for s in track.segments])
            lon = np.concatenate([s.lon for s in track.segments])
            t = np.concatenate([s.time for s in track.segments])
        else:
            lat = lon = t = np.zeros(0)
        offsets = np.concatenate(([0], np.cumsum([len(s) for s in track.segments], dtype=np.int64)))
        meta = json.dumps(
            {"warnings": track.warnings, "sport": track.sport, "sub_sport": track.sub_sport}
        )
        # Write to a temp file and rename so a crash never leaves half a file.
        fd, tmp = tempfile.mkstemp(dir=f.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, lat=lat, lon=lon, time=t, offsets=offsets, meta=np.array(meta))
            os.replace(tmp, f)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
