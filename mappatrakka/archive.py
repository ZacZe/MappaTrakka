"""Reading a Strava export straight from the .zip Strava emails you.

The zip also holds every photo and video you've uploaded, which can run to
gigabytes, so only activities.csv and the GPS files are pulled out.

Member names are never trusted as paths - everything is written under the
target folder by its bare file name, so a crafted zip can't write elsewhere
("zip slip").
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterator

from mappatrakka.export import is_gps_file

log = logging.getLogger(__name__)


def _safe_name(member: str) -> str | None:
    name = PurePosixPath(member.replace("\\", "/")).name
    if not name or name in (".", "..") or name.startswith("."):
        return None
    return name


def extract_export(zip_path: str | Path, dest: str | Path) -> Path:
    """Unpack the useful parts of an export zip into `dest`.

    Returns the folder to treat as the export root: it gets activities.csv (if
    the zip had one) and an activities/ folder with every GPS file.
    """
    dest = Path(dest)
    activities = dest / "activities"
    activities.mkdir(parents=True, exist_ok=True)

    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"can't open {zip_path} as a zip file: {exc}") from None

    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        # The export is sometimes wrapped in a top-level folder; take the
        # activities.csv closest to the root.
        csvs = sorted(
            (m for m in members if PurePosixPath(m.filename).name.lower() == "activities.csv"),
            key=lambda m: m.filename.count("/"),
        )
        if csvs:
            with zf.open(csvs[0]) as src, (dest / "activities.csv").open("wb") as out:
                shutil.copyfileobj(src, out)

        count = 0
        for m in members:
            name = _safe_name(m.filename)
            if name is None or not is_gps_file(Path(name)):
                continue
            target = activities / name
            if target.exists():
                log.debug("duplicate file name in zip, keeping the first: %s", m.filename)
                continue
            try:
                with zf.open(m) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
            except (zipfile.BadZipFile, OSError, EOFError) as exc:
                log.warning("couldn't extract %s from the zip: %s", m.filename, exc)
                target.unlink(missing_ok=True)
                continue
            count += 1
    log.info("Extracted %d GPS files from %s", count, Path(zip_path).name)
    return dest


@contextlib.contextmanager
def open_export(path: str | Path) -> Iterator[Path]:
    """Yield an export folder, unpacking a zip to a temp folder first if needed."""
    path = Path(path)
    if path.is_file() and zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory(prefix="mappatrakka-export-") as tmp:
            yield extract_export(path, tmp)
    elif path.is_dir():
        yield path
    elif path.exists():
        raise ValueError(f"{path} is neither a folder nor a zip file")
    else:
        raise FileNotFoundError(f"export not found: {path}")
