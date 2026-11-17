"""Glue for the in-browser version (see web/).

The page runs this package under Pyodide in a Web Worker. Uploaded files are
mounted read-only into the virtual filesystem and `generate()` is called with
the options from the form. Everything stays on the visitor's machine.

Kept free of any Pyodide imports so it can be tested with plain CPython.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from mappatrakka.config import config_from_dict
from mappatrakka.export import is_gps_file
from mappatrakka.pipeline import ProgressFn, run
from mappatrakka.render import shrink

MAX_WARNINGS = 50
# The page shows this while zoomed out and swaps in the full image on zoom.
PREVIEW_SIZE = 1600

# Form fields the page may send, and where they go in the config.
_OPTION_MAP = {
    "size": ("render", ("width", "height")),
    "colormap": ("render", ("colormap",)),
    "scale": ("render", ("scale",)),
    "line_width": ("render", ("line_width",)),
    "glow": ("render", ("glow_strength",)),
    "include_types": ("filters", ("include_types",)),
    "exclude_commutes": ("filters", ("exclude_commutes",)),
    "region": ("region", ("mode",)),
    "area": ("region", ("cluster_rank",)),
}


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.info: list[str] = []
        self.warnings: list[str] = []
        self.warning_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            self.warning_count += 1
            if len(self.warnings) < MAX_WARNINGS:
                self.warnings.append(msg)
        else:
            self.info.append(msg)


def build_config(options: dict[str, Any]) -> Any:
    """Turn the page's options into a validated Config (raises ConfigError)."""
    sections: dict[str, dict[str, Any]] = {}
    for key, value in options.items():
        if value is None or value == "" or key not in _OPTION_MAP:
            continue
        section, fields = _OPTION_MAP[key]
        if key == "include_types" and isinstance(value, str):
            value = [t.strip() for t in value.split(",") if t.strip()]
        for f in fields:
            sections.setdefault(section, {})[f] = value
    return config_from_dict(sections)


def find_export(root: str | Path) -> Path:
    """Work out what the visitor uploaded: a zip, or a folder of files."""
    root = Path(root)
    zips = [p for p in root.iterdir() if p.suffix.lower() == ".zip"]
    gps = [p for p in root.iterdir() if is_gps_file(p)]
    if len(zips) == 1 and not gps:
        return zips[0]
    if not gps and not (root / "activities.csv").exists():
        raise ValueError("no Strava export or GPS files found in what was uploaded")
    return root


def generate(upload_dir: str, options: dict[str, Any], progress: ProgressFn | None = None) -> dict[str, Any]:
    """Make a heatmap from uploaded files. Returns the PNG bytes and stats."""
    cfg = build_config(options)
    cfg.output.use_cache = False
    cfg.output.workers = 1

    handler = _Collect()
    root_logger = logging.getLogger("mappatrakka")
    root_logger.addHandler(handler)
    old_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    try:
        with tempfile.TemporaryDirectory(prefix="mappatrakka-web-") as tmp:
            cfg.output.path = str(Path(tmp) / "heatmap.png")
            result = run(find_export(upload_dir), cfg, show_progress=False, progress=progress)
            png = Path(result.image_path).read_bytes()
            with Image.open(result.image_path) as img:
                small = shrink(img, PREVIEW_SIZE)
            buf = io.BytesIO()
            small.save(buf, format="PNG")
            preview = buf.getvalue()
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)

    return {
        "png": png,
        "preview": preview,
        "preview_width": small.width,
        "preview_height": small.height,
        "width": cfg.render.width,
        "height": cfg.render.height,
        # (south, west, north, east) of the image edges, for laying it on a map.
        # The image is Web Mercator, same as web map tiles, so the corners are enough.
        "bounds": list(result.transform.bounds.to_latlon()),
        "stats": result.stats,
        "skipped": result.skipped,
        "log": handler.info,
        "warnings": handler.warnings,
        "warning_count": handler.warning_count,
    }


def generate_demo(options: dict[str, Any], progress: ProgressFn | None = None) -> dict[str, Any]:
    """Same as generate(), on a freshly made fake export."""
    from mappatrakka.demo import build_demo_export

    with tempfile.TemporaryDirectory(prefix="mappatrakka-demo-") as tmp:
        build_demo_export(tmp)
        return generate(tmp, options, progress)
