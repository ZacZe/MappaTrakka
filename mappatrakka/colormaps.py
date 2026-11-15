"""Colour gradients for the heatmap.

A gradient is a list of (position, colour) stops from 0 to 1. Position 0 is
used for pixels nobody has been through, so it should match the background.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

PRESETS: dict[str, list[tuple[float, str]]] = {
    "fire": [
        (0.00, "#000000"),
        (0.10, "#4a0303"),
        (0.25, "#9b0a00"),
        (0.45, "#e03800"),
        (0.62, "#ff8a00"),
        (0.78, "#ffd23a"),
        (0.90, "#fff4a8"),
        (1.00, "#ffffff"),
    ],
    "ice": [
        (0.00, "#000000"),
        (0.12, "#04163a"),
        (0.35, "#0b4fa8"),
        (0.60, "#1fa6e8"),
        (0.82, "#9ce8ff"),
        (1.00, "#ffffff"),
    ],
    "neon": [
        (0.00, "#000000"),
        (0.12, "#2b0438"),
        (0.35, "#7a0aa0"),
        (0.60, "#e3229a"),
        (0.82, "#ff9ad2"),
        (1.00, "#ffffff"),
    ],
    "mono": [
        (0.00, "#000000"),
        (0.15, "#2a2a2a"),
        (1.00, "#ffffff"),
    ],
}


def parse_colour(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        raise ValueError(f"not a hex colour: {value!r}")
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        raise ValueError(f"not a hex colour: {value!r}") from None


class Colormap:
    def __init__(self, stops: Sequence[Sequence[Any]], size: int = 1024):
        if len(stops) < 2:
            raise ValueError("a colormap needs at least two stops")
        parsed = sorted((float(p), parse_colour(str(c))) for p, c in stops)
        pos = np.array([p for p, _ in parsed])
        if pos[0] != 0.0 or pos[-1] != 1.0:
            raise ValueError("colormap stops must start at 0.0 and end at 1.0")
        rgb = np.array([c for _, c in parsed], dtype=np.float64)
        x = np.linspace(0.0, 1.0, size)
        self.lut = np.stack([np.interp(x, pos, rgb[:, i]) for i in range(3)], axis=1)
        self.lut = np.round(self.lut).astype(np.uint8)
        self.size = size

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Map an array of 0..1 floats to uint8 RGB."""
        idx = np.clip(values, 0.0, 1.0) * (self.size - 1)
        return self.lut[np.round(idx).astype(np.intp)]


def get_colormap(name: str, stops: Sequence[Sequence[Any]] | None = None) -> Colormap:
    if stops:
        return Colormap(stops)
    try:
        return Colormap(PRESETS[name])
    except KeyError:
        raise ValueError(
            f"unknown colormap {name!r}; pick one of {', '.join(sorted(PRESETS))} "
            "or set render.colormap_stops"
        ) from None
