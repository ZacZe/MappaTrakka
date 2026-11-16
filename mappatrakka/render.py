"""Drawing tracks and turning the counts into an image.

Every activity is drawn on its own (anti-aliased by supersampling), so a pixel
holds at most 1.0 from any single activity. Summing those gives "how many
activities went through here", which is what the brightness should mean. It
doesn't depend on how often the device logged a point, and riding out and
back along a road doesn't count it twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from mappatrakka.colormaps import Colormap, parse_colour
from mappatrakka.config import RenderConfig
from mappatrakka.projection import PixelTransform

# Activities are drawn in square chunks of this many output pixels so a route
# spanning the whole map doesn't need one huge supersampled image.
CHUNK = 1024
# Long lines are cut into pieces of this many points, each with its own
# bounding box, so a chunk only draws the bits that actually cross it.
PIECE_POINTS = 256


@dataclass
class _Piece:
    xy: np.ndarray  # (N, 2) pixel coordinates
    x0: float
    y0: float
    x1: float
    y1: float


def _pieces(lines: list[np.ndarray], margin: float, w: int, h: int) -> list[_Piece]:
    out = []
    for line in lines:
        if len(line) < 2:
            continue
        # Throw away stretches that are nowhere near the canvas, but keep one
        # point either side of every visible stretch so lines run off the edge
        # instead of stopping short.
        inside = (
            (line[:, 0] >= -margin)
            & (line[:, 0] <= w + margin)
            & (line[:, 1] >= -margin)
            & (line[:, 1] <= h + margin)
        )
        if not inside.any():
            continue
        near = inside.copy()
        near[1:] |= inside[:-1]
        near[:-1] |= inside[1:]
        padded = np.concatenate(([False], near, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for a, b in zip(edges[::2], edges[1::2]):
            run = line[a:b]
            for i in range(0, len(run) - 1, PIECE_POINTS - 1):
                part = run[i : i + PIECE_POINTS]
                if len(part) < 2:
                    continue
                mn, mx = part.min(axis=0), part.max(axis=0)
                out.append(_Piece(part, mn[0], mn[1], mx[0], mx[1]))
    return out


class HeatmapCanvas:
    """Accumulates activities. `counts[y, x]` ~= activities through that pixel."""

    def __init__(self, transform: PixelTransform, line_width: float = 2.0, supersample: int = 3):
        self.transform = transform
        self.width = transform.width
        self.height = transform.height
        self.line_width = line_width
        self.ss = supersample
        self.counts = np.zeros((self.height, self.width), dtype=np.float32)
        self.activities_drawn = 0

    def add_activity(self, lines_xy: list[tuple[np.ndarray, np.ndarray]]) -> bool:
        """Add one activity, given as Mercator (x, y) arrays per line.

        Returns False if none of it falls on the canvas.
        """
        px_lines = []
        for x, y in lines_xy:
            px, py = self.transform(x, y)
            px_lines.append(np.column_stack((px, py)))
        return self.add_pixel_lines(px_lines)

    def add_pixel_lines(self, lines: list[np.ndarray]) -> bool:
        pad = self.line_width / 2 + 2
        pieces = _pieces(lines, max(pad, 8.0), self.width, self.height)
        if not pieces:
            return False

        x0 = max(0, int(np.floor(min(p.x0 for p in pieces) - pad)))
        y0 = max(0, int(np.floor(min(p.y0 for p in pieces) - pad)))
        x1 = min(self.width, int(np.ceil(max(p.x1 for p in pieces) + pad)))
        y1 = min(self.height, int(np.ceil(max(p.y1 for p in pieces) + pad)))
        if x0 >= x1 or y0 >= y1:
            return False

        drew = False
        for cy in range(y0, y1, CHUNK):
            for cx in range(x0, x1, CHUNK):
                cw, ch = min(CHUNK, x1 - cx), min(CHUNK, y1 - cy)
                hits = [
                    p
                    for p in pieces
                    if p.x1 >= cx - pad and p.x0 <= cx + cw + pad
                    and p.y1 >= cy - pad and p.y0 <= cy + ch + pad
                ]
                if hits:
                    tile = self._draw_chunk(hits, cx, cy, cw, ch)
                    if tile is not None:
                        self.counts[cy : cy + ch, cx : cx + cw] += tile
                        drew = True
        if drew:
            self.activities_drawn += 1
        return drew

    def _draw_chunk(self, pieces: list[_Piece], cx: int, cy: int, cw: int, ch: int) -> np.ndarray | None:
        s = self.ss
        img = Image.new("L", (cw * s, ch * s), 0)
        draw = ImageDraw.Draw(img)
        width = max(1, int(round(self.line_width * s)))
        # Pixel centre (i + 0.5) in output space -> sub-pixel index in the tile.
        offset = np.array([cx, cy], dtype=np.float64)
        for p in pieces:
            pts = (p.xy - offset) * s - 0.5
            flat = pts.ravel().tolist()
            draw.line(flat, fill=255, width=width, joint="curve" if width > 2 else None)
            if width > 2:
                # Round caps, otherwise segment ends look chopped off.
                r = width / 2
                for ex, ey in (pts[0], pts[-1]):
                    draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=255)
        if s > 1:
            img = img.reduce(s)
        tile = np.asarray(img, dtype=np.float32)
        if not tile.any():
            return None
        return tile / 255.0


def normalise(counts: np.ndarray, cfg: RenderConfig) -> tuple[np.ndarray, float]:
    """Map activity counts to 0..1 brightness. Returns (values, reference)."""
    lit = counts[counts > 0.5]
    if lit.size == 0:
        return np.zeros_like(counts), 0.0
    ref = float(np.percentile(lit, cfg.reference_percentile))
    ref = max(ref, cfg.min_reference)
    c = np.maximum(counts, 0)
    if cfg.scale == "log":
        v = np.log1p(c) / np.log1p(ref)
    elif cfg.scale == "sqrt":
        v = np.sqrt(c / ref)
    else:
        v = c / ref
    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    if cfg.gamma != 1.0:
        v **= cfg.gamma
    return v, ref


def _blur(values: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray(np.round(values * 255).astype(np.uint8))
    small = max(1, int(radius // 4))
    if small > 1:
        # Glow is soft anyway; blurring at lower resolution is much faster.
        w, h = img.size
        reduced = img.resize((max(1, w // small), max(1, h // small)), Image.Resampling.BOX)
        reduced = reduced.filter(ImageFilter.GaussianBlur(radius / small))
        img = reduced.resize((w, h), Image.Resampling.BILINEAR)
    else:
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def colourise(values: np.ndarray, cmap: Colormap, cfg: RenderConfig) -> Image.Image:
    """Opaque RGB poster image."""
    glow = cfg.glow_strength > 0 and cfg.glow_radius > 0
    halo_values = _blur(values, cfg.glow_radius) if glow else None
    bg = np.array(parse_colour(cfg.background), dtype=np.float32) / 255.0

    h, w = values.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    # Work in bands of rows: full-size float RGB arrays for a 4000px image
    # are ~200 MB each, which hurts in a browser tab.
    band = 256
    for y in range(0, h, band):
        v = values[y : y + band]
        rgb = cmap(v).astype(np.float32) / 255.0
        if halo_values is not None:
            halo = cmap(halo_values[y : y + band]).astype(np.float32) / 255.0
            halo *= cfg.glow_strength
            # Screen blend: brightens without ever blowing past white, and
            # leaves the sharp lines on top untouched where the halo is dark.
            rgb = 1.0 - (1.0 - rgb) * (1.0 - halo)
        if bg.any():
            # Colormaps start at black; put a custom background under unlit pixels.
            unlit = (v <= 0)[..., None] & (rgb.sum(axis=2, keepdims=True) == 0)
            rgb = np.where(unlit, bg, rgb)
        out[y : y + band] = np.round(rgb * 255).astype(np.uint8)
    return Image.fromarray(out)


def shrink(img: Image.Image, longest: int) -> Image.Image:
    """Smaller copy that keeps thin lines visible.

    Plain downscaling averages a 2px line with the black around it until it
    all but disappears. Spreading each bright pixel over its neighbours first
    (a max filter about as wide as the shrink factor) keeps the routes about
    as bright as they look at full size.
    """
    factor = max(img.size) / longest
    if factor <= 1:
        return img.copy()
    k = int(factor)
    if k % 2 == 0:
        k += 1
    if k > 1:
        img = img.filter(ImageFilter.MaxFilter(min(k, 9)))
    size = (max(1, round(img.width / factor)), max(1, round(img.height / factor)))
    return img.resize(size, Image.Resampling.LANCZOS)
