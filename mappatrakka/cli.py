"""Command line entry point: `python -m mappatrakka EXPORT_DIR`."""

from __future__ import annotations

import argparse
import logging
import sys

from mappatrakka import __version__
from mappatrakka.colormaps import PRESETS
from mappatrakka.config import Config, ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mappatrakka",
        description="Make a heatmap of your most travelled routes from a Strava bulk export.",
    )
    p.add_argument("export", help="Strava export: the .zip Strava sent you, or the unzipped folder")
    p.add_argument("-c", "--config", help="TOML settings file (see config.example.toml)")
    p.add_argument("-o", "--output", help="PNG to write (default output/heatmap.png)")

    g = p.add_argument_group("image")
    g.add_argument("--size", type=int, metavar="PX", help="square image, PX by PX")
    g.add_argument("--width", type=int)
    g.add_argument("--height", type=int)
    g.add_argument("--colormap", choices=sorted(PRESETS))
    g.add_argument("--scale", choices=["log", "sqrt", "linear"], help="brightness curve")
    g.add_argument("--line-width", type=float, metavar="PX")
    g.add_argument("--glow", type=float, metavar="STRENGTH", help="0 turns the glow off")

    g = p.add_argument_group("which activities")
    g.add_argument("--include-types", nargs="+", metavar="TYPE", help="only these, e.g. Ride Run")
    g.add_argument("--exclude-types", nargs="+", metavar="TYPE", help="add to the excluded types")
    g.add_argument("--no-commutes", action="store_true", help="leave out activities marked as commutes")

    g = p.add_argument_group("map area")
    g.add_argument("--region", choices=["auto", "all", "bbox", "center"])
    g.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    g.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"))
    g.add_argument("--radius-km", type=float)
    g.add_argument("--area", type=int, metavar="N", help="use the Nth busiest area instead of the busiest")

    g = p.add_argument_group("running")
    g.add_argument("--workers", type=int, help="parser processes (default: automatic)")
    g.add_argument("--no-cache", action="store_true", help="don't keep parsed tracks between runs")
    g.add_argument("-q", "--quiet", action="store_true")
    g.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def apply_args(cfg: Config, args: argparse.Namespace) -> Config:
    """Command line flags win over the config file."""
    r, f, g, o = cfg.render, cfg.filters, cfg.region, cfg.output
    if args.output:
        o.path = args.output
    if args.size:
        r.width = r.height = args.size
    if args.width:
        r.width = args.width
    if args.height:
        r.height = args.height
    if args.colormap:
        r.colormap = args.colormap
        r.colormap_stops = None
    if args.scale:
        r.scale = args.scale
    if args.line_width:
        r.line_width = args.line_width
    if args.glow is not None:
        r.glow_strength = args.glow
    if args.include_types:
        f.include_types = args.include_types
    if args.exclude_types:
        f.exclude_types = f.exclude_types + args.exclude_types
    if args.no_commutes:
        f.exclude_commutes = True
    if args.bbox:
        g.bbox = list(args.bbox)
        g.mode = "bbox"
    if args.center:
        g.center = list(args.center)
        g.mode = "center"
    if args.radius_km:
        g.radius_km = args.radius_km
    if args.region:
        g.mode = args.region
    if args.area:
        g.cluster_rank = args.area
    if args.workers is not None:
        o.workers = args.workers
    if args.no_cache:
        o.use_cache = False
    cfg.validate()
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s" if args.verbose else "%(message)s")

    try:
        cfg = apply_args(load_config(args.config), args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    from mappatrakka.pipeline import run

    try:
        result = run(args.export, cfg, show_progress=not args.quiet)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    if not args.quiet:
        s = result.stats
        print(
            f"\nDone: {s['activities_drawn']} activities drawn, "
            f"{s['distance_km']:.0f} km, busiest pixel visited {s['max_overlap']} times."
        )
        print(f"Image: {result.image_path}")
    return 0
