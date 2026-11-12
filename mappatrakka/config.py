"""Settings, with defaults that work for a typical export.

Everything can be overridden from a TOML file (see config.example.toml) or
from the command line. Unknown keys are rejected so typos don't silently do
nothing.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Strava activity types that never have a meaningful GPS track. Compared after
# normalising (lowercase, letters/digits only), so "Virtual Ride" and
# "VirtualRide" both match "virtualride".
DEFAULT_EXCLUDED_TYPES = [
    "virtualride",
    "virtualrun",
    "virtualrow",
    "workout",
    "weighttraining",
    "yoga",
    "pilates",
    "crossfit",
    "hiit",
    "elliptical",
    "stairstepper",
    "rowing",
    "indoorrowing",
    "rockclimbing",
    "highintensityintervaltraining",
]

# FIT session sub-sports that mean "this happened indoors".
DEFAULT_EXCLUDED_SUB_SPORTS = [
    "virtual_activity",
    "indoor_cycling",
    "indoor_running",
    "treadmill",
    "indoor_rowing",
    "indoor_walking",
    "spin",
]


class ConfigError(ValueError):
    pass


@dataclass
class FilterConfig:
    # If non-empty, only these types are used. Otherwise everything that isn't
    # in exclude_types.
    include_types: list[str] = field(default_factory=list)
    exclude_types: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDED_TYPES))
    # Case-insensitive substrings of the activity name.
    exclude_name_patterns: list[str] = field(
        default_factory=lambda: ["zwift", "rouvy", "trainerroad", "indoor"]
    )
    exclude_sub_sports: list[str] = field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_SUB_SPORTS)
    )
    exclude_commutes: bool = False


@dataclass
class CleaningConfig:
    # Consecutive points further apart than this start a new line, so GPS
    # dropouts and pauses-while-moving don't draw straight lines across the map.
    max_gap_m: float = 250.0
    # Implied speeds above this are treated as GPS glitches (needs timestamps).
    max_speed_mps: float = 60.0
    # Fallback spike detection for files without timestamps.
    spike_distance_m: float = 400.0
    # Moving-average window in points. 1 disables smoothing.
    smoothing_window: int = 1
    min_segment_points: int = 3


@dataclass
class RegionConfig:
    # auto   - find the area with the most activities
    # all    - fit everything
    # bbox   - use `bbox` = [min_lon, min_lat, max_lon, max_lat]
    # center - use `center` = [lat, lon] and `radius_km`
    mode: str = "auto"
    bbox: list[float] | None = None
    center: list[float] | None = None
    radius_km: float = 15.0
    # Activities whose median points are within this distance of each other
    # (transitively) belong to the same area.
    cluster_radius_km: float = 30.0
    # 1 = busiest area, 2 = second busiest, ...
    cluster_rank: int = 1
    # Percent of route distance trimmed from each edge when sizing the map.
    # Stops a single 200 km ride from zooming the map right out.
    trim_percent: float = 0.5
    padding: float = 0.05


@dataclass
class RenderConfig:
    width: int = 4000
    height: int = 4000
    line_width: float = 2.0
    supersample: int = 3
    colormap: str = "fire"
    # Optional custom gradient, e.g. [[0.0, "#000000"], [1.0, "#ffffff"]].
    colormap_stops: list[list[Any]] | None = None
    # log | sqrt | linear
    scale: str = "log"
    # Activity count at this percentile of lit pixels maps to full brightness.
    reference_percentile: float = 99.5
    # Never let fewer than this many overlapping activities hit full brightness,
    # otherwise a small export renders entirely white.
    min_reference: float = 3.0
    gamma: float = 1.0
    glow_strength: float = 0.35
    glow_radius: float = 4.0
    background: str = "#000000"


@dataclass
class PrivacyConfig:
    # Each zone is {lat = .., lon = .., radius_m = ..}. GPS points inside a
    # zone are dropped before anything is drawn.
    zones: list[dict[str, float]] = field(default_factory=list)


@dataclass
class OutputConfig:
    path: str = "output/heatmap.png"
    # Write the showcase website here as well (None = don't).
    site_dir: str | None = None
    cache_dir: str = ".mappatrakka-cache"
    use_cache: bool = True
    # 0 = pick automatically.
    workers: int = 0


@dataclass
class SiteConfig:
    title: str = "Where I Go"
    subtitle: str = "Every ride, run and walk I've recorded, stacked on top of each other."
    author: str = ""


@dataclass
class Config:
    filters: FilterConfig = field(default_factory=FilterConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    site: SiteConfig = field(default_factory=SiteConfig)

    def validate(self) -> None:
        r = self.render
        if r.width < 16 or r.height < 16:
            raise ConfigError("render.width/height must be at least 16 pixels")
        if r.width * r.height > 400_000_000:
            raise ConfigError("render size is unreasonably large")
        if r.line_width <= 0:
            raise ConfigError("render.line_width must be positive")
        if not 1 <= r.supersample <= 8:
            raise ConfigError("render.supersample must be between 1 and 8")
        if r.scale not in ("log", "sqrt", "linear"):
            raise ConfigError(f"render.scale must be log, sqrt or linear (got {r.scale!r})")
        if not 0 < r.reference_percentile <= 100:
            raise ConfigError("render.reference_percentile must be in (0, 100]")
        if r.gamma <= 0:
            raise ConfigError("render.gamma must be positive")

        g = self.region
        if g.mode not in ("auto", "all", "bbox", "center"):
            raise ConfigError(f"region.mode must be auto, all, bbox or center (got {g.mode!r})")
        if g.mode == "bbox":
            if not g.bbox or len(g.bbox) != 4:
                raise ConfigError("region.bbox needs [min_lon, min_lat, max_lon, max_lat]")
            min_lon, min_lat, max_lon, max_lat = g.bbox
            if not (min_lon < max_lon and min_lat < max_lat):
                raise ConfigError("region.bbox min values must be below max values")
        if g.mode == "center":
            if not g.center or len(g.center) != 2:
                raise ConfigError("region.center needs [lat, lon]")
            if g.radius_km <= 0:
                raise ConfigError("region.radius_km must be positive")
        if g.cluster_rank < 1:
            raise ConfigError("region.cluster_rank starts at 1")
        if not 0 <= g.trim_percent < 50:
            raise ConfigError("region.trim_percent must be between 0 and 50")
        if g.cluster_radius_km <= 0:
            raise ConfigError("region.cluster_radius_km must be positive")

        c = self.cleaning
        if c.max_gap_m <= 0 or c.max_speed_mps <= 0 or c.spike_distance_m <= 0:
            raise ConfigError("cleaning distances and speeds must be positive")
        if c.smoothing_window < 1:
            raise ConfigError("cleaning.smoothing_window must be >= 1")
        if c.min_segment_points < 2:
            raise ConfigError("cleaning.min_segment_points must be >= 2")

        for zone in self.privacy.zones:
            missing = {"lat", "lon", "radius_m"} - set(zone)
            if missing:
                raise ConfigError(f"privacy zone {zone} is missing {sorted(missing)}")

        if self.output.workers < 0:
            raise ConfigError("output.workers can't be negative")


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown setting(s) in [{where}]: {', '.join(sorted(unknown))}")
    return cls(**data)


def config_from_dict(data: dict[str, Any]) -> Config:
    sections = {f.name: f.type for f in dataclasses.fields(Config)}
    unknown = set(data) - set(sections)
    if unknown:
        raise ConfigError(f"unknown config section(s): {', '.join(sorted(unknown))}")

    kwargs = {}
    for f in dataclasses.fields(Config):
        if f.name in data:
            section = data[f.name]
            if not isinstance(section, dict):
                raise ConfigError(f"[{f.name}] must be a table")
            section_cls = type(f.default_factory())  # type: ignore[misc]
            kwargs[f.name] = _build(section_cls, section, f.name)
    cfg = Config(**kwargs)
    cfg.validate()
    return cfg


def load_config(path: str | Path | None) -> Config:
    if path is None:
        cfg = Config()
        cfg.validate()
        return cfg
    path = Path(path)
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"couldn't parse {path}: {exc}") from None
    return config_from_dict(data)
