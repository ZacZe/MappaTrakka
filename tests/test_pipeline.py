"""End-to-end runs over a generated export."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from mappatrakka.cli import main
from mappatrakka.config import Config
from mappatrakka.demo import build_demo_export
from mappatrakka.pipeline import run


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    return build_demo_export(tmp_path_factory.mktemp("export"), activities=30, seed=3)


def make_cfg(tmp_path, size=400, **render):
    cfg = Config()
    cfg.render.width = cfg.render.height = size
    for k, v in render.items():
        setattr(cfg.render, k, v)
    cfg.output.path = str(tmp_path / "out.png")
    cfg.output.cache_dir = str(tmp_path / "cache")
    cfg.output.workers = 1
    return cfg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_run(export, tmp_path):
    result = run(export, make_cfg(tmp_path), show_progress=False)
    img = Image.open(result.image_path)
    assert img.size == (400, 400)
    s = result.stats
    assert s["activities_drawn"] == 30  # holiday, virtual, corrupt etc. all left out
    assert s["areas_found"] == 2  # home + holiday (virtual rides were filtered by type)
    lat, lon = result.region.bounds.center_latlon()
    assert lat == pytest.approx(52.37, abs=0.1) and lon == pytest.approx(4.90, abs=0.1)
    px = np.asarray(img)
    assert px[0, 0].tolist() == [0, 0, 0]
    assert px.max() == 255  # the busiest roads reach white


def test_virtual_rides_dropped_by_fit_metadata_too(export, tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.filters.exclude_types = []
    cfg.filters.exclude_name_patterns = []
    result = run(export, cfg, show_progress=False)
    # Type filters off, but the FIT files say sub_sport=virtual_activity.
    assert result.stats["areas_found"] == 2


def test_runs_are_deterministic(export, tmp_path):
    a = run(export, make_cfg(tmp_path / "a"), show_progress=False)
    b = run(export, make_cfg(tmp_path / "a"), show_progress=False)  # cached
    cfg = make_cfg(tmp_path / "c")
    cfg.output.workers = 2
    c = run(export, cfg, show_progress=False)  # fresh, parallel
    assert digest(a.image_path) == digest(b.image_path) == digest(c.image_path)


def test_non_square_output_keeps_geographic_aspect(export, tmp_path):
    result = run(export, make_cfg(tmp_path, size=400, width=600, height=300), show_progress=False)
    assert Image.open(result.image_path).size == (600, 300)
    b = result.region.bounds
    assert b.width / b.height == pytest.approx(2.0)
    # Same metres per pixel horizontally and vertically.
    assert b.width / 600 == pytest.approx(b.height / 300)


def test_holiday_area_selectable(export, tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.region.cluster_rank = 2
    result = run(export, cfg, show_progress=False)
    lat, _ = result.region.bounds.center_latlon()
    assert lat == pytest.approx(39.57, abs=0.2)
    assert result.stats["activities_drawn"] == 3


def test_privacy_zone_blanks_home(export, tmp_path):
    cfg = make_cfg(tmp_path, glow_strength=0)
    cfg.privacy.zones = [{"lat": 52.3676, "lon": 4.9041, "radius_m": 400}]
    result = run(export, cfg, show_progress=False)
    t = result.transform
    from mappatrakka.projection import to_mercator

    hx, hy = t(*to_mercator(52.3676, 4.9041))
    r = int(300 / t.metres_per_pixel())  # well inside the 400 m zone
    px = np.asarray(Image.open(result.image_path))
    patch = px[int(hy) - r : int(hy) + r, int(hx) - r : int(hx) + r]
    assert patch.max() == 0


def test_cli(export, tmp_path, capsys):
    out = tmp_path / "cli.png"
    code = main([str(export), "-o", str(out), "--size", "300", "--workers", "1", "--no-cache", "-q"])
    assert code == 0
    assert Image.open(out).size == (300, 300)


def test_cli_errors(tmp_path, capsys):
    assert main([str(tmp_path / "missing"), "-q"]) == 1
    bad = tmp_path / "bad.toml"
    bad.write_text("[render]\nnope = 1\n", encoding="utf-8")
    assert main([str(tmp_path), "-c", str(bad), "-q"]) == 2


def test_empty_export(tmp_path):
    (tmp_path / "activities.csv").write_text("Activity ID,Activity Type,Filename\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run(tmp_path, make_cfg(tmp_path), show_progress=False)
