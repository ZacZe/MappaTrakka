import numpy as np
import pytest

from mappatrakka.config import RegionConfig
from mappatrakka.projection import (
    EARTH_RADIUS_WEB,
    Bounds,
    PixelTransform,
    from_mercator,
    to_mercator,
)
from mappatrakka.region import ActivitySummary, choose_region, cluster_points, find_clusters

# --- projection ---------------------------------------------------------------


def test_mercator_known_points():
    x, y = to_mercator(0.0, 0.0)
    assert x == pytest.approx(0, abs=1e-6) and y == pytest.approx(0, abs=1e-6)
    x, _ = to_mercator(0.0, 180.0)
    assert x == pytest.approx(np.pi * EARTH_RADIUS_WEB)
    _, y = to_mercator(85.05112878, 0.0)
    assert y == pytest.approx(np.pi * EARTH_RADIUS_WEB, rel=1e-6)


def test_mercator_round_trip():
    lat = np.array([-60.0, -12.3, 0.0, 51.5, 70.1])
    lon = np.array([-179.0, -45.5, 0.0, -0.12, 25.0])
    back_lat, back_lon = from_mercator(*to_mercator(lat, lon))
    np.testing.assert_allclose(back_lat, lat, atol=1e-9)
    np.testing.assert_allclose(back_lon, lon, atol=1e-9)


@pytest.mark.parametrize("lat", [0.0, 51.5, 69.6])
def test_projection_preserves_local_aspect(lat):
    """1 km east and 1 km north must come out the same number of pixels."""
    d = 1000.0
    dlat = d / 111_195
    dlon = d / (111_195 * np.cos(np.radians(lat)))
    x0, y0 = to_mercator(lat, 10.0)
    xe, _ = to_mercator(lat, 10.0 + dlon)
    _, yn = to_mercator(lat + dlat, 10.0)
    assert (xe - x0) == pytest.approx(yn - y0, rel=2e-3)


def test_with_aspect_never_crops():
    b = Bounds(0, 0, 100, 50)
    sq = b.with_aspect(1.0)
    assert sq.width == pytest.approx(sq.height)
    assert sq.xmin <= b.xmin and sq.xmax >= b.xmax and sq.ymin <= b.ymin and sq.ymax >= b.ymax
    wide = b.with_aspect(3.0)
    assert wide.width / wide.height == pytest.approx(3.0)
    assert wide.height >= b.height


def test_pixel_transform_corners_and_flip():
    b = Bounds(1000, 2000, 3000, 3000)
    t = PixelTransform(b, 200, 100)
    px, py = t(np.array([1000, 3000]), np.array([3000, 2000]))
    np.testing.assert_allclose(px, [0, 200])
    np.testing.assert_allclose(py, [0, 100])  # north is up


# --- clustering -----------------------------------------------------------------


def test_cluster_points_basic():
    lat = np.array([51.50, 51.52, 51.55, 39.6, 39.62])
    lon = np.array([-0.10, -0.12, -0.05, 2.9, 2.91])
    labels = cluster_points(lat, lon, radius_km=30)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_cluster_points_chains():
    # Points 20 km apart in a row: all linked at 30 km, none at 10 km.
    lat = np.full(5, 0.0)
    lon = np.arange(5) * (20 / 111.195)
    assert len(set(cluster_points(lat, lon, 30))) == 1
    assert len(set(cluster_points(lat, lon, 10))) == 5


def test_cluster_points_across_antimeridian():
    labels = cluster_points(np.array([-17.0, -17.0]), np.array([179.99, -179.99]), 5)
    assert labels[0] == labels[1]


def test_cluster_points_many_identical_points_is_fast():
    lat = np.full(5000, 51.5) + np.random.default_rng(0).normal(0, 0.01, 5000)
    lon = np.full(5000, -0.1)
    labels = cluster_points(lat, lon, 30)
    assert len(set(labels)) == 1


def _summary(index, lat, lon, half_km=3.0):
    """Fake activity summary: a square route of +-half_km around a point."""
    d_lat = half_km / 111.195
    d_lon = half_km / (111.195 * np.cos(np.radians(lat)))
    x0, y0 = to_mercator(lat - d_lat, lon - d_lon)
    x1, y1 = to_mercator(lat + d_lat, lon + d_lon)
    return ActivitySummary(index, lat, lon, 4 * 2 * half_km * 1000, Bounds(float(x0), float(y0), float(x1), float(y1)))


@pytest.fixture
def home_and_holiday():
    rng = np.random.default_rng(1)
    acts = [_summary(i, 52.37 + rng.normal(0, 0.02), 4.90 + rng.normal(0, 0.03)) for i in range(40)]
    acts += [_summary(40 + i, 39.57, 2.65) for i in range(3)]
    return acts


def test_clusters_sorted_by_activity_count(home_and_holiday):
    clusters = find_clusters(home_and_holiday, 30)
    assert [c.count for c in clusters] == [40, 3]


def test_auto_region_ignores_holiday(home_and_holiday):
    choice = choose_region(home_and_holiday, RegionConfig(), aspect=1.0)
    s, w, n, e = choice.bounds.to_latlon()
    assert s < 52.37 < n and w < 4.90 < e
    assert s > 51.5  # nowhere near Mallorca
    assert choice.bounds.width == pytest.approx(choice.bounds.height)
    km_w, km_h = choice.bounds.ground_size_km()
    assert km_w < 30


def test_region_rank_two_picks_holiday(home_and_holiday):
    choice = choose_region(home_and_holiday, RegionConfig(cluster_rank=2), aspect=1.0)
    lat, lon = choice.bounds.center_latlon()
    assert lat == pytest.approx(39.57, abs=0.1) and lon == pytest.approx(2.65, abs=0.1)


def test_region_all_includes_everything(home_and_holiday):
    choice = choose_region(home_and_holiday, RegionConfig(mode="all"), aspect=1.0)
    s, w, n, e = choice.bounds.to_latlon()
    assert s < 39.57 and n > 52.37


def test_region_bbox_and_center_modes():
    choice = choose_region([], RegionConfig(mode="bbox", bbox=[4.8, 52.3, 5.0, 52.4]), aspect=2.0)
    assert choice.bounds.width / choice.bounds.height == pytest.approx(2.0)
    choice = choose_region([], RegionConfig(mode="center", center=[52.37, 4.9], radius_km=5), aspect=1.0)
    w, h = choice.bounds.ground_size_km()
    assert w == pytest.approx(10, rel=0.01) and h == pytest.approx(10, rel=0.01)


def test_single_long_ride_does_not_zoom_out_map():
    acts = [_summary(i, 52.37, 4.90, half_km=4) for i in range(60)]
    acts.append(_summary(60, 52.37, 4.90, half_km=60))  # one huge ride from home
    choice = choose_region(acts, RegionConfig(), aspect=1.0)
    km_w, _ = choice.bounds.ground_size_km()
    assert km_w < 12


def test_popular_long_route_is_kept():
    acts = [_summary(i, 52.37, 4.90, half_km=4) for i in range(60)]
    acts += [_summary(60 + i, 52.37, 4.90, half_km=10) for i in range(10)]
    choice = choose_region(acts, RegionConfig(), aspect=1.0)
    km_w, _ = choice.bounds.ground_size_km()
    assert km_w > 20


def test_small_export_not_trimmed():
    acts = [_summary(0, 52.37, 4.90, half_km=2), _summary(1, 52.37, 4.90, half_km=8)]
    choice = choose_region(acts, RegionConfig(), aspect=1.0)
    km_w, _ = choice.bounds.ground_size_km()
    assert km_w > 16
