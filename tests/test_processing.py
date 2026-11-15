from datetime import timedelta

import numpy as np
import pytest

from conftest import as_arrays, straight_line
from mappatrakka.config import CleaningConfig
from mappatrakka.parsers.model import Segment, Track
from mappatrakka.processing import PrivacyZone, clean_segment, clean_track, haversine, line_length_m


def seg(points):
    return Segment(*as_arrays(points))


def test_haversine_known_distances():
    assert haversine(0, 0, 1, 0) == pytest.approx(111_195, rel=1e-3)
    assert haversine(51.5, -0.1, 51.5, -0.1) == 0
    # London -> Paris is about 344 km.
    assert haversine(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(343_500, rel=0.01)


def test_clean_line_passes_through(cleaning):
    pts = straight_line(51.0, 0.0, 51.01, 0.0, n=50)
    lines = clean_segment(seg(pts), cleaning)
    assert len(lines) == 1
    assert lines[0].shape == (50, 2)


def test_invalid_coordinates_removed(cleaning):
    pts = straight_line(51.0, 0.0, 51.005, 0.0, n=20)
    bad = list(pts)
    bad.insert(5, (95.0, 0.0, pts[5][2]))
    bad.insert(9, (float("nan"), 0.0, pts[9][2]))
    bad.insert(12, (0.0, 0.0, pts[12][2]))
    bad.insert(14, (51.0, 200.0, pts[14][2]))
    lines = clean_segment(seg(bad), cleaning)
    assert sum(len(line) for line in lines) == 20
    for line in lines:
        assert np.all(np.abs(line[:, 0] - 51.0) < 0.01)


def test_beyond_mercator_latitude_removed(cleaning):
    pts = straight_line(85.2, 0.0, 85.3, 0.0, n=10)
    assert clean_segment(seg(pts), cleaning) == []


def test_big_gap_splits_line(cleaning):
    a = straight_line(51.0, 0.0, 51.005, 0.0, n=30)
    b = straight_line(51.02, 0.0, 51.025, 0.0, n=30, start=a[-1][2])  # ~1.6 km jump
    lines = clean_segment(seg(a + b), cleaning)
    assert len(lines) == 2
    assert len(lines[0]) == 30 and len(lines[1]) == 30


def test_teleport_within_gap_limit_split_by_speed(cleaning):
    # 200 m in one second is under max_gap_m but impossibly fast.
    a = straight_line(51.0, 0.0, 51.001, 0.0, n=20, step_s=1)
    b = straight_line(51.0028, 0.0, 51.004, 0.0, n=20, start=a[-1][2] + timedelta(seconds=1), step_s=1)
    lines = clean_segment(seg(a + b), cleaning)
    assert len(lines) == 2


def test_antimeridian_crossing_split(cleaning):
    pts = straight_line(-17.0, 179.9990, -17.0, 179.9999, n=5) + straight_line(
        -17.0, -179.9999, -17.0, -179.999, n=5
    )
    lines = clean_segment(seg(pts), cleaning)
    assert len(lines) == 2
    for line in lines:
        assert np.ptp(line[:, 1]) < 1


@pytest.mark.parametrize("timed", [True, False])
def test_single_point_spike_removed(cleaning, timed):
    pts = straight_line(51.0, 0.0, 51.002, 0.0, n=21, timed=timed)
    spiked = list(pts)
    lat, lon, t = spiked[10]
    spiked[10] = (lat + 0.01, lon, t)  # ~1.1 km off and straight back
    lines = clean_segment(seg(spiked), cleaning)
    assert len(lines) == 1
    assert len(lines[0]) == 20
    assert lines[0][:, 0].max() < 51.0021


def test_sharp_hairpin_is_not_a_spike(cleaning):
    # Out 300 m and back, sampled every 10 s at 10 m/s - looks spiky on
    # distance alone but the speed is fine.
    out = straight_line(51.0, 0.0, 51.0027, 0.0, n=4, step_s=10)
    back = straight_line(51.0027, 0.0, 51.0, 0.0, n=4, start=out[-1][2], step_s=10)[1:]
    lines = clean_segment(seg(out + back), cleaning)
    assert sum(len(line) for line in lines) == 7


def test_duplicates_dropped(cleaning):
    pts = straight_line(51.0, 0.0, 51.001, 0.0, n=5)
    dup = [pts[0], pts[0], pts[1], pts[1], pts[1], pts[2], pts[3], pts[4]]
    lines = clean_segment(seg(dup), cleaning)
    assert len(lines[0]) == 5


def test_short_fragments_dropped():
    cfg = CleaningConfig(min_segment_points=5)
    pts = straight_line(51.0, 0.0, 51.0003, 0.0, n=3)
    assert clean_segment(seg(pts), cfg) == []


def test_privacy_zone_hides_points_and_breaks_line(cleaning):
    pts = straight_line(51.0, 0.0, 51.02, 0.0, n=101)  # ~2.2 km north
    zone = PrivacyZone(51.01, 0.0, 300)
    lines = clean_segment(seg(pts), cleaning, [zone])
    assert len(lines) == 2
    for line in lines:
        assert np.all(haversine(line[:, 0], line[:, 1], 51.01, 0.0) > 300)


def test_smoothing_keeps_endpoints():
    cfg = CleaningConfig(smoothing_window=5)
    pts = straight_line(51.0, 0.0, 51.002, 0.0, n=30)
    noisy = [(la + (0.00002 if i % 2 else -0.00002), lo, t) for i, (la, lo, t) in enumerate(pts)]
    line = clean_segment(seg(noisy), cfg)[0]
    assert line[0, 0] == noisy[0][0]
    assert line[-1, 0] == noisy[-1][0]
    raw_wiggle = np.abs(np.diff(np.array([p[0] for p in noisy]), 2)).mean()
    assert np.abs(np.diff(line[:, 0], 2)).mean() < raw_wiggle / 3


def test_clean_track_keeps_segments_apart(cleaning):
    a = straight_line(51.0, 0.0, 51.001, 0.0, n=10)
    b = straight_line(51.0011, 0.0, 51.002, 0.0, n=10)
    lines = clean_track(Track([seg(a), seg(b)]), cleaning)
    assert len(lines) == 2


def test_line_length():
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert line_length_m(line) == pytest.approx(2 * 111_195, rel=1e-3)
