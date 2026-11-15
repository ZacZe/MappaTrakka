import gzip

import numpy as np
import pytest

from conftest import T0, straight_line
from mappatrakka.demo import FitWriter, degrees_to_semicircles, write_fit, write_gpx, write_tcx
from mappatrakka.parsers import ParseError, detect_format, parse_bytes, read_track
from mappatrakka.parsers.fit import SEMICIRCLE_TO_DEG, semicircles_to_degrees

LONDON = (51.5074, -0.1278)


def _writers():
    return {
        "fit": lambda segs: write_fit(segs),
        "gpx": lambda segs: write_gpx(segs),
        "tcx": lambda segs: write_tcx(segs),
    }


# --- FIT specifics ----------------------------------------------------------


def test_semicircle_constants():
    assert semicircles_to_degrees(2**30) == pytest.approx(90.0)
    assert semicircles_to_degrees(-(2**30)) == pytest.approx(-90.0)
    assert semicircles_to_degrees(0) == 0.0
    assert SEMICIRCLE_TO_DEG == pytest.approx(180 / 2147483648)


def test_semicircle_invalid_values():
    assert semicircles_to_degrees(None) is None
    assert semicircles_to_degrees(0x7FFFFFFF) is None


def test_fit_coordinates_round_trip():
    pts = [(LONDON[0], LONDON[1], T0), (-33.8688, 151.2093, T0), (64.1466, -21.9426, T0)]
    track = parse_bytes(write_fit([pts]), "a.fit")
    seg = track.segments[0]
    # Semicircles resolve to ~8e-8 degrees (under a centimetre).
    np.testing.assert_allclose(seg.lat, [p[0] for p in pts], atol=1e-7)
    np.testing.assert_allclose(seg.lon, [p[1] for p in pts], atol=1e-7)
    assert seg.time[0] == pytest.approx(T0.timestamp())


def test_fit_semicircles_written_as_expected():
    # Guard the test writer itself, so the round trip above means something.
    assert degrees_to_semicircles(180.0) == 2**31
    assert degrees_to_semicircles(LONDON[0]) == round(LONDON[0] * 2**31 / 180)


def test_fit_skips_records_without_position():
    w = FitWriter()
    w.file_id(T0)
    w.timer(T0, start=True)
    w.record(T0, 51.0, 0.0)
    w.record(T0, None, None)  # no fix yet
    w.record(T0, 51.001, 0.0)
    w.timer(T0, start=False)
    track = parse_bytes(w.to_bytes(), "a.fit")
    assert track.point_count == 2


def test_fit_timer_stop_splits_segments():
    a = straight_line(51.0, 0.0, 51.01, 0.0, n=20)
    b = straight_line(51.05, 0.05, 51.06, 0.05, n=20, start=T0.replace(hour=10))
    track = parse_bytes(write_fit([a, b]), "a.fit")
    assert len(track.segments) == 2
    assert track.segments[0].lat[-1] == pytest.approx(51.01, abs=1e-6)
    assert track.segments[1].lat[0] == pytest.approx(51.05, abs=1e-6)


def test_fit_sport_metadata():
    pts = straight_line(51.0, 0.0, 51.01, 0.0, n=5)
    track = parse_bytes(write_fit([pts], "cycling", "virtual_activity"), "a.fit")
    assert track.sport == "cycling"
    assert track.sub_sport == "virtual_activity"


def test_fit_truncated_keeps_earlier_records():
    pts = straight_line(51.0, 0.0, 51.01, 0.0, n=200)
    data = write_fit([pts])
    track = parse_bytes(data[: len(data) // 2], "a.fit")
    assert 50 < track.point_count < 200
    assert track.warnings


def test_fit_bad_crc_is_a_warning_not_a_failure():
    data = bytearray(write_fit([straight_line(51.0, 0.0, 51.01, 0.0, n=20)]))
    data[-1] ^= 0xFF
    track = parse_bytes(bytes(data), "a.fit")
    assert track.point_count == 20


# --- all formats ------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["fit", "gpx", "tcx"])
@pytest.mark.parametrize("gz", [False, True])
def test_every_format_plain_and_gzipped(tmp_path, fmt, gz):
    pts = straight_line(51.5, -0.12, 51.52, -0.10, n=30)
    data = _writers()[fmt]([pts])
    name = f"act.{fmt}" + (".gz" if gz else "")
    path = tmp_path / name
    path.write_bytes(gzip.compress(data) if gz else data)

    track = read_track(path)
    assert track.point_count == 30
    seg = track.segments[0]
    np.testing.assert_allclose(seg.lat, [p[0] for p in pts], atol=1e-6)
    np.testing.assert_allclose(seg.lon, [p[1] for p in pts], atol=1e-6)


@pytest.mark.parametrize("fmt", ["fit", "gpx", "tcx"])
def test_point_order_is_preserved(fmt):
    # A wiggly path where any reordering would show up.
    lat = 51.5 + 0.001 * np.arange(40)
    lon = -0.1 + 0.002 * np.sin(np.arange(40))
    pts = [(a, b, T0) for a, b in zip(lat, lon)]
    track = parse_bytes(_writers()[fmt]([pts]), f"x.{fmt}")
    np.testing.assert_allclose(track.segments[0].lat, lat, atol=1e-6)
    np.testing.assert_allclose(track.segments[0].lon, lon, atol=1e-6)


def test_gpx_segments_stay_separate():
    a = straight_line(51.0, 0.0, 51.01, 0.0, n=10)
    b = straight_line(51.2, 0.2, 51.21, 0.2, n=10)
    track = parse_bytes(write_gpx([a, b]), "x.gpx")
    assert [len(s) for s in track.segments] == [10, 10]


def test_gpx_multiple_tracks():
    data = b"""<?xml version="1.0"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
<trk><trkseg><trkpt lat="1" lon="1"/><trkpt lat="1.001" lon="1"/></trkseg></trk>
<trk><trkseg><trkpt lat="2" lon="2"/><trkpt lat="2.001" lon="2"/></trkseg></trk>
</gpx>"""
    track = parse_bytes(data, "x.gpx")
    assert len(track.segments) == 2


def test_gpx_without_times():
    pts = straight_line(51.0, 0.0, 51.01, 0.0, n=5, timed=False)
    track = parse_bytes(write_gpx([pts]), "x.gpx")
    assert np.isnan(track.segments[0].time).all()


def test_gpx_gpx10_namespace_and_route_fallback():
    data = b"""<?xml version="1.0"?>
<gpx version="1.0" xmlns="http://www.topografix.com/GPX/1/0">
<wpt lat="9" lon="9"><name>ignored</name></wpt>
<rte><rtept lat="1" lon="1"/><rtept lat="1.001" lon="1.001"/></rte>
</gpx>"""
    track = parse_bytes(data, "x.gpx")
    assert track.point_count == 2


def test_gpx_bad_numbers_skipped():
    data = b"""<gpx><trk><trkseg>
<trkpt lat="abc" lon="1"/><trkpt lat="1" lon="1"/><trkpt lat="nan" lon="1"/><trkpt lat="1.001" lon="1"/>
</trkseg></trk></gpx>"""
    track = parse_bytes(data, "x.gpx")
    assert track.point_count == 2


def test_gpx_truncated_keeps_earlier_points():
    data = write_gpx([straight_line(51.0, 0.0, 51.01, 0.0, n=100)])
    track = parse_bytes(data[: len(data) // 2], "x.gpx")
    assert 10 < track.point_count < 100
    assert track.warnings


def test_tcx_leading_whitespace_and_laps_joined():
    lap1 = straight_line(51.0, 0.0, 51.01, 0.0, n=10)
    lap2 = straight_line(51.0101, 0.0, 51.02, 0.0, n=10)
    data = write_tcx([lap1, lap2])
    assert data.startswith(b"   ")
    track = parse_bytes(data, "x.tcx")
    assert len(track.segments) == 1
    assert track.point_count == 20
    assert track.sport == "biking"


def test_tcx_trackpoints_without_position_skipped():
    data = b"""<?xml version="1.0"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">
<Activities><Activity Sport="Running"><Lap><Track>
<Trackpoint><Time>2024-01-01T00:00:00Z</Time><HeartRateBpm><Value>120</Value></HeartRateBpm></Trackpoint>
<Trackpoint><Time>2024-01-01T00:00:01Z</Time><Position><LatitudeDegrees>1</LatitudeDegrees><LongitudeDegrees>1</LongitudeDegrees></Position></Trackpoint>
<Trackpoint><Time>2024-01-01T00:00:02Z</Time><Position><LatitudeDegrees>1.0001</LatitudeDegrees><LongitudeDegrees>1</LongitudeDegrees></Position></Trackpoint>
</Track></Lap></Activity></Activities></TrainingCenterDatabase>"""
    track = parse_bytes(data, "x.tcx")
    assert track.point_count == 2


# --- detection and failure handling -------------------------------------------


def test_format_detected_from_content_not_extension():
    data = write_gpx([straight_line(51.0, 0.0, 51.01, 0.0, n=5)])
    assert detect_format(data, "mislabelled.fit") == "gpx"
    assert parse_bytes(data, "mislabelled.fit").point_count == 5


def test_detect_format_by_sniffing():
    assert detect_format(write_fit([straight_line(1, 1, 1.01, 1, n=3)])) == "fit"
    assert detect_format(write_tcx([straight_line(1, 1, 1.01, 1, n=3)])) == "tcx"
    assert detect_format(b"hello", "x.txt") is None


@pytest.mark.parametrize(
    "data",
    [b"", b"   \n", b"not a gps file at all", b"\x0e\x20garbage .FIT garbage", b"<html></html>"],
)
def test_garbage_raises_parse_error(data):
    with pytest.raises(ParseError):
        parse_bytes(data, "x.fit")


def test_gpx_named_file_with_other_xml_root():
    with pytest.raises(ParseError):
        parse_bytes(b"<kml><Placemark/></kml>", "x.gpx")


def test_truncated_gzip_still_gives_points():
    data = gzip.compress(write_gpx([straight_line(51.0, 0.0, 51.01, 0.0, n=500)]))
    track = parse_bytes(data[: int(len(data) * 0.7)], "x.gpx.gz")
    assert track.point_count > 0
    assert any("gzip" in w for w in track.warnings)


def test_missing_file(tmp_path):
    with pytest.raises(ParseError):
        read_track(tmp_path / "nope.fit")
