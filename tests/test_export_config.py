import csv

import pytest

from mappatrakka.config import ConfigError, FilterConfig, config_from_dict, load_config
from mappatrakka.export import normalise_type, read_activities, select_activities

HEADER = ["Activity ID", "Activity Date", "Activity Name", "Activity Type", "Commute", "Filename", "Commute"]


def write_export(tmp_path, rows, files=()):
    (tmp_path / "activities").mkdir(exist_ok=True)
    for f in files:
        (tmp_path / f).write_bytes(b"<gpx></gpx>")
    with (tmp_path / "activities.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    return tmp_path


def test_normalise_type():
    assert normalise_type("Virtual Ride") == "virtualride"
    assert normalise_type("VirtualRide") == "virtualride"
    assert normalise_type("E-Bike Ride") == "ebikeride"


def test_reads_rows_and_resolves_files(tmp_path):
    write_export(
        tmp_path,
        [
            ["1", "d", "Morning Ride", "Ride", "true", "activities/1.gpx", "false"],
            ["2", "d", "Gym", "Weight Training", "false", "", "false"],
            ["3", "d", "Lost", "Ride", "false", "activities/3.fit.gz", "false"],
        ],
        files=["activities/1.gpx"],
    )
    acts = read_activities(tmp_path)
    assert [a.activity_id for a in acts] == ["1", "2", "3"]
    assert acts[0].path == tmp_path / "activities" / "1.gpx"
    assert acts[1].path is None and acts[2].path is None
    # First of the duplicated Commute columns wins.
    assert acts[0].commute is True


def test_duplicate_rows_and_files_ignored(tmp_path):
    write_export(
        tmp_path,
        [
            ["1", "d", "A", "Ride", "", "activities/1.gpx", ""],
            ["1", "d", "A again", "Ride", "", "activities/1.gpx", ""],
            ["2", "d", "Same file", "Ride", "", "activities/1.gpx", ""],
            [],
        ],
        files=["activities/1.gpx"],
    )
    assert [a.activity_id for a in read_activities(tmp_path)] == ["1"]


def test_short_rows_and_bom(tmp_path):
    (tmp_path / "activities").mkdir()
    (tmp_path / "activities" / "9.gpx").write_bytes(b"<gpx/>")
    text = "﻿Activity ID,Activity Type,Filename\n9,Ride,activities/9.gpx\n10\n"
    (tmp_path / "activities.csv").write_text(text, encoding="utf-8")
    acts = read_activities(tmp_path)
    assert acts[0].type == "Ride" and acts[0].path is not None
    assert acts[1].path is None


def test_folder_scan_without_csv(tmp_path):
    (tmp_path / "activities").mkdir()
    for name in ["b.fit.gz", "a.gpx", "notes.txt"]:
        (tmp_path / "activities" / name).write_bytes(b"x")
    acts = read_activities(tmp_path)
    assert [a.path.name for a in acts] == ["a.gpx", "b.fit.gz"]


def test_filters(tmp_path):
    files = [f"activities/{i}.gpx" for i in range(1, 8)]
    write_export(
        tmp_path,
        [
            ["1", "d", "Ride", "Ride", "false", files[0], ""],
            ["2", "d", "Zwift ride", "Virtual Ride", "false", files[1], ""],
            ["3", "d", "Yoga", "Yoga", "false", files[2], ""],
            ["4", "d", "Indoor spin", "Ride", "false", files[3], ""],
            ["5", "d", "Run", "Run", "false", files[4], ""],
            ["6", "d", "Commute", "Ride", "true", files[5], ""],
            ["7", "d", "Watopia", "VirtualRide", "false", files[6], ""],
        ],
        files=files,
    )
    acts = read_activities(tmp_path)

    kept, skipped = select_activities(acts, FilterConfig())
    assert [a.activity_id for a in kept] == ["1", "5", "6"]
    assert sum(skipped.values()) == 4

    kept, _ = select_activities(acts, FilterConfig(include_types=["Ride"], exclude_commutes=True))
    assert [a.activity_id for a in kept] == ["1"]

    kept, _ = select_activities(acts, FilterConfig(exclude_types=[], exclude_name_patterns=[]))
    assert len(kept) == 7


def test_default_config_is_valid():
    cfg = load_config(None)
    assert cfg.render.width == cfg.render.height == 4000
    assert cfg.render.colormap == "fire"
    assert cfg.region.mode == "auto"


def test_config_from_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        '[render]\nwidth = 800\nheight = 600\ncolormap = "ice"\n'
        "[region]\nmode = \"bbox\"\nbbox = [4.8, 52.3, 5.0, 52.4]\n"
        "[[privacy.zones]]\nlat = 52.37\nlon = 4.9\nradius_m = 250\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert (cfg.render.width, cfg.render.height) == (800, 600)
    assert cfg.region.bbox == [4.8, 52.3, 5.0, 52.4]
    assert cfg.privacy.zones[0]["radius_m"] == 250


@pytest.mark.parametrize(
    "data",
    [
        {"render": {"widht": 100}},
        {"rendering": {}},
        {"render": {"scale": "cubic"}},
        {"render": {"width": 0}},
        {"region": {"mode": "bbox"}},
        {"region": {"mode": "bbox", "bbox": [5, 52, 4, 53]}},
        {"region": {"mode": "center", "center": [1]}},
        {"privacy": {"zones": [{"lat": 1, "lon": 2}]}},
        {"cleaning": {"smoothing_window": 0}},
        {"render": "big"},
    ],
)
def test_bad_config_rejected(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.toml")


def test_example_config_matches_defaults():
    from pathlib import Path

    from mappatrakka.config import Config

    example = load_config(Path(__file__).parent.parent / "config.example.toml")
    defaults = Config()
    defaults.validate()
    # Same settings apart from type names being written in CamelCase.
    example.filters.exclude_types = [t.lower() for t in example.filters.exclude_types]
    defaults.filters.exclude_types = [t.lower() for t in defaults.filters.exclude_types]
    assert example == defaults


def test_include_types_keeps_untyped_loose_files(tmp_path):
    (tmp_path / "activities").mkdir()
    (tmp_path / "activities" / "a.gpx").write_bytes(b"<gpx/>")
    acts = read_activities(tmp_path)
    kept, _ = select_activities(acts, FilterConfig(include_types=["Ride"]))
    assert len(kept) == 1
