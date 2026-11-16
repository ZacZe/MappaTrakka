import io
import shutil
import zipfile

import pytest
from PIL import Image

from mappatrakka.archive import extract_export, open_export
from mappatrakka.config import ConfigError, Config
from mappatrakka.demo import build_demo_export
from mappatrakka.pipeline import run
from mappatrakka.web import build_config, find_export, generate, generate_demo


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    return build_demo_export(tmp_path_factory.mktemp("export"), activities=20, seed=5)


def zip_export(export, dest, prefix=""):
    path = dest / "export.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for f in export.rglob("*"):
            if f.is_file():
                zf.write(f, prefix + f.relative_to(export).as_posix())
        zf.writestr(prefix + "media/holiday.jpg", b"\xff\xd8" + b"x" * 1000)
    return path


def small_cfg(tmp_path):
    cfg = Config()
    cfg.render.width = cfg.render.height = 300
    cfg.output.path = str(tmp_path / "out.png")
    cfg.output.use_cache = False
    cfg.output.workers = 1
    return cfg


@pytest.mark.parametrize("prefix", ["", "export_12345/"])
def test_extract_only_csv_and_gps_files(export, tmp_path, prefix):
    z = zip_export(export, tmp_path, prefix)
    out = extract_export(z, tmp_path / "x")
    assert (out / "activities.csv").is_file()
    names = {p.name for p in (out / "activities").iterdir()}
    assert names == {p.name for p in (export / "activities").iterdir()}
    assert not (out / "media").exists()


def test_zip_slip_is_harmless(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../escaped.gpx", "<gpx/>")
        zf.writestr("/abs/also.gpx", "<gpx/>")
        zf.writestr("activities/..", "x")
    out = extract_export(z, tmp_path / "dest" / "x")
    assert not (tmp_path / "escaped.gpx").exists()
    assert not (tmp_path / "dest" / "escaped.gpx").exists()
    assert sorted(p.name for p in (out / "activities").iterdir()) == ["also.gpx", "escaped.gpx"]


def test_open_export_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        with open_export(tmp_path / "missing"):
            pass
    junk = tmp_path / "junk.txt"
    junk.write_text("hello")
    with pytest.raises(ValueError):
        with open_export(junk):
            pass


def test_run_straight_from_zip_matches_folder(export, tmp_path):
    z = zip_export(export, tmp_path)
    a = run(export, small_cfg(tmp_path / "a"), show_progress=False)
    b = run(z, small_cfg(tmp_path / "b"), show_progress=False)
    assert a.image_path.read_bytes() == b.image_path.read_bytes()
    assert b.stats["activities_drawn"] == 20


def test_progress_callback(export, tmp_path):
    calls = []
    run(export, small_cfg(tmp_path), show_progress=False, progress=lambda *a: calls.append(a))
    stages = [c[0] for c in calls]
    assert stages[0] == "Reading" and "Drawing" in stages and stages[-1] == "Colouring"
    reading = [c for c in calls if c[0] == "Reading"]
    assert reading[-1][1] == reading[-1][2]


def test_build_config_from_form():
    cfg = build_config({"size": 1200, "colormap": "ice", "include_types": "Ride, Run", "area": 2, "glow": 0})
    assert (cfg.render.width, cfg.render.height) == (1200, 1200)
    assert cfg.filters.include_types == ["Ride", "Run"]
    assert cfg.region.cluster_rank == 2
    assert cfg.render.glow_strength == 0
    assert build_config({"include_types": "", "unknown": 1}).filters.include_types == []
    with pytest.raises(ConfigError):
        build_config({"scale": "cubic"})


def test_find_export(export, tmp_path):
    assert find_export(export) == export
    single = tmp_path / "zip_only"
    single.mkdir()
    zip_export(export, single)
    assert find_export(single).suffix == ".zip"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        find_export(empty)


def test_generate_from_flat_upload(export, tmp_path):
    # What the browser mounts: the csv and GPS files side by side, no folders.
    flat = tmp_path / "uploads"
    flat.mkdir()
    shutil.copy(export / "activities.csv", flat)
    for f in (export / "activities").iterdir():
        shutil.copy(f, flat)
    out = generate(str(flat), {"size": 300})
    img = Image.open(io.BytesIO(out["png"]))
    assert img.size == (300, 300)
    assert out["stats"]["activities_drawn"] == 20
    assert out["skipped"]  # virtual rides, the gym session, the corrupt file...
    assert out["warning_count"] >= 1


def test_generate_loose_gps_files_without_csv(export, tmp_path):
    flat = tmp_path / "loose"
    flat.mkdir()
    for f in sorted((export / "activities").iterdir())[:6]:
        shutil.copy(f, flat)
    out = generate(str(flat), {"size": 200})
    assert out["stats"]["activities_drawn"] >= 4


def test_generate_demo():
    out = generate_demo({"size": 300})
    assert out["stats"]["activities_drawn"] == 120
