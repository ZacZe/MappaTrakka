import numpy as np
import pytest

from mappatrakka.colormaps import PRESETS, Colormap, get_colormap
from mappatrakka.config import RenderConfig
from mappatrakka.projection import Bounds, PixelTransform
from mappatrakka.render import CHUNK, HeatmapCanvas, colourise, normalise, overlay


def canvas(w=400, h=400, line_width=2.0):
    return HeatmapCanvas(PixelTransform(Bounds(0, 0, w, h), w, h), line_width, supersample=3)


def hline(y, x0=20, x1=380, n=50):
    return np.column_stack((np.linspace(x0, x1, n), np.full(n, float(y))))


def test_repeated_route_brighter_than_single_route():
    c = canvas()
    for _ in range(5):
        c.add_pixel_lines([hline(100)])
    c.add_pixel_lines([hline(300)])
    assert c.counts[100, 200] == pytest.approx(5.0, abs=0.3)
    assert c.counts[300, 200] == pytest.approx(1.0, abs=0.1)

    values, _ = normalise(c.counts, RenderConfig())
    assert values[100, 200] > values[300, 200] > 0
    img = np.asarray(colourise(values, get_colormap("fire"), RenderConfig(glow_strength=0)))
    assert img[100, 200].sum() > img[300, 200].sum() > 0


def test_one_activity_counts_once_even_out_and_back():
    c = canvas()
    out_and_back = np.concatenate([hline(200), hline(200)[::-1]])
    c.add_pixel_lines([out_and_back, hline(200)])
    assert c.counts.max() <= 1.0 + 1e-6


def test_intensity_does_not_depend_on_point_density():
    sparse, dense = canvas(), canvas()
    sparse.add_pixel_lines([hline(200, n=5)])
    dense.add_pixel_lines([hline(200, n=2000)])
    assert sparse.counts.sum() == pytest.approx(dense.counts.sum(), rel=0.02)


def test_separate_lines_not_joined():
    c = canvas()
    c.add_pixel_lines([hline(200, 20, 150), hline(200, 250, 380)])
    assert c.counts[200, 100] > 0.5
    assert c.counts[200, 300] > 0.5
    assert c.counts[195:206, 170:230].max() == 0


def test_no_seam_at_chunk_boundary():
    w = CHUNK * 2
    c = HeatmapCanvas(PixelTransform(Bounds(0, 0, w, 64), w, 64), 2.0, 3)
    c.add_pixel_lines([np.array([[10.0, 32.0], [w - 10.0, 32.0]])])
    row = c.counts[30:35].sum(axis=0)
    around = row[CHUNK - 5 : CHUNK + 5]
    assert around.min() == pytest.approx(row[CHUNK // 2], rel=0.01)


def test_line_off_canvas_is_skipped_and_partial_line_drawn():
    c = canvas()
    assert not c.add_pixel_lines([hline(-500), hline(10_000)])
    assert c.add_pixel_lines([np.array([[-1e6, 200.0], [200.0, 200.0]])])
    assert c.counts[200, 5] > 0.5


def test_line_width_scales():
    thin, thick = canvas(line_width=1.0), canvas(line_width=4.0)
    thin.add_pixel_lines([hline(200)])
    thick.add_pixel_lines([hline(200)])
    assert thick.counts.sum() == pytest.approx(4 * thin.counts.sum(), rel=0.15)


def test_normalise_empty_and_scales():
    zeros = np.zeros((10, 10), np.float32)
    v, ref = normalise(zeros, RenderConfig())
    assert not v.any() and ref == 0

    counts = np.zeros((10, 10), np.float32)
    counts[0, :] = np.arange(1, 11)
    for scale in ("log", "sqrt", "linear"):
        v, _ = normalise(counts, RenderConfig(scale=scale, reference_percentile=100))
        assert np.all(np.diff(v[0]) >= 0)
        assert v[0, -1] == pytest.approx(1.0)
        assert 0 < v[0, 0] < 1


def test_log_scale_keeps_single_use_visible_next_to_heavy_use():
    counts = np.zeros((10, 10), np.float32)
    counts[0, 0] = 1
    counts[1, 1] = 500
    v, _ = normalise(counts, RenderConfig(reference_percentile=100))
    assert v[0, 0] > 0.1
    rgb = get_colormap("fire")(v)
    assert rgb[0, 0].max() > 60  # visibly red, not black


def test_small_export_does_not_wash_out_to_white():
    counts = np.zeros((10, 10), np.float32)
    counts[0, :] = 1  # one activity
    v, _ = normalise(counts, RenderConfig())
    assert v.max() < 0.8


def test_output_dimensions_non_square():
    c = HeatmapCanvas(PixelTransform(Bounds(0, 0, 300, 150), 300, 150), 2.0, 2)
    c.add_pixel_lines([np.array([[10.0, 10.0], [290.0, 140.0]])])
    v, _ = normalise(c.counts, RenderConfig())
    img = colourise(v, get_colormap("fire"), RenderConfig())
    assert img.size == (300, 150)
    assert img.mode == "RGB"
    ov = overlay(v, get_colormap("fire"))
    assert ov.size == (300, 150) and ov.mode == "RGBA"
    alpha = np.asarray(ov)[..., 3]
    assert alpha[0, 299] == 0 and alpha.max() > 0


def test_background_is_black_and_custom_background():
    v = np.zeros((4, 4), np.float32)
    v[0, 0] = 1
    img = np.asarray(colourise(v, get_colormap("fire"), RenderConfig(glow_strength=0)))
    assert img[3, 3].tolist() == [0, 0, 0]
    assert img[0, 0].tolist() == [255, 255, 255]
    img = np.asarray(colourise(v, get_colormap("fire"), RenderConfig(glow_strength=0, background="#102030")))
    assert img[3, 3].tolist() == [16, 32, 48]


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_colormaps_dark_to_white_and_brightening(name):
    cmap = get_colormap(name)
    ramp = cmap(np.linspace(0, 1, 50)).astype(int)
    assert ramp[0].tolist() == [0, 0, 0]
    assert ramp[-1].tolist() == [255, 255, 255]
    luminance = ramp @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(luminance) >= -1)


def test_fire_gradient_goes_red_orange_yellow():
    cmap = get_colormap("fire")
    r, g, b = cmap(np.array([0.3]))[0]
    assert r > 150 and g < 60  # red
    r, g, b = cmap(np.array([0.62]))[0]
    assert r > 240 and 100 < g < 180 and b < 40  # orange
    r, g, b = cmap(np.array([0.8]))[0]
    assert r > 240 and g > 200 and b < 120  # yellow


def test_custom_colormap_validation():
    with pytest.raises(ValueError):
        Colormap([[0.0, "#000"]])
    with pytest.raises(ValueError):
        Colormap([[0.1, "#000"], [1.0, "#fff"]])
    with pytest.raises(ValueError):
        get_colormap("nope")
    cmap = get_colormap("ignored", [[0.0, "#000000"], [1.0, "#00ff00"]])
    assert cmap(np.array([1.0]))[0].tolist() == [0, 255, 0]
