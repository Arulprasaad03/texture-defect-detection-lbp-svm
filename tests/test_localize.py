import numpy as np

from src.localize import heatmap, localization_metrics, positions, regions, train_patch_model, window_features


def test_window_count():
    assert len(positions(256, 64, 16)) == 169
    assert len(positions(256, 64, 32)) == 49


def test_heatmap_and_metrics():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    pos = positions(256, 64, 16)
    f = window_features(img, pos[:30], 64)
    assert f.shape == (30, 201)
    X = rng.normal(size=(60, 59))
    y = np.array([0, 1] * 30)
    model = train_patch_model(X, y, "P1", 0)
    h = heatmap(model, rng.normal(size=(169, 59)), pos, 64)
    assert h.shape == (256, 256) and h.dtype == np.float32
    mask = np.zeros((1, 256, 256), np.uint8)
    mask[0, 100:150, 100:150] = 1
    m = localization_metrics(h[None], mask, float(np.median(h)))
    assert 0 <= m["dice"] <= 1 and 0 <= m["iou"] <= 1 and np.isfinite(m["pixel_auroc"])
    assert isinstance(regions(h, float(np.median(h))), list)
