import numpy as np

from src.features_glcm import extract_glcm
from src.features_lbp import extract_p1, extract_p2, extract_p2_literal, extract_p3_raw
from src.preprocessing import preprocess

rng = np.random.default_rng(0)
RGB = rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)
IMG = preprocess(RGB, 256)


def test_preprocess():
    assert IMG.dtype == np.uint8 and IMG.shape == (256, 256)
    assert IMG.min() == 0 and IMG.max() == 255


def test_dims():
    assert extract_p1(IMG).shape == (59,)
    assert extract_p2(IMG).shape == (177,)
    assert extract_p2_literal(IMG).shape == (54,)
    assert extract_glcm(IMG).shape == (24,)
    assert extract_p3_raw(IMG).shape == (201,)


def test_hist_sums_to_one():
    assert abs(extract_p1(IMG).sum() - 1) < 1e-5
    assert abs(extract_p2(IMG).sum() - 3) < 1e-4
