from pathlib import Path

import pytest

from src.data_loader import build_pool, make_split

ROOT = Path("D:/MV Project/texture-defect-detection-lbp-svm/dataset/MVTec AD")
CATS = ["carpet", "grid", "leather", "tile", "wood"]
SIZES = {"carpet": 397, "grid": 342, "leather": 369, "tile": 347, "wood": 326}


@pytest.mark.skipif(not ROOT.exists(), reason="dataset missing")
def test_pools_and_splits():
    total = normal = defect = 0
    for c in CATS:
        pool = build_pool(ROOT, c)
        assert len(pool) == SIZES[c]
        total += len(pool)
        normal += int((pool.label == 0).sum())
        defect += int((pool.label == 1).sum())
        assert all(Path(p).exists() for p in pool[pool.label == 1].mask_path)
        for seed in (42, 43):
            tr, te = make_split(pool, seed)
            assert not set(tr.img_id) & set(te.img_id)
            assert set(tr.label) == {0, 1} and set(te.label) == {0, 1}
            assert len(tr) + len(te) == len(pool)
    assert (total, normal, defect) == (1781, 1399, 382)
