"""Training-free LBP rarity map, group permutation importance, per-bin Cohen's d (plan §5.8, §5.9)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.metrics import f1_score

from .data_loader import build_pool, make_split
from .features_glcm import PROPS
from .features_lbp import extract_cached, lbp_codes
from .utils import output_root

GROUPS_P2 = {"LBP_R1": slice(0, 59), "LBP_R2": slice(59, 118), "LBP_R3": slice(118, 177)}
GROUPS_P3 = {**GROUPS_P2, **{f"GLCM_{p}": slice(177 + 4 * i, 181 + 4 * i) for i, p in enumerate(PROPS)}}


def normal_mean_hists(X_p2_train_normal: np.ndarray) -> list:
    """Mean LBP histogram per scale (59 bins each) from training-split NORMAL images."""
    m = X_p2_train_normal.mean(axis=0)
    return [m[0:59], m[59:118], m[118:177]]


def rarity_map(img_u8: np.ndarray, hists: list) -> np.ndarray:
    maps = []
    for r, h in zip((1, 2, 3), hists):
        codes = lbp_codes(img_u8, 8, r, "nri_uniform")
        maps.append(-np.log(h[codes] + 1e-6))
    return gaussian_filter(np.mean(maps, axis=0), sigma=4).astype(np.float32)


def group_permutation_importance(models: dict, data: dict, groups: dict, n_repeats: int = 20, seed: int = 42):
    """models/data keyed by category; data[cat] = (X_raw_test, y). Pooled F1 drop per group."""
    rng = np.random.default_rng(seed)

    def pooled_f1(perm_group=None, perms=None):
        ys, ps = [], []
        for cat, (X, y) in data.items():
            Xp = X.copy()
            if perm_group is not None:
                Xp[:, perm_group] = X[perms[cat]][:, perm_group]  # same row permutation inside the group
            ys.append(y)
            ps.append(models[cat].predict(Xp))
        return f1_score(np.concatenate(ys), np.concatenate(ps), zero_division=0)

    base = pooled_f1()
    rows = []
    for g, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            perms = {c: rng.permutation(len(data[c][1])) for c in data}
            drops.append(base - pooled_f1(cols, perms))
        rows.append(dict(group=g, mean_drop=float(np.mean(drops)), std_drop=float(np.std(drops)), baseline_f1=base))
    return pd.DataFrame(rows)


def bin_cohens_d(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b = X[y == 0], X[y == 1]
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(axis=0, ddof=1) + (n2 - 1) * b.var(axis=0, ddof=1)) / max(n1 + n2 - 2, 1))
    return (b.mean(axis=0) - a.mean(axis=0)) / (sp + 1e-12)


def run(cfg, log, skip_cnn=False, **_):
    seed = cfg.seeds[0]
    out = output_root() / "tables"
    perm_rows, d_rows = [], []
    p2m, p3m, d2, d3 = {}, {}, {}, {}
    for cat in cfg.categories:
        pool = build_pool(cfg.dataset_root, cat)
        tr, te = make_split(pool, seed, cfg.test_size)
        Xtr, *_a = extract_cached(cat, "p2", cfg.image_size, tr, cfg.cache_root)
        d = bin_cohens_d(Xtr, tr.label.values)
        d_rows += [dict(category=cat, bin=i, cohens_d=float(v)) for i, v in enumerate(d)]
        X3, *_b = extract_cached(cat, "p3raw", cfg.image_size, te, cfg.cache_root)
        d3[cat] = (X3, te.label.values)
        d2[cat] = (X3[:, :177], te.label.values)
        p2m[cat] = joblib.load(output_root() / "models" / f"P2_{cat}_seed{seed}.joblib")
        p3m[cat] = joblib.load(output_root() / "models" / f"P3_{cat}_seed{seed}.joblib")
    pd.DataFrame(d_rows).to_csv(out / "lbp_bin_cohens_d.csv", index=False, float_format="%.6f")
    for name, models, data, groups in (("P2", p2m, d2, GROUPS_P2), ("P3", p3m, d3, GROUPS_P3)):
        df = group_permutation_importance(models, data, groups, 20, seed)
        perm_rows.append(df.assign(pipeline=name))
    pd.concat(perm_rows).to_csv(out / "permutation_importance.csv", index=False, float_format="%.6f")
    log.info("explain: permutation importance + Cohen's d saved")
