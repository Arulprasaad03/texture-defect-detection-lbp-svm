"""LBP features + the shared on-disk feature cache (extract_cached)."""
from __future__ import annotations

import time

import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm

from .preprocessing import load_rgb, preprocess
from .utils import cache_path

PAD = 3
N_BINS_NRI = {8: 59}
KINDS = ("p1", "p2", "p3raw", "p2literal", "glcm", "resnet18")


def n_bins(P: int, method: str) -> int:
    return P * (P - 1) + 3 if method == "nri_uniform" else P + 2


def lbp_codes(img_u8: np.ndarray, P: int, R: int, method: str) -> np.ndarray:
    padded = np.pad(img_u8, PAD, mode="reflect")
    codes = local_binary_pattern(padded, P, R, method=method)
    return codes[PAD:-PAD, PAD:-PAD].astype(np.uint16)


def lbp_hist(codes: np.ndarray, nb: int) -> np.ndarray:
    return (np.bincount(codes.ravel(), minlength=nb)[:nb] / codes.size).astype(np.float32)


def _h(img, P, R, method):
    return lbp_hist(lbp_codes(img, P, R, method), n_bins(P, method))


def extract_p1(img: np.ndarray) -> np.ndarray:
    return _h(img, 8, 1, "nri_uniform")


def extract_p2(img: np.ndarray) -> np.ndarray:
    return np.concatenate([_h(img, 8, r, "nri_uniform") for r in (1, 2, 3)])


def extract_p2_literal(img: np.ndarray) -> np.ndarray:
    return np.concatenate([_h(img, P, R, "uniform") for P, R in ((8, 1), (16, 2), (24, 3))])


def lbp_feature_names(kind: str = "p2") -> list:
    if kind == "p1":
        return [f"LBP_R1_bin{i}" for i in range(59)]
    if kind == "p2literal":
        return [f"LBPlit_{P}_{R}_bin{i}" for P, R in ((8, 1), (16, 2), (24, 3)) for i in range(P + 2)]
    return [f"LBP_R{r}_bin{i}" for r in (1, 2, 3) for i in range(59)]


def extract_p3_raw(img: np.ndarray) -> np.ndarray:
    from .features_glcm import extract_glcm
    return np.concatenate([extract_p2(img), extract_glcm(img)])


def _extractor(kind: str):
    from .features_glcm import extract_glcm
    return {"p1": extract_p1, "p2": extract_p2, "p3raw": extract_p3_raw,
            "p2literal": extract_p2_literal, "glcm": extract_glcm}[kind]


def feature_names(kind: str) -> list:
    from .features_glcm import glcm_feature_names
    if kind == "p3raw":
        return lbp_feature_names("p2") + glcm_feature_names()
    if kind == "glcm":
        return glcm_feature_names()
    if kind == "resnet18":
        return [f"resnet_{i}" for i in range(512)]
    return lbp_feature_names(kind)


def extract_cached(category: str, kind: str, size: int, df, cache_root, rebuild: bool = False):
    """Return (X, t_pre, t_feat, t_read) aligned with df rows. Computes and caches missing ids.
    Each kind is timed in its own pass (times are never shared between kinds)."""
    assert kind in KINDS, kind
    f = cache_path(cache_root, category, kind, size)
    store = {}
    if f.exists() and not rebuild:
        z = np.load(f, allow_pickle=False)
        for i, id_ in enumerate(z["ids"]):
            store[str(id_)] = (z["X"][i], z["t_read"][i], z["t_pre"][i], z["t_feat"][i])
    missing = df[~df["img_id"].isin(store)]
    if len(missing):
        if kind == "resnet18":
            from .cnn_baselines import resnet_features
            X, t_pre, t_feat, t_read = resnet_features(missing["path"].tolist())
            for j, id_ in enumerate(missing["img_id"]):
                store[id_] = (X[j], t_read[j], t_pre[j], t_feat[j])
        else:
            fn = _extractor(kind)
            for id_, path in tqdm(zip(missing["img_id"], missing["path"]), total=len(missing),
                                  desc=f"{kind} {category}", unit="img"):
                t0 = time.perf_counter_ns()
                rgb = load_rgb(path)
                t1 = time.perf_counter_ns()
                g = preprocess(rgb, size)
                t2 = time.perf_counter_ns()
                x = fn(g)
                t3 = time.perf_counter_ns()
                store[id_] = (x.astype(np.float32), (t1 - t0) / 1e9, (t2 - t1) / 1e9, (t3 - t2) / 1e9)
        ids = np.array(list(store.keys()))
        np.savez(f, ids=ids, X=np.stack([store[i][0] for i in ids]),
                 t_read=np.array([store[i][1] for i in ids]), t_pre=np.array([store[i][2] for i in ids]),
                 t_feat=np.array([store[i][3] for i in ids]), names=np.array(feature_names(kind)))
    rows = [store[i] for i in df["img_id"]]
    return (np.stack([r[0] for r in rows]), np.array([r[2] for r in rows]),
            np.array([r[3] for r in rows]), np.array([r[1] for r in rows]))
