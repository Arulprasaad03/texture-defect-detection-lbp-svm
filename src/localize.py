"""Region-level localization via patch SVMs (+ B2 Grad-CAM) (plan §5.7). Stage 'localize'."""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, gaussian_filter, label
from skimage.transform import resize
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from .data_loader import build_pool, load_mask, make_split
from .features_glcm import extract_glcm
from .features_lbp import lbp_codes, lbp_hist
from .preprocessing import load_rgb, preprocess
from .utils import output_root, set_seed

SIZE = 256
DIM = {"P1": 59, "P2": 177, "P3": 201}
METHODS = ["P1", "P2", "P3"]


def positions(size: int, window: int, stride: int) -> list:
    return [(y, x) for y in range(0, size - window + 1, stride) for x in range(0, size - window + 1, stride)]


def window_features(img_u8: np.ndarray, pos: list, window: int = 64) -> np.ndarray:
    """(n,201) raw P3 window features; P2 = [:, :177], P1 = [:, :59]. LBP maps computed once per image."""
    maps = [lbp_codes(img_u8, 8, r, "nri_uniform") for r in (1, 2, 3)]
    out = np.empty((len(pos), 201), np.float32)
    for i, (y, x) in enumerate(pos):
        out[i, :177] = np.concatenate([lbp_hist(m[y:y + window, x:x + window], 59) for m in maps])
        out[i, 177:] = extract_glcm(img_u8[y:y + window, x:x + window])
    return out


def train_patch_model(X, y, name: str, seed: int, k: int = 100) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if name == "P3":
        steps.append(("select", SelectKBest(partial(mutual_info_classif, random_state=seed), k=min(k, X.shape[1]))))
    steps.append(("svm", SVC(kernel="linear", C=1, class_weight="balanced", random_state=seed)))
    return Pipeline(steps).fit(X, y)


def heatmap(model, feats: np.ndarray, pos: list, window: int, sigma: float = 8, size: int = SIZE) -> np.ndarray:
    scores = model.decision_function(feats)
    acc = np.zeros((size, size), np.float64)
    cnt = np.zeros((size, size), np.float64)
    for s, (y, x) in zip(scores, pos):
        acc[y:y + window, x:x + window] += s
        cnt[y:y + window, x:x + window] += 1
    return gaussian_filter(acc / np.maximum(cnt, 1), sigma=sigma).astype(np.float32)


def fit_threshold(heats: np.ndarray, masks: np.ndarray) -> float:
    h, g = heats.ravel(), masks.ravel().astype(bool)
    taus = np.quantile(h, np.linspace(0.50, 0.995, 50))
    best, best_f1 = taus[0], -1.0
    for t in taus:
        p = h > t
        tp = np.sum(p & g)
        f1 = 2 * tp / max(p.sum() + g.sum(), 1)
        if f1 > best_f1:
            best, best_f1 = t, f1
    return float(best)


def regions(heat: np.ndarray, tau: float, min_px: int = 100) -> list:
    lab, n = label(heat > tau)
    objs = find_objects(lab)
    return [sl for i, sl in enumerate(objs, 1) if sl is not None and (lab[sl] == i).sum() >= min_px]


def localization_metrics(heats: np.ndarray, masks: np.ndarray, tau: float, out: int = 128) -> dict:
    H = np.stack([resize(h, (out, out), order=1, preserve_range=True, anti_aliasing=False) for h in heats])
    M = masks[:, ::SIZE // out, ::SIZE // out].astype(bool)
    auc = roc_auc_score(M.ravel(), H.ravel()) if 0 < M.sum() < M.size else float("nan")
    p = H > tau
    tp = np.sum(p & M)
    dice = 2 * tp / max(p.sum() + M.sum(), 1)
    iou = tp / max(np.sum(p | M), 1)
    return dict(pixel_auroc=float(auc), dice=float(dice), iou=float(iou), tau=float(tau))


def _load_b2(cfg, cat, seed):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18
    f = output_root() / "models" / f"b2_seed{seed}_{cat}.pt"
    if not f.exists():
        return None
    m = resnet18(weights=None)
    m.fc = nn.Linear(512, 2)
    m.load_state_dict(torch.load(f, map_location="cpu"))
    return m.eval()


def run(cfg, log, skip_cnn=False, **_):
    from .cnn_baselines import gradcam
    p = cfg.patch
    w = p["window"]
    seed = cfg.seeds[0]
    out = output_root()
    (out / "heatmaps").mkdir(exist_ok=True)
    pos_tr, pos_in = positions(SIZE, w, p["stride_train"]), positions(SIZE, w, p["stride_infer"])
    t5 = []
    for cat in cfg.categories:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        pool = build_pool(cfg.dataset_root, cat)
        tr, te = make_split(pool, seed, cfg.test_size)
        loc_tr, loc_val = train_test_split(tr, test_size=p["loc_val_frac"], stratify=tr.label, random_state=seed)

        def prep(row):
            return preprocess(load_rgb(row.path), SIZE)

        Xs, ys = [], []
        for row in tqdm(list(loc_tr.itertuples()), desc=f"patch-train {cat}", unit="img"):
            mask = load_mask(row._asdict(), SIZE)
            f = window_features(prep(row), pos_tr, w)
            frac = np.array([mask[y:y + w, x:x + w].mean() for y, x in pos_tr])
            keep = (frac >= p["pos_frac"]) | (frac == 0)
            Xs.append(f[keep])
            ys.append((frac[keep] >= p["pos_frac"]).astype(int))
        X, y = np.concatenate(Xs), np.concatenate(ys)
        pos_i, neg_i = np.where(y == 1)[0], np.where(y == 0)[0]
        neg_i = rng.choice(neg_i, min(len(neg_i), p["neg_ratio"] * max(len(pos_i), 1)), replace=False)
        sel = np.concatenate([pos_i, neg_i])
        models = {m: train_patch_model(X[sel][:, :DIM[m]], y[sel], m, seed, cfg.mi_k) for m in METHODS}

        def heats_for(df, with_cam):
            feats = [window_features(prep(r), pos_in, w) for r in tqdm(list(df.itertuples()), desc=f"infer {cat}",
                                                                      unit="img")]
            H = {m: np.stack([heatmap(models[m], f[:, :DIM[m]], pos_in, w, p["heat_sigma"]) for f in feats])
                 for m in METHODS}
            if with_cam is not None:
                H["B2"] = np.stack([gradcam(with_cam, load_rgb(r.path), SIZE) for r in df.itertuples()])
            return H

        b2 = None if skip_cnn else _load_b2(cfg, cat, seed)
        Hv, Ht = heats_for(loc_val, b2), heats_for(te, b2)
        Mv = np.stack([load_mask(r._asdict(), SIZE) for r in loc_val.itertuples()])
        Mt = np.stack([load_mask(r._asdict(), SIZE) for r in te.itertuples()])
        taus = {m: fit_threshold(Hv[m], Mv) for m in Hv}
        for m in Ht:
            t5.append(dict(category=cat, method="B2-GradCAM" if m == "B2" else m,
                           **localization_metrics(Ht[m], Mt, taus[m])))
        np.savez_compressed(out / "heatmaps" / f"{cat}.npz", ids=te.img_id.values,
                            **{f"heat_{m}": Ht[m] for m in Ht}, tau=np.array([taus.get(m, np.nan) for m in
                                                                              METHODS + ["B2"]]))
        log.info(f"localize {cat}: " + ", ".join(f"{r['method']} AUROC={r['pixel_auroc']:.3f}" for r in t5[-len(Ht):]))
    pd.DataFrame(t5).to_csv(out / "tables" / "T5_localization_metrics.csv", index=False, float_format="%.4f")
