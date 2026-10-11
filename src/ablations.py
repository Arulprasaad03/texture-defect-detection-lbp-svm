"""Ablations A1-A4, A6 on cached features (plan §9). Stage 'ablate'. A5 is read from outputs/size512."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .classifier import train_model
from .data_loader import build_pool, make_split
from .evaluate import measure_latency
from .features_lbp import extract_cached
from .utils import output_root

COLS = ["ablation", "variant", "category", "seed", "f1", "roc_auc", "dim", "fit_seconds", "latency_ms"]


def _fit_eval(Xtr, ytr, Xte, yte, name, seed, cv, k=100, selector=None):
    cv = max(2, min(cv, int(np.bincount(ytr).min())))
    model, info = train_model(Xtr, ytr, name, seed, None, cv, k, selector)
    score = model.decision_function(Xte)
    f1 = f1_score(yte, (score > 0).astype(int), zero_division=0)
    auc = roc_auc_score(yte, score)
    rows = Xte[:30]
    lat = measure_latency(lambda r: model.decision_function(r[None]), list(rows), warmup=3)["median_ms"]
    return dict(f1=f1, roc_auc=auc, fit_seconds=info["fit_seconds"], latency_ms=lat)


def run(cfg, log, skip_cnn=False, **_):
    rows = []
    for cat in cfg.categories:
        pool = build_pool(cfg.dataset_root, cat)
        for seed in cfg.seeds:
            tr, te = make_split(pool, seed, cfg.test_size)
            ytr, yte = tr.label.values, te.label.values
            F = {k: (extract_cached(cat, k, cfg.image_size, tr, cfg.cache_root)[0],
                     extract_cached(cat, k, cfg.image_size, te, cfg.cache_root)[0])
                 for k in ("p1", "p2", "p3raw", "p2literal", "glcm")}

            def add(abl, var, Xtr, Xte, name="P2", dim=None, k=100, selector=None, ytr_=ytr, **kw):
                r = _fit_eval(Xtr, ytr_, Xte, yte, name, seed, cfg.cv_folds, k, selector)
                rows.append(dict(ablation=abl, variant=var, category=cat, seed=seed, dim=dim or Xtr.shape[1], **r))
                log.info(f"{abl} {var} {cat} s{seed} F1={r['f1']:.3f}")

            p2tr, p2te = F["p2"]
            # A1: LBP scales
            for var, sl in (("R1", slice(0, 59)), ("R2", slice(59, 118)), ("R3", slice(118, 177)),
                            ("R1+R2", slice(0, 118)), ("R1+R2+R3 (P2)", slice(0, 177))):
                add("A1", var, p2tr[:, sl], p2te[:, sl])
            add("A1", "literal (8,1)(16,2)(24,3) uniform 54-d", *F["p2literal"])
            # A2: feature groups
            add("A2", "LBP-only (P2)", p2tr, p2te)
            add("A2", "GLCM-only", *F["glcm"])
            add("A2", "P2+GLCM 201 no selection", *F["p3raw"])
            add("A2", "P3 (MI k=100)", *F["p3raw"], name="P3", dim=100)
            # A3: selector variants on 201
            add("A3", "none", *F["p3raw"])
            add("A3", "SelectKBest-MI k=100 (P3)", *F["p3raw"], name="P3", dim=100)
            add("A3", "PCA 95% variance", *F["p3raw"], selector=PCA(n_components=0.95, svd_solver="full",
                                                                      random_state=seed))
            # A4: k sweep
            for k in (10, 25, 50, 100, 150, 201):
                add("A4", f"k={k}", *F["p3raw"], name="P3", k=k, dim=k)
            # A6: learning curve
            for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
                idx = np.arange(len(ytr)) if frac == 1.0 else train_test_split(
                    np.arange(len(ytr)), train_size=frac, stratify=ytr, random_state=seed)[0]
                for m, k in (("P1", "p1"), ("P2", "p2"), ("P3", "p3raw")) + (() if skip_cnn else (("B1", "resnet18"),)):
                    Xtr_all = F[k][0] if k in F else extract_cached(cat, k, cfg.image_size, tr, cfg.cache_root)[0]
                    Xte_k = F[k][1] if k in F else extract_cached(cat, k, cfg.image_size, te, cfg.cache_root)[0]
                    add("A6", f"{m} frac={frac}", Xtr_all[idx], Xte_k, name="P3" if m == "P3" else "P2",
                        dim=Xtr_all.shape[1], ytr_=ytr[idx])
    pd.DataFrame(rows)[COLS].to_csv(output_root() / "tables" / "T7_ablations.csv", index=False, float_format="%.4f")
