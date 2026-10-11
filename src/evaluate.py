"""Metrics, latency helper, and the training+evaluation loop for P1-P3, B1, B2 (T1/T2/T3/T6/T9)."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

from .classifier import train_model
from .data_loader import build_pool, make_split
from .features_lbp import extract_cached
from .utils import output_root, set_seed

KIND = {"P1": "p1", "P2": "p2", "P3": "p3raw", "B1": "resnet18"}
METRIC_COLS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]


def metrics(y, score, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, score),
                pr_auc=average_precision_score(y, score), mcc=matthews_corrcoef(y, pred),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def measure_latency(fn, inputs, warmup: int = 20) -> dict:
    """fn(x) -> anything; batch size 1. Returns median/mean/p95 ms and FPS."""
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    ts = []
    for x in inputs:
        t0 = time.perf_counter_ns()
        fn(x)
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    ts = np.array(ts)
    return dict(median_ms=float(np.median(ts)), mean_ms=float(ts.mean()), p95_ms=float(np.percentile(ts, 95)),
                fps=float(1000 / np.median(ts)))


def _run_b2(cfg, tr, te, seed, cat, log):
    import torch
    from .cnn_baselines import finetune_resnet, to_tensor
    from .preprocessing import load_rgb
    c = cfg.cnn
    model, st = finetune_resnet(tr, seed, log, epochs=c["epochs"], batch_size=c["batch_size"], lr=c["lr"],
                                weight_decay=c["weight_decay"])
    torch.save(model.state_dict(), output_root() / "models" / f"b2_seed{seed}_{cat}.pt")
    scores = []
    paths = te["path"].tolist()
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            xb = torch.cat([to_tensor(load_rgb(p)) for p in paths[i:i + 16]])
            scores += torch.softmax(model(xb), dim=1)[:, 1].tolist()
    return np.array(scores), st


def evaluate_all(cfg, methods, log, tlog, rebuild=False, seeds=None) -> dict:
    out = output_root()
    seeds = seeds or cfg.seeds
    rows, t3, t6, preds, ep_rows = [], [], [], [], []
    for cat in cfg.categories:
        pool = build_pool(cfg.dataset_root, cat)
        for seed in seeds:
            tr, te = make_split(pool, seed, cfg.test_size)
            for m in methods:
                set_seed(seed)
                if m == "B2":
                    if seed not in cfg.b2_seeds:
                        continue
                    score, st = _run_b2(cfg, tr, te, seed, cat, log)
                    pred = (score >= 0.5).astype(int)
                    t3.append(dict(category=cat, method=m, seed=seed, read=0.0, preprocess=0.0, features=0.0,
                                   cnn_forward=0.0, scaler_selector_svm_grid=0.0,
                                   cnn_finetune=st["finetune_seconds"], total=st["finetune_seconds"]))
                    for e, (s_, l_) in enumerate(zip(st["epoch_seconds"], st["epoch_loss"])):
                        ep_rows.append(dict(category=cat, seed=seed, epoch=e + 1, seconds=s_, loss=l_))
                    info = dict(fit_seconds=st["finetune_seconds"])
                else:
                    kind = KIND[m]
                    Xtr, tp, tf, trd = extract_cached(cat, kind, cfg.image_size, tr, cfg.cache_root, rebuild)
                    Xte, *_ = extract_cached(cat, kind, cfg.image_size, te, cfg.cache_root, rebuild)
                    model, info = train_model(Xtr, tr.label.values, m, seed, tlog, cfg.cv_folds, cfg.mi_k)
                    score = model.decision_function(Xte)
                    pred = (score > 0).astype(int)
                    t3.append(dict(category=cat, method=m, seed=seed, read=trd.sum(), preprocess=tp.sum(),
                                   features=0.0 if m == "B1" else tf.sum(),
                                   cnn_forward=tf.sum() if m == "B1" else 0.0,
                                   scaler_selector_svm_grid=info["fit_seconds"], cnn_finetune=0.0,
                                   total=tp.sum() + tf.sum() + info["fit_seconds"]))
                    t6.append(dict(category=cat, method=m, seed=seed,
                                   **{k: v for k, v in info.items() if k != "fit_seconds"}))
                    joblib.dump(model, out / "models" / f"{m}_{cat}_seed{seed}.joblib")
                r = metrics(te.label.values, score, pred)
                rows.append(dict(category=cat, method=m, seed=seed, **r))
                preds.append(te[["img_id", "category", "defect_type", "label"]].assign(
                    method=m, seed=seed, score=score, pred=pred))
                log.info(f"{cat} {m} seed={seed} F1={r['f1']:.3f} AUC={r['roc_auc']:.3f} "
                         f"time={info['fit_seconds']:.1f}s")
    tbl = out / "tables"
    res = pd.DataFrame(rows)
    res.to_csv(tbl / "T2_raw_per_seed.csv", index=False, float_format="%.4f")
    pd.DataFrame(t3).to_csv(tbl / "T3_training_time.csv", index=False, float_format="%.4f")
    pd.DataFrame(t6).to_csv(tbl / "T6_best_params.csv", index=False, float_format="%.4f")
    pd.concat(preds).to_csv(tbl / "predictions.csv", index=False, float_format="%.6f")
    if ep_rows:
        pd.DataFrame(ep_rows).to_csv(tbl / "B2_epoch_log.csv", index=False, float_format="%.4f")
    write_t1_t2(res)
    write_t9(res)
    return dict(results=res)


def write_t1_t2(res: pd.DataFrame) -> None:
    tbl = output_root() / "tables"
    cols = METRIC_COLS
    t2 = res.groupby(["category", "method"])[cols].agg(["mean", "std"])
    t2.columns = [f"{a}_{b}" for a, b in t2.columns]
    t2.reset_index().to_csv(tbl / "T2_metrics_per_category.csv", index=False, float_format="%.4f")
    macro = res.groupby(["method", "seed"])[cols].mean().reset_index()
    g = macro.groupby("method")[cols].agg(["mean", "std"])
    t1 = pd.DataFrame(index=g.index)
    for c in cols:
        sd = g[(c, "std")]
        t1[c] = [f"{m:.4f}±{s:.4f}" if not np.isnan(s) else f"{m:.4f}±—" for m, s in zip(g[(c, "mean")], sd)]
        t1[c + "_mean"], t1[c + "_std"] = g[(c, "mean")], sd
    order = [m for m in ["P1", "P2", "P3", "B1", "B2"] if m in t1.index]
    t1.loc[order].reset_index().to_csv(tbl / "T1_metrics_per_method.csv", index=False, float_format="%.4f")


def write_t9(res: pd.DataFrame) -> None:
    rows = []
    piv = res.pivot_table(index=["category", "seed"], columns="method", values="f1")
    for a, b in (("P2", "P1"), ("P3", "P2"), ("P3", "B1")):
        if a not in piv or b not in piv:
            continue
        d = piv[[a, b]].dropna()
        if len(d) < 6 or (d[a] - d[b]).abs().sum() == 0:
            rows.append(dict(comparison=f"{a} vs {b}", n=len(d), W="", p="", mean_diff=float((d[a] - d[b]).mean()),
                             note=f"n<6 or identical: skipped (n={len(d)})"))
            continue
        w = wilcoxon(d[a], d[b], alternative="two-sided")
        rows.append(dict(comparison=f"{a} vs {b}", n=len(d), W=float(w.statistic), p=float(w.pvalue),
                         mean_diff=float((d[a] - d[b]).mean()), note=""))
    pd.DataFrame(rows).to_csv(output_root() / "tables" / "T9_stat_tests.csv", index=False, float_format="%.6f")
