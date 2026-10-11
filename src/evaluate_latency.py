"""Stage 'latency': T4 inference latency, T8 resources, T0 environment + targets (plan §7.3, §8, §10)."""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

from .data_loader import build_pool, make_split
from .evaluate import measure_latency
from .features_glcm import extract_glcm
from .features_lbp import extract_p1, extract_p2, extract_p3_raw
from .preprocessing import load_rgb, preprocess
from .utils import cpu_name, output_root

EXT = {"P1": extract_p1, "P2": extract_p2, "P3": extract_p3_raw}
STAGE_COLS = ["preprocess_ms", "features_ms", "scaler_selector_ms", "svm_ms"]


def _items(cfg, seed):
    tests = []
    for cat in cfg.categories:
        _, te = make_split(build_pool(cfg.dataset_root, cat), seed, cfg.test_size)
        tests.append(te)
    te = pd.concat(tests, ignore_index=True)
    rng = np.random.default_rng(cfg.latency["sample_seed"])
    n = min(cfg.latency["n_images"], len(te))
    sel = te.iloc[rng.choice(len(te), n, replace=False)]
    t0 = time.perf_counter_ns()
    items = [(r.category, load_rgb(r.path)) for r in sel.itertuples()]
    t_read_ms = (time.perf_counter_ns() - t0) / 1e6 / n
    return items, t_read_ms


def _b2_models(cfg, seed):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18
    models = {}
    for cat in cfg.categories:
        f = output_root() / "models" / f"b2_seed{seed}_{cat}.pt"
        if not f.exists():
            return None
        m = resnet18(weights=None)
        m.fc = nn.Linear(512, 2)
        m.load_state_dict(torch.load(f, map_location="cpu"))
        models[cat] = m.eval()
    return models


def build(cfg, method, seed):
    """Return (stages, full_fn). stages = [(name, fn(cat, x))] where the last output is the decision."""
    import torch
    size = cfg.image_size
    base = method.replace("_allthreads", "")
    if base in EXT:
        models = {c: joblib.load(output_root() / "models" / f"{base}_{c}_seed{seed}.joblib") for c in cfg.categories}
        stages = [("preprocess_ms", lambda c, x: preprocess(x, size)),
                  ("features_ms", lambda c, x: EXT[base](x)[None]),
                  ("scaler_selector_ms", lambda c, x: models[c][:-1].transform(x)),
                  ("svm_ms", lambda c, x: models[c][-1].decision_function(x))]
    elif base == "B1":
        from torchvision.models import ResNet18_Weights, resnet18

        from .cnn_baselines import to_tensor
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        net.eval()
        models = {c: joblib.load(output_root() / "models" / f"B1_{c}_seed{seed}.joblib") for c in cfg.categories}

        def fwd(c, t):
            with torch.no_grad():
                return net(t).numpy()
        stages = [("preprocess_ms", lambda c, x: to_tensor(x)), ("features_ms", fwd),
                  ("scaler_selector_ms", lambda c, x: models[c][:-1].transform(x)),
                  ("svm_ms", lambda c, x: models[c][-1].decision_function(x))]
    elif base == "B2":
        from .cnn_baselines import to_tensor
        models = _b2_models(cfg, seed)
        if models is None:
            return None, None

        def fwd(c, t):
            with torch.no_grad():
                return torch.softmax(models[c](t), dim=1)[0, 1].item()
        stages = [("preprocess_ms", lambda c, x: to_tensor(x)), ("features_ms", fwd)]
    else:
        raise ValueError(method)

    def full(item):
        c, x = item
        for _, fn in stages:
            x = fn(c, x)
        return x
    return stages, full


def _breakdown(stages, items, warmup):
    acc = {n: [] for n, _ in stages}
    for k, item in enumerate(list(items[:warmup]) + list(items)):
        c, x = item
        for n, fn in stages:
            t0 = time.perf_counter_ns()
            x = fn(c, x)
            dt = (time.perf_counter_ns() - t0) / 1e6
            if k >= warmup:
                acc[n].append(dt)
    return {n: float(np.median(v)) for n, v in acc.items()}


def _rss_child(config_path, method):
    import torch

    from .config import load_config
    from .utils import set_output_root
    cfg = load_config(config_path)
    set_output_root(cfg.output_root)
    torch.set_num_threads(os.cpu_count() if method.endswith("_allthreads") else 1)
    seed = cfg.seeds[0]
    items, _ = _items(cfg, seed)
    _, full = build(cfg, method, seed)
    measure_latency(full, items, cfg.latency["warmup"])
    p = psutil.Process().memory_info()
    print("RSS_MB=%.2f" % (getattr(p, "peak_wset", p.rss) / 1024 ** 2))


def _rss(config_path, method) -> float:
    try:
        r = subprocess.run([sys.executable, "-m", "src.evaluate_latency", "--config", str(config_path),
                            "--method", method], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[1], timeout=1800)
        for line in r.stdout.splitlines():
            if line.startswith("RSS_MB="):
                return float(line.split("=")[1])
    except Exception:
        pass
    return float("nan")


def run(cfg, log, skip_cnn=False, config_path=None, **_):
    import sklearn
    import torch
    out = output_root()
    tbl = out / "tables"
    seed = cfg.seeds[0]
    torch.set_num_threads(1)
    items, t_read_ms = _items(cfg, seed)
    methods = ["P1", "P2", "P3"] + ([] if skip_cnn else ["B1", "B2", "B2_allthreads"])
    rows = []
    for m in methods:
        stages, full = build(cfg, m, seed)
        if stages is None:
            log.info(f"{m}: models missing, skipped")
            continue
        torch.set_num_threads(os.cpu_count() if m.endswith("_allthreads") else 1)
        lat = measure_latency(full, items, cfg.latency["warmup"])
        bd = _breakdown(stages, items, cfg.latency["warmup"])
        torch.set_num_threads(1)
        row = dict(method=m, **lat, **{c: bd.get(c, 0.0) for c in STAGE_COLS}, t_read_ms=t_read_ms,
                   peak_rss_mb=_rss(config_path, m) if config_path else float("nan"), n_timed=len(items))
        rows.append(row)
        log.info(f"{m}: median {lat['median_ms']:.2f} ms, {lat['fps']:.1f} FPS")
    pd.DataFrame(rows).to_csv(tbl / "T4_inference_latency.csv", index=False, float_format="%.4f")
    write_environment(tbl)
    write_resources(cfg, tbl, seed, skip_cnn)
    write_targets(tbl)


def write_environment(tbl):
    import sklearn
    import torch
    import torchvision
    vm = psutil.virtual_memory()
    rows = dict(cpu=cpu_name(), cpu_processor_string=platform.processor(), cores_logical=os.cpu_count(),
                cores_physical=psutil.cpu_count(logical=False), ram_gb=round(vm.total / 1024 ** 3, 2),
                os=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                sklearn=sklearn.__version__, torch=torch.__version__, torchvision=torchvision.__version__,
                pandas=pd.__version__)
    pd.DataFrame(list(rows.items()), columns=["key", "value"]).to_csv(tbl / "T0_environment.csv", index=False)


def write_resources(cfg, tbl, seed, skip_cnn):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18

    from .cnn_baselines import model_stats
    t6 = pd.read_csv(tbl / "T6_best_params.csv")
    mdir = output_root() / "models"

    def svm_info(m):
        sv = t6[t6.method == m].n_support_vectors.mean()
        sz = np.mean([f.stat().st_size for f in mdir.glob(f"{m}_*_seed{seed}.joblib")]) / 1024 ** 2
        return sv, sz
    rows = []
    for m, dim in (("P1", 59), ("P2", 177), ("P3", "201 raw -> 100 selected")):
        sv, sz = svm_info(m)
        rows.append(dict(method=m, feature_dim=dim, n_params_or_SVs=f"{sv:.1f} SVs (mean)", model_size_MB=sz,
                         gflops="N/A", gpu_required="no", training_data_needed="labelled normal + defect images"))
    if not skip_cnn:
        net = resnet18(weights=None)
        net.fc = nn.Identity()
        st = model_stats(net)
        sv, sz = svm_info("B1")
        rows.append(dict(method="B1", feature_dim=512, n_params_or_SVs=f"{st['n_params']} params + {sv:.1f} SVs",
                         model_size_MB=st["fp32_size_MB"] + sz, gflops=st["gflops"], gpu_required="no",
                         training_data_needed="labelled normal + defect images"))
        b2 = _b2_models(cfg, seed)
        if b2:
            st = model_stats(next(iter(b2.values())))
            rows.append(dict(method="B2", feature_dim=512, n_params_or_SVs=f"{st['n_params']} params",
                             model_size_MB=st["fp32_size_MB"], gflops=st["gflops"], gpu_required="no",
                             training_data_needed="labelled normal + defect images"))
    pd.DataFrame(rows).to_csv(tbl / "T8_resources.csv", index=False, float_format="%.4f")


def write_targets(tbl):
    rows = []
    t2 = pd.read_csv(tbl / "T2_metrics_per_category.csv")
    for cat, g in t2[t2.method.isin(["P1", "P2", "P3"])].groupby("category"):
        b = g.loc[g.f1_mean.idxmax()]
        rows.append(dict(target=f"F1 > 0.90 ({cat}, best of P1-P3 = {b.method})", value=b.f1_mean, threshold=0.90,
                         met="yes" if b.f1_mean > 0.90 else "no"))
    t4 = pd.read_csv(tbl / "T4_inference_latency.csv")
    for r in t4[t4.method.isin(["P1", "P2", "P3"])].itertuples():
        rows.append(dict(target=f"latency < 30 ms ({r.method})", value=r.median_ms, threshold=30,
                         met="yes" if r.median_ms < 30 else "no"))
    t1 = pd.read_csv(tbl / "T1_metrics_per_method.csv").set_index("method")
    if {"P1", "P2"} <= set(t1.index):
        gain = t1.loc["P2", "f1_mean"] - t1.loc["P1", "f1_mean"]
        rows.append(dict(target="P2 - P1 macro F1 gain in [0.08, 0.15]", value=gain, threshold="0.08-0.15",
                         met="yes" if 0.08 <= gain <= 0.15 else "no"))
    pd.DataFrame(rows).to_csv(tbl / "T0_targets.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    a = ap.parse_args()
    _rss_child(a.config, a.method)
