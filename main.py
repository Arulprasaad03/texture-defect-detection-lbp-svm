"""CLI entry. Run after `. .\\scripts\\env_d.ps1` and activating .venv.

python main.py --config configs/quick.yaml [--stage STAGE] [--rebuild-cache] [--skip-cnn] [--seeds 42 43]
"""
from __future__ import annotations

import argparse
import importlib
import platform
import time

import pandas as pd

from src.config import load_config
from src.data_loader import dataset_stats_table, split_manifest
from src.timing import TimingLog
from src.utils import assert_on_d, check_environment, cpu_name, get_logger, set_output_root

STAGES = ["data", "features", "train", "eval", "latency", "localize", "explain", "ablate", "figures",
          "gallery", "report"]


def stage_data(cfg, log, **_):
    tbl = cfg.output_root / "tables"
    dataset_stats_table(cfg.dataset_root, cfg.categories).to_csv(tbl / "T10_dataset_stats.csv", index=False)
    for s in cfg.seeds:
        split_manifest(cfg.dataset_root, cfg.categories, s, cfg.test_size).to_csv(
            tbl / f"split_manifest_seed{s}.csv", index=False)


def stage_features(cfg, log, rebuild=False, skip_cnn=False, **_):
    from src.data_loader import build_pool
    from src.features_lbp import extract_cached
    kinds = ["p1", "p2", "p3raw"] + ([] if skip_cnn else ["resnet18"])
    for cat in cfg.categories:
        pool = build_pool(cfg.dataset_root, cat)
        for k in kinds:
            extract_cached(cat, k, cfg.image_size, pool, cfg.cache_root, rebuild)


def stage_train(cfg, log, rebuild=False, skip_cnn=False, seeds=None, **_):
    from src.evaluate import evaluate_all
    methods = ["P1", "P2", "P3"] + ([] if skip_cnn else ["B1", "B2"])
    evaluate_all(cfg, methods, log, TimingLog(), rebuild, seeds)


def _optional(name):
    def run(cfg, log, **kw):
        try:
            mod = importlib.import_module(f"src.{name}")
        except ImportError:
            log.info(f"stage needs src/{name}.py which is not implemented yet - skipped")
            return
        mod.run(cfg, log, **kw)
    return run


DISPATCH = {"data": stage_data, "features": stage_features, "train": stage_train, "eval": stage_train,
            "latency": _optional("evaluate_latency"), "localize": _optional("localize"),
            "explain": _optional("explain"), "ablate": _optional("ablations"),
            "figures": _optional("visualize"), "gallery": _optional("report"), "report": _optional("report")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--stage", choices=["all"] + STAGES, default="all")
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+")
    a = ap.parse_args()

    check_environment()
    cfg = load_config(a.config)
    if a.seeds:
        cfg.seeds = a.seeds
    for p in (cfg.output_root, cfg.cache_root, cfg.dataset_root):
        assert_on_d(p)
    set_output_root(cfg.output_root, f"CPU: {cpu_name()} | 1 thread | n_seeds={len(cfg.seeds)}")
    log = get_logger(cfg.output_root)
    if a.rebuild_cache and a.stage in ("all", "features"):
        for f in cfg.cache_root.glob(f"*_{cfg.image_size}.npz"):
            f.unlink()
    full = ["data", "features", "train", "latency", "localize", "explain", "ablate", "figures", "gallery", "report"]
    if cfg.image_size != 256:  # size ablation (A5): classification + latency only, no CNNs
        full = ["data", "features", "train", "latency"]
        a.skip_cnn = True
    if not cfg.run_ablations:
        full.remove("ablate") if "ablate" in full else None
    todo = full if a.stage == "all" else [a.stage]
    for st in todo:
        log.info(f"=== stage: {st} ===")
        t0 = time.perf_counter()
        DISPATCH[st](cfg, log, rebuild=False, skip_cnn=a.skip_cnn, seeds=cfg.seeds, stage=st,
                     config_path=a.config)
        log.info(f"stage {st} done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
